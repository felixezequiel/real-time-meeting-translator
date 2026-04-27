use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use shared::{PipelineStage, StageError};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use thiserror::Error;
use tracing;

const F32_SIZE_BYTES: usize = 4;

#[derive(Debug, Error)]
pub enum SeparationError {
    #[error("Separation bridge not started")]
    BridgeNotStarted,

    #[error("Separation request failed: {0}")]
    RequestFailed(String),

    #[error("Invalid response from bridge: {0}")]
    InvalidResponse(String),
}

#[derive(Serialize)]
struct SeparateRequestHeader {
    num_samples: u32,
    sample_rate: u32,
}

#[derive(Deserialize)]
struct SeparateResponseHeader {
    sample_rate: u32,
    num_samples: u32,
    rms_a: f32,
    rms_b: f32,
}

/// Result of separating one mixed-mono chunk into two channels.
#[derive(Debug, Clone)]
pub struct Separated {
    pub channel_a: Vec<f32>,
    pub channel_b: Vec<f32>,
    pub sample_rate: u32,
    /// RMS of channel A — the caller can use this to skip downstream
    /// processing when the channel is essentially silent (single-
    /// speaker chunk where the model placed everything on the OTHER
    /// channel).
    pub rms_a: f32,
    pub rms_b: f32,
}

/// Client for `scripts/separation_bridge.py`.
///
/// Serialises concurrent callers via a `Mutex` so the binary protocol
/// over stdin/stdout doesn't interleave bytes from different chunks.
/// One bridge process per `Sepformer` instance.
pub struct Sepformer {
    bridge_script_path: PathBuf,
    process: Option<Mutex<SeparationBridgeProcess>>,
    /// Set to `true` after the first request fails — we don't want a
    /// dead bridge to block the audio capture loop forever, so we
    /// short-circuit subsequent calls to "no separation" once dead.
    dead: Arc<AtomicBool>,
}

struct SeparationBridgeProcess {
    child: Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl Sepformer {
    pub fn new(bridge_script_path: PathBuf) -> Self {
        Self {
            bridge_script_path,
            process: None,
            dead: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Returns `Ok(None)` when the bridge has been marked dead — the
    /// caller should fall back to passing the mono mix straight through.
    pub fn separate(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Option<Separated>, SeparationError> {
        if self.dead.load(Ordering::Acquire) {
            return Ok(None);
        }

        let process_mutex = self
            .process
            .as_ref()
            .ok_or(SeparationError::BridgeNotStarted)?;

        let mut bridge = process_mutex.lock().map_err(|e| {
            SeparationError::RequestFailed(format!("Lock poisoned: {}", e))
        })?;

        let header = SeparateRequestHeader {
            num_samples: samples.len() as u32,
            sample_rate,
        };
        let header_json = serde_json::to_string(&header)
            .map_err(|e| SeparationError::RequestFailed(e.to_string()))?;

        let result: Result<Separated, SeparationError> = (|| {
            writeln!(bridge.stdin, "{}", header_json)
                .map_err(|e| SeparationError::RequestFailed(e.to_string()))?;

            let mut pcm_bytes = Vec::with_capacity(samples.len() * F32_SIZE_BYTES);
            for sample in samples {
                pcm_bytes.extend_from_slice(&sample.to_le_bytes());
            }
            bridge
                .stdin
                .write_all(&pcm_bytes)
                .map_err(|e| SeparationError::RequestFailed(e.to_string()))?;
            bridge
                .stdin
                .flush()
                .map_err(|e| SeparationError::RequestFailed(e.to_string()))?;

            let mut response_line = String::new();
            bridge
                .stdout
                .read_line(&mut response_line)
                .map_err(|e| SeparationError::RequestFailed(e.to_string()))?;

            let response: SeparateResponseHeader = serde_json::from_str(response_line.trim())
                .map_err(|e| {
                    SeparationError::InvalidResponse(format!("{}: {}", e, response_line))
                })?;

            let n = response.num_samples as usize;
            let channel_a = read_f32_samples(&mut bridge.stdout, n)
                .map_err(|e| SeparationError::RequestFailed(e.to_string()))?;
            let channel_b = read_f32_samples(&mut bridge.stdout, n)
                .map_err(|e| SeparationError::RequestFailed(e.to_string()))?;

            Ok(Separated {
                channel_a,
                channel_b,
                sample_rate: response.sample_rate,
                rms_a: response.rms_a,
                rms_b: response.rms_b,
            })
        })();

        match result {
            Ok(separated) => Ok(Some(separated)),
            Err(e) => {
                self.dead.store(true, Ordering::Release);
                tracing::error!(
                    "Separation bridge marked dead after failure: {}. \
                     Pipeline continues without source separation.",
                    e
                );
                Ok(None)
            }
        }
    }
}

fn read_f32_samples<R: Read>(
    reader: &mut BufReader<R>,
    num_samples: usize,
) -> std::io::Result<Vec<f32>> {
    let mut bytes = vec![0u8; num_samples * F32_SIZE_BYTES];
    reader.read_exact(&mut bytes)?;
    let mut samples = Vec::with_capacity(num_samples);
    for chunk in bytes.chunks_exact(F32_SIZE_BYTES) {
        samples.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(samples)
}

#[async_trait]
impl PipelineStage for Sepformer {
    fn name(&self) -> &str {
        "sepformer"
    }

    async fn initialize(&mut self) -> Result<(), StageError> {
        let script_path = self.bridge_script_path.to_string_lossy().to_string();
        tracing::info!("Starting separation bridge: {}", script_path);

        let python = shared::find_python();
        let mut child = Command::new(&python)
            .arg(&script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| StageError::NotInitialized(format!(
                "Failed to start Python (tried '{}'). Error: {}", python, e
            )))?;

        let stdin = child.stdin.take().ok_or_else(|| {
            StageError::NotInitialized("Failed to capture stdin".to_string())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            StageError::NotInitialized("Failed to capture stdout".to_string())
        })?;

        let mut reader = BufReader::new(stdout);
        let mut ready_line = String::new();
        reader
            .read_line(&mut ready_line)
            .map_err(|e| StageError::NotInitialized(format!("Sep bridge startup failed: {}", e)))?;

        if !ready_line.trim().contains("ready") {
            return Err(StageError::NotInitialized(format!(
                "Separation bridge did not signal ready: {}",
                ready_line
            )));
        }

        self.process = Some(Mutex::new(SeparationBridgeProcess {
            child,
            stdin,
            stdout: reader,
        }));

        tracing::info!("Separation bridge ready");
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<(), StageError> {
        if let Some(process_mutex) = self.process.take() {
            if let Ok(mut bridge) = process_mutex.into_inner() {
                let _ = bridge.child.kill();
                let _ = bridge.child.wait();
            }
        }
        tracing::info!("Separation bridge stopped");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_header_serialises() {
        let header = SeparateRequestHeader { num_samples: 8000, sample_rate: 16000 };
        let json = serde_json::to_string(&header).unwrap();
        assert!(json.contains("8000"));
        assert!(json.contains("16000"));
    }

    #[test]
    fn response_header_deserialises() {
        let json = r#"{"sample_rate": 16000, "num_samples": 8000, "rms_a": 0.05, "rms_b": 0.001}"#;
        let r: SeparateResponseHeader = serde_json::from_str(json).unwrap();
        assert_eq!(r.num_samples, 8000);
        assert!((r.rms_a - 0.05).abs() < 1e-6);
    }

    #[test]
    fn separate_errors_without_bridge() {
        let s = Sepformer::new(PathBuf::from("nonexistent.py"));
        let samples = vec![0.0f32; 8000];
        let result = s.separate(&samples, 16000);
        assert!(result.is_err());
    }
}
