use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use shared::{PipelineStage, StageError};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use thiserror::Error;
use tracing;

const F32_SIZE_BYTES: usize = 4;

#[derive(Debug, Error)]
pub enum DiarizationError {
    #[error("Diarization bridge not started")]
    BridgeNotStarted,

    #[error("Diarization request failed: {0}")]
    RequestFailed(String),

    #[error("Invalid response from bridge: {0}")]
    InvalidResponse(String),
}

#[derive(Serialize)]
struct IdentifyRequestHeader {
    num_samples: u32,
    sample_rate: u32,
}

#[derive(Deserialize)]
struct IdentifyResponse {
    speaker_id: i32,
    #[serde(default)]
    is_new: bool,
    #[serde(default)]
    f0_hz: f32,
}

/// Result of identifying a speaker for an audio window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerIdentification {
    /// Stable integer id the bridge assigned to the speaker. Same id
    /// across subsequent chunks of the same person.
    pub speaker_id: u32,
    /// True iff this speaker was not known before this call.
    pub is_new: bool,
    /// Mean F0 (Hz) of voiced frames in the chunk that produced this
    /// identification. 0.0 when no voiced frames could be detected.
    /// Used by the pipeline to maintain a running per-speaker pitch
    /// profile that drives the TTS pitch shifter.
    pub f0_hz: f32,
}

/// Client for `scripts/diarization_bridge.py`.
///
/// Serialises concurrent callers through an internal `Mutex` so PCM bytes
/// from different chunks don't interleave over stdin/stdout. Clone the
/// `Arc<OnlineDiarizer>` to share between pipelines.
pub struct OnlineDiarizer {
    bridge_script_path: PathBuf,
    process: Option<Mutex<DiarizationBridgeProcess>>,
    /// Flipped to `true` the first time a request fails — protects the
    /// pipeline from spamming a broken bridge with chunks forever. Further
    /// `identify()` calls short-circuit to `Ok(None)` and the pipeline
    /// continues without diarization.
    dead: Arc<AtomicBool>,
}

struct DiarizationBridgeProcess {
    child: Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl OnlineDiarizer {
    pub fn new(bridge_script_path: PathBuf) -> Self {
        Self {
            bridge_script_path,
            process: None,
            dead: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Return the speaker id for `samples` (f32 mono at `sample_rate`).
    ///
    /// Returns `Ok(None)` when the bridge decided the window was too short
    /// or silent for a stable embedding — callers should fall back to the
    /// last known speaker (or the default voice) in that case.
    pub fn identify(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Option<SpeakerIdentification>, DiarizationError> {
        // Short-circuit: once the bridge has died, further attempts just
        // flood the logs and block the pipeline on broken pipes. Accept
        // that diarization is off for the rest of the session.
        if self.dead.load(Ordering::Acquire) {
            return Ok(None);
        }

        let process_mutex = self
            .process
            .as_ref()
            .ok_or(DiarizationError::BridgeNotStarted)?;

        let mut bridge = process_mutex.lock().map_err(|e| {
            DiarizationError::RequestFailed(format!("Lock poisoned: {}", e))
        })?;

        let header = IdentifyRequestHeader {
            num_samples: samples.len() as u32,
            sample_rate,
        };
        let header_json = serde_json::to_string(&header)
            .map_err(|e| DiarizationError::RequestFailed(e.to_string()))?;

        let result: Result<SpeakerIdentification, DiarizationError> = (|| {
            writeln!(bridge.stdin, "{}", header_json)
                .map_err(|e| DiarizationError::RequestFailed(e.to_string()))?;

            // Binary f32 LE payload right after the header line.
            let mut pcm_bytes = Vec::with_capacity(samples.len() * F32_SIZE_BYTES);
            for sample in samples {
                pcm_bytes.extend_from_slice(&sample.to_le_bytes());
            }
            bridge
                .stdin
                .write_all(&pcm_bytes)
                .map_err(|e| DiarizationError::RequestFailed(e.to_string()))?;
            bridge
                .stdin
                .flush()
                .map_err(|e| DiarizationError::RequestFailed(e.to_string()))?;

            let mut response_line = String::new();
            bridge
                .stdout
                .read_line(&mut response_line)
                .map_err(|e| DiarizationError::RequestFailed(e.to_string()))?;

            let response: IdentifyResponse = serde_json::from_str(response_line.trim())
                .map_err(|e| DiarizationError::InvalidResponse(format!("{}: {}", e, response_line)))?;

            if response.speaker_id < 0 {
                // Not an error — bridge just says "no confident id for this
                // window". Encode with a sentinel and translate to Ok(None)
                // one layer up.
                Ok(SpeakerIdentification {
                    speaker_id: u32::MAX,
                    is_new: false,
                    f0_hz: response.f0_hz,
                })
            } else {
                Ok(SpeakerIdentification {
                    speaker_id: response.speaker_id as u32,
                    is_new: response.is_new,
                    f0_hz: response.f0_hz,
                })
            }
        })();

        match result {
            Ok(id) if id.speaker_id == u32::MAX => Ok(None),
            Ok(id) => Ok(Some(id)),
            Err(e) => {
                // One failure is enough — mark the bridge dead so the
                // pipeline stops hammering a zombie process.
                self.dead.store(true, Ordering::Release);
                tracing::error!(
                    "Diarization bridge marked dead after failure: {}. \
                     Pipeline continues without speaker identification.",
                    e
                );
                Ok(None)
            }
        }
    }
}

#[async_trait]
impl PipelineStage for OnlineDiarizer {
    fn name(&self) -> &str {
        "online-diarizer"
    }

    async fn initialize(&mut self) -> Result<(), StageError> {
        let script_path = self.bridge_script_path.to_string_lossy().to_string();
        tracing::info!("Starting diarization bridge: {}", script_path);

        let python = shared::find_python();
        let mut child = Command::new(&python)
            .arg(&script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Inherit the parent's stderr: Python tracebacks and Resemblyzer
            // init logs then appear directly in our console. Piping stderr
            // into a forwarder thread seemed to race with stdout's
            // `read_line` on the wait for startup.
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| StageError::NotInitialized(
                format!("Failed to start Python (tried '{}'). Error: {}", python, e),
            ))?;

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
            .map_err(|e| StageError::NotInitialized(format!("Diarization startup failed: {}", e)))?;

        if !ready_line.trim().contains("ready") {
            return Err(StageError::NotInitialized(format!(
                "Diarization bridge did not signal ready: {}",
                ready_line
            )));
        }

        self.process = Some(Mutex::new(DiarizationBridgeProcess {
            child,
            stdin,
            stdout: reader,
        }));

        tracing::info!("Diarization bridge ready");
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<(), StageError> {
        if let Some(process_mutex) = self.process.take() {
            if let Ok(mut bridge) = process_mutex.into_inner() {
                let _ = bridge.child.kill();
                let _ = bridge.child.wait();
            }
        }
        tracing::info!("Diarization bridge stopped");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_header_serializes() {
        let header = IdentifyRequestHeader { num_samples: 16000, sample_rate: 16000 };
        let json = serde_json::to_string(&header).unwrap();
        assert!(json.contains("16000"));
        assert!(json.contains("num_samples"));
    }

    #[test]
    fn response_deserializes_new_speaker() {
        let json = r#"{"speaker_id": 3, "is_new": true}"#;
        let r: IdentifyResponse = serde_json::from_str(json).unwrap();
        assert_eq!(r.speaker_id, 3);
        assert!(r.is_new);
    }

    #[test]
    fn response_deserializes_unknown_as_negative() {
        let json = r#"{"speaker_id": -1, "is_new": false}"#;
        let r: IdentifyResponse = serde_json::from_str(json).unwrap();
        assert_eq!(r.speaker_id, -1);
        assert!(!r.is_new);
        assert_eq!(r.f0_hz, 0.0);
    }

    #[test]
    fn response_carries_f0_when_present() {
        let json = r#"{"speaker_id": 0, "is_new": false, "f0_hz": 142.3}"#;
        let r: IdentifyResponse = serde_json::from_str(json).unwrap();
        assert!((r.f0_hz - 142.3).abs() < 1e-3);
    }

    #[test]
    fn identify_errors_without_bridge() {
        let d = OnlineDiarizer::new(PathBuf::from("nonexistent.py"));
        let samples = vec![0.0f32; 16000];
        let result = d.identify(&samples, 16000);
        assert!(result.is_err());
    }
}
