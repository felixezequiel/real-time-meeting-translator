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

/// Default tail size in samples (50 ms at 16 kHz). Long enough to
/// capture a stable loudness estimate, short enough that the tracker
/// reacts within one chunk to a real speaker change.
pub const DEFAULT_TRACKER_TAIL_SAMPLES: usize = 800;

/// Floor used when comparing log-RMS, to keep silence comparable
/// without producing −inf.
const TRACKER_RMS_FLOOR: f32 = 1e-6;

/// Aligned output of `PermutationTracker::align`.
///
/// `swapped` reports whether the tracker re-routed the input pair —
/// useful for telemetry. The waveform fields and RMS values always
/// reflect what was forwarded downstream (post-swap when applicable).
#[derive(Debug, Clone)]
pub struct AlignedPair {
    pub channel_a: Vec<f32>,
    pub channel_b: Vec<f32>,
    pub rms_a: f32,
    pub rms_b: f32,
    pub swapped: bool,
}

/// Stateful permutation tracker for Sepformer outputs (ADR 0012).
///
/// Sepformer is permutation-invariant per chunk: across consecutive
/// chunks the model can flip the meaning of `channel_a` / `channel_b`,
/// fragmenting the streaming-STT context. The tracker keeps a short
/// tail of each previously-published channel and, for every new pair,
/// decides whether to swap them so the same speaker stays bound to
/// the same downstream pipeline.
///
/// Continuity metric: log-RMS of the head window vs the stored tail.
/// Energy alone is enough for the dominant single-speaker case;
/// see ADR 0012 for the full rationale and limitations.
pub struct PermutationTracker {
    tail_a: Vec<f32>,
    tail_b: Vec<f32>,
    tail_samples: usize,
}

impl PermutationTracker {
    pub fn new(tail_samples: usize) -> Self {
        Self {
            tail_a: Vec::new(),
            tail_b: Vec::new(),
            tail_samples,
        }
    }

    /// Re-align `(ch_a, ch_b)` so the loud channel stays on the same
    /// downstream pipeline as the previous chunk. On the first call
    /// (no history) the input pair is returned unchanged. The tails
    /// are updated from the post-swap pair so future decisions reflect
    /// what was actually published.
    pub fn align(&mut self, ch_a: Vec<f32>, ch_b: Vec<f32>) -> AlignedPair {
        let head_window = self
            .tail_samples
            .min(ch_a.len())
            .min(ch_b.len());
        let head_a_rms = rms(&ch_a[..head_window]);
        let head_b_rms = rms(&ch_b[..head_window]);

        let swapped = if self.has_history() {
            let tail_a_rms = rms(&self.tail_a);
            let tail_b_rms = rms(&self.tail_b);
            let same_cost = log_rms_distance(tail_a_rms, head_a_rms)
                          + log_rms_distance(tail_b_rms, head_b_rms);
            let swap_cost = log_rms_distance(tail_a_rms, head_b_rms)
                          + log_rms_distance(tail_b_rms, head_a_rms);
            swap_cost < same_cost
        } else {
            false
        };

        let (out_a, out_b, rms_a, rms_b) = if swapped {
            (ch_b, ch_a, head_b_rms, head_a_rms)
        } else {
            (ch_a, ch_b, head_a_rms, head_b_rms)
        };

        self.tail_a = tail_of(&out_a, self.tail_samples);
        self.tail_b = tail_of(&out_b, self.tail_samples);

        AlignedPair {
            channel_a: out_a,
            channel_b: out_b,
            rms_a,
            rms_b,
            swapped,
        }
    }

    fn has_history(&self) -> bool {
        !self.tail_a.is_empty() && !self.tail_b.is_empty()
    }
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

fn log_rms_distance(a: f32, b: f32) -> f32 {
    ((a + TRACKER_RMS_FLOOR).ln() - (b + TRACKER_RMS_FLOOR).ln()).abs()
}

fn tail_of(samples: &[f32], n: usize) -> Vec<f32> {
    let start = samples.len().saturating_sub(n);
    samples[start..].to_vec()
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

    // ─── PermutationTracker tests (ADR 0012) ─────────────────────────────────

    fn loud_chunk(value: f32, len: usize) -> Vec<f32> {
        vec![value; len]
    }

    #[test]
    fn tracker_does_not_swap_on_first_chunk() {
        let mut tracker = PermutationTracker::new(800);
        let result = tracker.align(loud_chunk(0.1, 1000), loud_chunk(0.001, 1000));
        assert!(!result.swapped, "first chunk has no history to swap against");
        assert!((result.rms_a - 0.1).abs() < 1e-3);
        assert!((result.rms_b - 0.001).abs() < 1e-3);
    }

    #[test]
    fn tracker_keeps_pair_when_active_channel_continues() {
        let mut tracker = PermutationTracker::new(800);
        tracker.align(loud_chunk(0.1, 1000), loud_chunk(0.001, 1000));
        // Sepformer kept the same assignment: A loud, B silent again.
        let result = tracker.align(loud_chunk(0.12, 1000), loud_chunk(0.0008, 1000));
        assert!(!result.swapped);
    }

    #[test]
    fn tracker_swaps_when_active_channel_flips() {
        let mut tracker = PermutationTracker::new(800);
        tracker.align(loud_chunk(0.1, 1000), loud_chunk(0.001, 1000));
        // Sepformer flipped: now B has the speaker, A has residual silence.
        let result = tracker.align(loud_chunk(0.0008, 1000), loud_chunk(0.12, 1000));
        assert!(result.swapped, "tracker should re-route loud channel back to A");
        assert!(result.rms_a > 0.05, "channel_a now carries the loud waveform");
        assert!(result.rms_b < 0.01, "channel_b carries the silent residual");
    }

    #[test]
    fn tracker_state_follows_published_channels_after_swap() {
        let mut tracker = PermutationTracker::new(800);
        tracker.align(loud_chunk(0.1, 1000), loud_chunk(0.001, 1000));
        // Sepformer flipped — tracker swaps so A stays loud downstream.
        tracker.align(loud_chunk(0.0008, 1000), loud_chunk(0.12, 1000));
        // Sepformer flips back to its original assignment (raw A loud again).
        // Because tracker's stored "A tail" is loud and "B tail" is silent,
        // this raw input matches the same assignment — no swap expected.
        let result = tracker.align(loud_chunk(0.1, 1000), loud_chunk(0.001, 1000));
        assert!(!result.swapped);
    }

    #[test]
    fn tracker_handles_inputs_shorter_than_tail_window() {
        let mut tracker = PermutationTracker::new(800);
        // 100-sample inputs (much smaller than 800-sample tail target).
        let first = tracker.align(loud_chunk(0.1, 100), loud_chunk(0.001, 100));
        assert!(!first.swapped);
        // Next chunk flips: tracker should still detect the swap using
        // whatever tail was captured.
        let second = tracker.align(loud_chunk(0.001, 100), loud_chunk(0.1, 100));
        assert!(second.swapped);
    }

    #[test]
    fn tracker_align_is_idempotent_on_silent_chunks() {
        let mut tracker = PermutationTracker::new(800);
        let result = tracker.align(loud_chunk(0.0, 1000), loud_chunk(0.0, 1000));
        // No history, no signal — should not swap and should not panic.
        assert!(!result.swapped);
        assert_eq!(result.rms_a, 0.0);
        assert_eq!(result.rms_b, 0.0);
    }
}
