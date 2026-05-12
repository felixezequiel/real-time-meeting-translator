//! Sentence Boundary Detection client.
//!
//! Wraps the Python `scripts/sbd_bridge.py` (spaCy under the hood) so
//! the V2 accumulator can ask "is the buffered text complete enough
//! to flush to the translator?" instead of trusting only punctuation
//! regex. See `docs/pipeline-architecture.md` for the role of this
//! layer in the overall flow.
//!
//! The bridge is a long-lived Python subprocess kept alive across
//! every call. A background reader thread pushes each JSON response
//! through a bounded channel, and `split()` reads with a timeout so
//! a hung spaCy parse never blocks the pipeline forever — same
//! resilience pattern as the TTS bridge (`crates/tts`).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use shared::{Language, PipelineStage, StageError};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc as std_mpsc;
use std::sync::Mutex;
use std::time::Duration;
use thiserror::Error;
use tracing;

/// Maximum time `split()` will wait for a response from the bridge
/// before giving up. spaCy's small models parse a ~100-character span
/// in <50 ms even on cold cache; 2 s covers warm-up plus comfortable
/// headroom on a loaded machine. Past this, the accumulator's own
/// safety nets (`MAX_HOLD`, `MAX_WORDS`) take over.
const SBD_RESPONSE_TIMEOUT_MS: u64 = 2_000;

/// Bounded channel between the bridge reader thread and the caller.
/// Bigger than 1 so a burst of late responses from a previous timed-out
/// request doesn't deadlock the reader; small enough that stale events
/// can't accumulate to GBs.
const SBD_BRIDGE_CHANNEL_CAPACITY: usize = 8;

#[derive(Debug, Error)]
pub enum SbdError {
    #[error("SBD bridge not started")]
    BridgeNotStarted,

    #[error("SBD request failed: {0}")]
    RequestFailed(String),

    #[error("Invalid response from SBD bridge: {0}")]
    InvalidResponse(String),

    #[error("SBD request timed out after {0} ms")]
    Timeout(u64),
}

/// Outcome of a single boundary-detection call. `complete` is the
/// portion of the input that the bridge considers ready for the
/// translator; `rest` is the suffix the accumulator should keep
/// buffering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SbdResult {
    pub complete: String,
    pub rest: String,
}

#[derive(Serialize)]
struct SbdRequest<'a> {
    text: &'a str,
    language: &'a str,
}

#[derive(Deserialize)]
struct SbdResponse {
    #[serde(default)]
    complete: String,
    #[serde(default)]
    rest: String,
}

enum SbdBridgeEvent {
    Response(SbdResponse),
    Error(String),
    Eof,
}

/// Client for `scripts/sbd_bridge.py`.
pub struct SbdService {
    bridge_script_path: PathBuf,
    process: Option<Mutex<SbdBridgeProcess>>,
}

struct SbdBridgeProcess {
    child: Child,
    stdin: std::process::ChildStdin,
    frame_rx: std_mpsc::Receiver<SbdBridgeEvent>,
}

impl SbdService {
    pub fn new(bridge_script_path: PathBuf) -> Self {
        Self {
            bridge_script_path,
            process: None,
        }
    }

    /// Ask the bridge to split `text` into the part that's ready to
    /// flush and the part still pending. `Ok(SbdResult { complete: "",
    /// rest: text })` means "hold everything". A timeout, IO failure
    /// or invalid JSON returns an error; the caller is expected to
    /// fall back to its own safety nets (MAX_HOLD, MAX_WORDS) and
    /// keep the text buffered.
    pub fn split(&self, text: &str, language: Language) -> Result<SbdResult, SbdError> {
        let process_mutex = self.process.as_ref().ok_or(SbdError::BridgeNotStarted)?;
        let mut bridge = process_mutex
            .lock()
            .map_err(|e| SbdError::RequestFailed(format!("Lock poisoned: {}", e)))?;

        // Drain stale events left over from a previous request that
        // timed out on our side while the bridge eventually finished.
        // Without this the next call would consume that stale response
        // as if it belonged to the new request.
        let mut drained = 0_usize;
        while bridge.frame_rx.try_recv().is_ok() {
            drained += 1;
        }
        if drained > 0 {
            tracing::debug!("SBD: drained {} stale event(s) before new request", drained);
        }

        let lang_code = match language {
            Language::Portuguese => "pt",
            Language::English => "en",
        };
        let request = SbdRequest {
            text,
            language: lang_code,
        };
        let request_json =
            serde_json::to_string(&request).map_err(|e| SbdError::RequestFailed(e.to_string()))?;
        writeln!(bridge.stdin, "{}", request_json)
            .map_err(|e| SbdError::RequestFailed(e.to_string()))?;
        bridge
            .stdin
            .flush()
            .map_err(|e| SbdError::RequestFailed(e.to_string()))?;

        let timeout = Duration::from_millis(SBD_RESPONSE_TIMEOUT_MS);
        let event = bridge.frame_rx.recv_timeout(timeout).map_err(|e| match e {
            std_mpsc::RecvTimeoutError::Timeout => SbdError::Timeout(SBD_RESPONSE_TIMEOUT_MS),
            std_mpsc::RecvTimeoutError::Disconnected => SbdError::RequestFailed(
                "SBD bridge reader thread disconnected".to_string(),
            ),
        })?;

        match event {
            SbdBridgeEvent::Response(r) => Ok(SbdResult {
                complete: r.complete,
                rest: r.rest,
            }),
            SbdBridgeEvent::Error(detail) => Err(SbdError::InvalidResponse(detail)),
            SbdBridgeEvent::Eof => Err(SbdError::RequestFailed(
                "SBD bridge stdout closed mid-request".to_string(),
            )),
        }
    }
}

/// Background reader: owns the BufReader, pushes one event per JSON
/// response (or a terminal event on EOF/error) into the channel.
fn bridge_reader_loop(
    mut reader: BufReader<std::process::ChildStdout>,
    tx: std_mpsc::SyncSender<SbdBridgeEvent>,
) {
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                let _ = tx.send(SbdBridgeEvent::Eof);
                return;
            }
            Err(e) => {
                let _ = tx.send(SbdBridgeEvent::Error(format!("read_line: {}", e)));
                return;
            }
            Ok(_) => {}
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let response: SbdResponse = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(SbdBridgeEvent::Error(format!("invalid JSON: {}", e)));
                return;
            }
        };
        if tx.send(SbdBridgeEvent::Response(response)).is_err() {
            // Receiver dropped — pipeline shutting down.
            return;
        }
    }
}

#[async_trait]
impl PipelineStage for SbdService {
    fn name(&self) -> &str {
        "sbd"
    }

    async fn initialize(&mut self) -> Result<(), StageError> {
        let script_path = self.bridge_script_path.to_string_lossy().to_string();
        tracing::info!("Starting SBD bridge: {}", script_path);

        let python = shared::find_python();
        let mut child = Command::new(&python)
            .arg(&script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| {
                StageError::NotInitialized(format!(
                    "Failed to start Python for SBD (tried '{}'): {}",
                    python, e
                ))
            })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            StageError::NotInitialized("Failed to capture SBD stdin".to_string())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            StageError::NotInitialized("Failed to capture SBD stdout".to_string())
        })?;

        let mut reader = BufReader::new(stdout);
        let mut ready_line = String::new();
        reader
            .read_line(&mut ready_line)
            .map_err(|e| StageError::NotInitialized(format!("SBD bridge startup failed: {}", e)))?;
        if !ready_line.trim().contains("ready") {
            return Err(StageError::NotInitialized(format!(
                "SBD bridge did not signal ready: {}",
                ready_line
            )));
        }

        let (frame_tx, frame_rx) =
            std_mpsc::sync_channel::<SbdBridgeEvent>(SBD_BRIDGE_CHANNEL_CAPACITY);
        std::thread::Builder::new()
            .name("sbd-bridge-reader".to_string())
            .spawn(move || bridge_reader_loop(reader, frame_tx))
            .map_err(|e| {
                StageError::NotInitialized(format!("Failed to spawn SBD reader thread: {}", e))
            })?;

        self.process = Some(Mutex::new(SbdBridgeProcess {
            child,
            stdin,
            frame_rx,
        }));
        tracing::info!("SBD bridge ready");
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<(), StageError> {
        if let Some(process_mutex) = self.process.take() {
            if let Ok(mut bridge) = process_mutex.into_inner() {
                let _ = bridge.child.kill();
                let _ = bridge.child.wait();
            }
        }
        tracing::info!("SBD bridge stopped");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_without_bridge_returns_error() {
        let service = SbdService::new(PathBuf::from("not-used.py"));
        let err = service
            .split("hello world.", Language::English)
            .expect_err("should fail without bridge");
        assert!(matches!(err, SbdError::BridgeNotStarted));
    }

    #[test]
    fn sbd_result_equality() {
        let a = SbdResult {
            complete: "Hi.".to_string(),
            rest: "world".to_string(),
        };
        let b = SbdResult {
            complete: "Hi.".to_string(),
            rest: "world".to_string(),
        };
        assert_eq!(a, b);
    }
}
