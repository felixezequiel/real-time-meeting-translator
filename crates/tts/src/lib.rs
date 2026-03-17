use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use shared::{AudioChunk, Language, PipelineStage, StageError, TextSegment};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use thiserror::Error;
use tracing;

const MONO_CHANNELS: u16 = 1;

#[derive(Debug, Error)]
pub enum TtsError {
    #[error("TTS bridge not started")]
    BridgeNotStarted,

    #[error("Synthesis failed: {0}")]
    SynthesisFailed(String),
}

#[derive(Serialize)]
struct TtsRequest {
    text: String,
    language: String,
}

#[derive(Deserialize)]
struct TtsResponse {
    samples: Vec<f32>,
    sample_rate: u32,
}

pub struct PiperTts {
    bridge_script_path: PathBuf,
    process: Option<Mutex<TtsBridgeProcess>>,
    language: Language,
}

struct TtsBridgeProcess {
    child: Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl PiperTts {
    pub fn new(bridge_script_path: PathBuf, language: Language) -> Self {
        Self {
            bridge_script_path,
            process: None,
            language,
        }
    }

    pub fn synthesize(&self, segment: &TextSegment) -> Result<AudioChunk, TtsError> {
        let process_mutex = self
            .process
            .as_ref()
            .ok_or(TtsError::BridgeNotStarted)?;

        let mut bridge = process_mutex.lock().map_err(|e| {
            TtsError::SynthesisFailed(format!("Lock poisoned: {}", e))
        })?;

        let request = TtsRequest {
            text: segment.text.clone(),
            language: match self.language {
                Language::Portuguese => "pt-br".to_string(),
                Language::English => "en-us".to_string(),
            },
        };

        let request_json =
            serde_json::to_string(&request).map_err(|e| TtsError::SynthesisFailed(e.to_string()))?;

        writeln!(bridge.stdin, "{}", request_json)
            .map_err(|e| TtsError::SynthesisFailed(e.to_string()))?;
        bridge
            .stdin
            .flush()
            .map_err(|e| TtsError::SynthesisFailed(e.to_string()))?;

        let mut response_line = String::new();
        bridge
            .stdout
            .read_line(&mut response_line)
            .map_err(|e| TtsError::SynthesisFailed(e.to_string()))?;

        let response: TtsResponse = serde_json::from_str(&response_line)
            .map_err(|e| TtsError::SynthesisFailed(format!("Invalid TTS response: {}", e)))?;

        tracing::debug!(
            "TTS synthesized {} samples at {}Hz",
            response.samples.len(),
            response.sample_rate
        );

        Ok(AudioChunk::new(
            response.samples,
            response.sample_rate,
            MONO_CHANNELS,
        ))
    }
}

#[async_trait]
impl PipelineStage for PiperTts {
    fn name(&self) -> &str {
        "piper-tts"
    }

    async fn initialize(&mut self) -> Result<(), StageError> {
        let script_path = self.bridge_script_path.to_string_lossy().to_string();
        tracing::info!("Starting TTS bridge: {}", script_path);

        let mut child = Command::new("python")
            .arg(&script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| StageError::NotInitialized(format!("Failed to start Python: {}", e)))?;

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
            .map_err(|e| StageError::NotInitialized(format!("TTS bridge startup failed: {}", e)))?;

        if !ready_line.trim().contains("ready") {
            return Err(StageError::NotInitialized(format!(
                "TTS bridge did not signal ready: {}",
                ready_line
            )));
        }

        self.process = Some(Mutex::new(TtsBridgeProcess {
            child,
            stdin,
            stdout: reader,
        }));

        tracing::info!("TTS bridge ready");
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<(), StageError> {
        if let Some(process_mutex) = self.process.take() {
            if let Ok(mut bridge) = process_mutex.into_inner() {
                let _ = bridge.child.kill();
                let _ = bridge.child.wait();
            }
        }
        tracing::info!("TTS bridge stopped");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tts_request_serializes_correctly() {
        let request = TtsRequest {
            text: "olá mundo".to_string(),
            language: "pt-br".to_string(),
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("olá mundo"));
        assert!(json.contains("pt-br"));
    }

    #[test]
    fn tts_response_deserializes_correctly() {
        let json = r#"{"samples": [0.1, 0.2, 0.3], "sample_rate": 22050}"#;
        let response: TtsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.samples.len(), 3);
        assert_eq!(response.sample_rate, 22050);
    }

    #[test]
    fn tts_without_bridge_returns_error() {
        let tts = PiperTts::new(PathBuf::from("nonexistent.py"), Language::Portuguese);
        let segment = TextSegment::new("olá".to_string(), Language::Portuguese);
        let result = tts.synthesize(&segment);
        assert!(result.is_err());
    }
}
