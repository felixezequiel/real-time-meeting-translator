use async_trait::async_trait;
use shared::{AudioChunk, Language, PipelineStage, StageError, TextSegment};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use thiserror::Error;
use tracing;

use serde::{Deserialize, Serialize};

#[derive(Debug, Error)]
pub enum SttError {
    #[error("STT bridge not started")]
    BridgeNotStarted,

    #[error("Transcription failed: {0}")]
    TranscriptionFailed(String),
}

#[derive(Serialize)]
struct SttRequest {
    samples: Vec<f32>,
    sample_rate: u32,
    language: String,
}

#[derive(Deserialize)]
struct SttResponse {
    text: String,
}

pub struct WhisperStt {
    bridge_script_path: PathBuf,
    model_path: PathBuf,
    process: Option<Mutex<SttBridgeProcess>>,
    language: Language,
}

struct SttBridgeProcess {
    child: Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl WhisperStt {
    pub fn new(bridge_script_path: PathBuf, model_path: PathBuf, language: Language) -> Self {
        Self {
            bridge_script_path,
            model_path,
            process: None,
            language,
        }
    }

    pub fn transcribe(&self, chunk: &AudioChunk) -> Result<TextSegment, SttError> {
        let process_mutex = self
            .process
            .as_ref()
            .ok_or(SttError::BridgeNotStarted)?;

        let mut bridge = process_mutex
            .lock()
            .map_err(|e| SttError::TranscriptionFailed(format!("Lock poisoned: {}", e)))?;

        let request = SttRequest {
            samples: chunk.samples.clone(),
            sample_rate: chunk.sample_rate,
            language: self.language.whisper_code().to_string(),
        };

        let request_json = serde_json::to_string(&request)
            .map_err(|e| SttError::TranscriptionFailed(e.to_string()))?;

        writeln!(bridge.stdin, "{}", request_json)
            .map_err(|e| SttError::TranscriptionFailed(e.to_string()))?;
        bridge
            .stdin
            .flush()
            .map_err(|e| SttError::TranscriptionFailed(e.to_string()))?;

        let mut response_line = String::new();
        bridge
            .stdout
            .read_line(&mut response_line)
            .map_err(|e| SttError::TranscriptionFailed(e.to_string()))?;

        let response: SttResponse = serde_json::from_str(&response_line)
            .map_err(|e| SttError::TranscriptionFailed(format!("Invalid response: {}: {}", e, response_line)))?;

        let trimmed = response.text.trim().to_string();
        tracing::debug!("STT result: \"{}\"", trimmed);

        Ok(TextSegment::new(trimmed, self.language))
    }
}

#[async_trait]
impl PipelineStage for WhisperStt {
    fn name(&self) -> &str {
        "whisper-stt"
    }

    async fn initialize(&mut self) -> Result<(), StageError> {
        let script_path = self.bridge_script_path.to_string_lossy().to_string();
        let model_path = self.model_path.to_string_lossy().to_string();
        tracing::info!("Starting STT bridge: {} with model {}", script_path, model_path);

        let mut child = Command::new("python")
            .arg(&script_path)
            .arg("--model")
            .arg(&model_path)
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
            .map_err(|e| StageError::NotInitialized(format!("STT bridge startup failed: {}", e)))?;

        if !ready_line.trim().contains("ready") {
            return Err(StageError::NotInitialized(format!(
                "STT bridge did not signal ready: {}",
                ready_line
            )));
        }

        self.process = Some(Mutex::new(SttBridgeProcess {
            child,
            stdin,
            stdout: reader,
        }));

        tracing::info!("STT bridge ready");
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<(), StageError> {
        if let Some(process_mutex) = self.process.take() {
            if let Ok(mut bridge) = process_mutex.into_inner() {
                let _ = bridge.child.kill();
                let _ = bridge.child.wait();
            }
        }
        tracing::info!("STT bridge stopped");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stt_request_serializes_correctly() {
        let request = SttRequest {
            samples: vec![0.1, 0.2, 0.3],
            sample_rate: 16000,
            language: "en".to_string(),
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("samples"));
        assert!(json.contains("16000"));
    }

    #[test]
    fn stt_response_deserializes_correctly() {
        let json = r#"{"text": "hello world"}"#;
        let response: SttResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.text, "hello world");
    }

    #[test]
    fn stt_without_bridge_returns_error() {
        let stt = WhisperStt::new(
            PathBuf::from("nonexistent.py"),
            PathBuf::from("nonexistent.bin"),
            Language::English,
        );
        let chunk = AudioChunk::new(vec![0.0; 16000], 16000, 1);
        let result = stt.transcribe(&chunk);
        assert!(result.is_err());
    }
}
