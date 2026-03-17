use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use shared::{Language, PipelineStage, StageError, TextSegment, TranslationDirection};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use thiserror::Error;
use tracing;

#[derive(Debug, Error)]
pub enum TranslationError {
    #[error("Translation bridge not started")]
    BridgeNotStarted,

    #[error("Failed to start Python bridge: {0}")]
    BridgeStartFailed(String),

    #[error("Translation request failed: {0}")]
    RequestFailed(String),

    #[error("Invalid response from bridge: {0}")]
    InvalidResponse(String),
}

#[derive(Serialize)]
struct TranslateRequest {
    text: String,
    source_lang: String,
    target_lang: String,
}

#[derive(Deserialize)]
struct TranslateResponse {
    translated: String,
}

pub struct OpusMtTranslator {
    bridge_script_path: PathBuf,
    process: Option<Mutex<BridgeProcess>>,
    direction: TranslationDirection,
}

struct BridgeProcess {
    child: Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl OpusMtTranslator {
    pub fn new(bridge_script_path: PathBuf, direction: TranslationDirection) -> Self {
        Self {
            bridge_script_path,
            process: None,
            direction,
        }
    }

    pub fn translate(&self, segment: &TextSegment) -> Result<TextSegment, TranslationError> {
        let process_mutex = self
            .process
            .as_ref()
            .ok_or(TranslationError::BridgeNotStarted)?;

        let mut bridge = process_mutex.lock().map_err(|e| {
            TranslationError::RequestFailed(format!("Lock poisoned: {}", e))
        })?;

        let request = TranslateRequest {
            text: segment.text.clone(),
            source_lang: lang_code(&self.direction.source),
            target_lang: lang_code(&self.direction.target),
        };

        let request_json =
            serde_json::to_string(&request).map_err(|e| TranslationError::RequestFailed(e.to_string()))?;

        writeln!(bridge.stdin, "{}", request_json)
            .map_err(|e| TranslationError::RequestFailed(e.to_string()))?;
        bridge
            .stdin
            .flush()
            .map_err(|e| TranslationError::RequestFailed(e.to_string()))?;

        let mut response_line = String::new();
        bridge
            .stdout
            .read_line(&mut response_line)
            .map_err(|e| TranslationError::RequestFailed(e.to_string()))?;

        let response: TranslateResponse = serde_json::from_str(&response_line)
            .map_err(|e| TranslationError::InvalidResponse(format!("{}: {}", e, response_line)))?;

        tracing::debug!(
            "Translation: \"{}\" -> \"{}\"",
            segment.text,
            response.translated
        );

        Ok(TextSegment::new(response.translated, self.direction.target))
    }
}

fn lang_code(language: &Language) -> String {
    match language {
        Language::English => "en".to_string(),
        Language::Portuguese => "pt".to_string(),
    }
}

#[async_trait]
impl PipelineStage for OpusMtTranslator {
    fn name(&self) -> &str {
        "opus-mt-translation"
    }

    async fn initialize(&mut self) -> Result<(), StageError> {
        let script_path = self.bridge_script_path.to_string_lossy().to_string();
        tracing::info!("Starting translation bridge: {}", script_path);

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

        // Wait for the bridge to signal readiness
        let mut ready_line = String::new();
        reader
            .read_line(&mut ready_line)
            .map_err(|e| StageError::NotInitialized(format!("Bridge startup failed: {}", e)))?;

        if !ready_line.trim().contains("ready") {
            return Err(StageError::NotInitialized(format!(
                "Bridge did not signal ready: {}",
                ready_line
            )));
        }

        self.process = Some(Mutex::new(BridgeProcess {
            child,
            stdin,
            stdout: reader,
        }));

        tracing::info!("Translation bridge ready");
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<(), StageError> {
        if let Some(process_mutex) = self.process.take() {
            if let Ok(mut bridge) = process_mutex.into_inner() {
                let _ = bridge.child.kill();
                let _ = bridge.child.wait();
            }
        }
        tracing::info!("Translation bridge stopped");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lang_code_returns_correct_codes() {
        assert_eq!(lang_code(&Language::English), "en");
        assert_eq!(lang_code(&Language::Portuguese), "pt");
    }

    #[test]
    fn translate_request_serializes_correctly() {
        let request = TranslateRequest {
            text: "hello world".to_string(),
            source_lang: "en".to_string(),
            target_lang: "pt".to_string(),
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("hello world"));
        assert!(json.contains("en"));
        assert!(json.contains("pt"));
    }

    #[test]
    fn translate_response_deserializes_correctly() {
        let json = r#"{"translated": "olá mundo"}"#;
        let response: TranslateResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.translated, "olá mundo");
    }

    #[test]
    fn translator_without_bridge_returns_error() {
        let translator = OpusMtTranslator::new(
            PathBuf::from("nonexistent.py"),
            TranslationDirection::new(Language::English, Language::Portuguese),
        );
        let segment = TextSegment::new("hello".to_string(), Language::English);
        let result = translator.translate(&segment);
        assert!(result.is_err());
    }
}
