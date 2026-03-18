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
    audio_file: String,
    language: String,
}

#[derive(Deserialize)]
struct SttResponse {
    text: String,
    /// ISO-639-1 code detected by Whisper (e.g. "en", "pt").
    /// Present in all responses from the updated bridge; may be absent in legacy responses.
    #[serde(default)]
    language: Option<String>,
}

pub struct WhisperStt {
    bridge_script_path: PathBuf,
    model_path: PathBuf,
    process: Option<Mutex<SttBridgeProcess>>,
    language: Language,
    request_counter: Mutex<u64>,
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
            request_counter: Mutex::new(0),
        }
    }

    pub fn transcribe(&self, chunk: &AudioChunk) -> Result<TextSegment, SttError> {
        let process_mutex = self
            .process
            .as_ref()
            .ok_or(SttError::BridgeNotStarted)?;

        // Write audio to temp WAV file
        let mut counter = self.request_counter.lock().unwrap();
        *counter += 1;
        let temp_path = std::env::temp_dir().join(format!("stt_chunk_{}.wav", *counter));
        drop(counter);

        write_wav(&temp_path, &chunk.samples, chunk.sample_rate)
            .map_err(|e| SttError::TranscriptionFailed(format!("Failed to write WAV: {}", e)))?;

        let mut bridge = process_mutex
            .lock()
            .map_err(|e| SttError::TranscriptionFailed(format!("Lock poisoned: {}", e)))?;

        let request = SttRequest {
            audio_file: temp_path.to_string_lossy().to_string(),
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
            .map_err(|e| SttError::TranscriptionFailed(format!("{}: {}", e, response_line.trim())))?;

        let trimmed = response.text.trim().to_string();

        // Use the language Whisper actually detected, not just what we expected.
        // This is critical for the feedback-loop guard: if TTS output (target language)
        // leaks back into the loopback capture, the pipeline can reject it because the
        // detected language won't match the expected source language.
        let detected_language = parse_language_code(response.language.as_deref(), self.language);

        Ok(TextSegment::new(trimmed, detected_language))
    }
}

/// Map a Whisper ISO-639-1 language code to our `Language` enum.
/// Falls back to `expected` when the code is absent or unrecognised so that
/// audio in languages we do not model (Spanish, French, …) is treated as if it
/// were in the expected source language and will still be picked up downstream.
/// The pipeline's language-guard then decides whether to process or drop it.
fn parse_language_code(code: Option<&str>, expected: Language) -> Language {
    match code {
        Some("en") | Some("english") => Language::English,
        Some("pt") | Some("portuguese") => Language::Portuguese,
        _ => expected,
    }
}

fn write_wav(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    let num_samples = samples.len() as u32;
    let byte_rate = sample_rate * 2; // 16-bit mono
    let data_size = num_samples * 2;

    // WAV header
    file.write_all(b"RIFF")?;
    file.write_all(&(36 + data_size).to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // chunk size
    file.write_all(&1u16.to_le_bytes())?; // PCM
    file.write_all(&1u16.to_le_bytes())?; // mono
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?; // block align
    file.write_all(&16u16.to_le_bytes())?; // bits per sample
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;

    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let int16 = (clamped * 32767.0) as i16;
        file.write_all(&int16.to_le_bytes())?;
    }

    Ok(())
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
    use std::path::PathBuf;

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

    #[test]
    fn write_wav_creates_valid_file() {
        let temp = std::env::temp_dir().join("test_write_wav.wav");
        let samples = vec![0.0f32; 16000];
        write_wav(&temp, &samples, 16000).unwrap();
        assert!(temp.exists());
        let metadata = std::fs::metadata(&temp).unwrap();
        assert!(metadata.len() > 44); // WAV header + data
        std::fs::remove_file(&temp).unwrap();
    }
}
