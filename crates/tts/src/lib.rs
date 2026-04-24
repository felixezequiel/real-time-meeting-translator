use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use shared::{AudioChunk, Language, PipelineStage, StageError, TextSegment};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use thiserror::Error;
use tracing;

const MONO_CHANNELS: u16 = 1;
const BYTES_PER_INT16_SAMPLE: usize = 2;

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

/// Header sent by the Python bridge before the raw int16 PCM bytes.
/// See scripts/tts_bridge.py — the binary-framed protocol avoids a
/// filesystem round-trip (write WAV → read WAV → delete) per synthesis.
#[derive(Deserialize)]
struct TtsResponseHeader {
    sample_rate: u32,
    num_samples: u32,
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

        let mut header_line = String::new();
        bridge
            .stdout
            .read_line(&mut header_line)
            .map_err(|e| TtsError::SynthesisFailed(e.to_string()))?;

        let header: TtsResponseHeader = serde_json::from_str(header_line.trim())
            .map_err(|e| TtsError::SynthesisFailed(format!("Invalid TTS header: {}", e)))?;

        let samples = read_pcm_samples(&mut bridge.stdout, header.num_samples as usize)
            .map_err(|e| TtsError::SynthesisFailed(format!("Failed to read PCM: {}", e)))?;

        tracing::debug!(
            "TTS synthesized {} samples at {}Hz",
            samples.len(),
            header.sample_rate
        );

        Ok(AudioChunk::new(samples, header.sample_rate, MONO_CHANNELS))
    }
}

/// Read exactly `num_samples` int16 samples from the bridge stdout and
/// convert to normalized f32. Uses `read_exact` on the BufReader so any
/// bytes already buffered after the header line are consumed first.
fn read_pcm_samples<R: Read>(
    reader: &mut BufReader<R>,
    num_samples: usize,
) -> std::io::Result<Vec<f32>> {
    let mut bytes = vec![0u8; num_samples * BYTES_PER_INT16_SAMPLE];
    reader.read_exact(&mut bytes)?;

    let mut samples = Vec::with_capacity(num_samples);
    for chunk in bytes.chunks_exact(BYTES_PER_INT16_SAMPLE) {
        let int16 = i16::from_le_bytes([chunk[0], chunk[1]]);
        samples.push(int16 as f32 / 32768.0);
    }
    Ok(samples)
}

#[async_trait]
impl PipelineStage for PiperTts {
    fn name(&self) -> &str {
        "piper-tts"
    }

    async fn initialize(&mut self) -> Result<(), StageError> {
        let script_path = self.bridge_script_path.to_string_lossy().to_string();
        tracing::info!("Starting TTS bridge: {}", script_path);

        let python = shared::find_python();
        let mut child = Command::new(&python)
            .arg(&script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| StageError::NotInitialized(
                format!("Failed to start Python (tried '{}'). Is Python 3.10+ installed and in PATH? Error: {}", python, e)
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
    use std::io::Cursor;

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
    fn tts_header_deserializes_correctly() {
        let json = r#"{"sample_rate": 22050, "num_samples": 3}"#;
        let header: TtsResponseHeader = serde_json::from_str(json).unwrap();
        assert_eq!(header.sample_rate, 22050);
        assert_eq!(header.num_samples, 3);
    }

    #[test]
    fn read_pcm_samples_converts_int16_to_normalized_f32() {
        // 3 samples: 0, i16::MAX, i16::MIN — little-endian.
        let bytes: Vec<u8> = vec![0x00, 0x00, 0xFF, 0x7F, 0x00, 0x80];
        let mut reader = BufReader::new(Cursor::new(bytes));
        let samples = read_pcm_samples(&mut reader, 3).unwrap();

        assert_eq!(samples.len(), 3);
        assert!((samples[0] - 0.0).abs() < 1e-6);
        assert!((samples[1] - (i16::MAX as f32 / 32768.0)).abs() < 1e-6);
        assert!((samples[2] - (i16::MIN as f32 / 32768.0)).abs() < 1e-6);
    }

    #[test]
    fn read_pcm_samples_errors_on_truncated_stream() {
        // Promise 2 samples (4 bytes) but provide only 2 bytes.
        let bytes: Vec<u8> = vec![0x00, 0x00];
        let mut reader = BufReader::new(Cursor::new(bytes));
        assert!(read_pcm_samples(&mut reader, 2).is_err());
    }

    #[test]
    fn tts_without_bridge_returns_error() {
        let tts = PiperTts::new(PathBuf::from("nonexistent.py"), Language::Portuguese);
        let segment = TextSegment::new("olá".to_string(), Language::Portuguese);
        let result = tts.synthesize(&segment);
        assert!(result.is_err());
    }
}
