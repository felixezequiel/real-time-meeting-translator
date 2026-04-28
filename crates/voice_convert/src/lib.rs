use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use shared::{AudioChunk, PipelineStage, StageError};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use thiserror::Error;
use tracing;

const BYTES_PER_INT16_SAMPLE: usize = 2;
const MONO_CHANNELS: u16 = 1;
const I16_MAX_AS_F32: f32 = 32767.0;

#[derive(Debug, Error)]
pub enum VcError {
    #[error("Voice-convert bridge not started")]
    BridgeNotStarted,

    #[error("Conversion failed: {0}")]
    ConversionFailed(String),
}

/// Frame the bridge expects on stdin: JSON header line, then the binary
/// PCM payload (`source_num_samples` int16 little-endian samples).
#[derive(Serialize)]
struct VcRequestHeader {
    source_sr: u32,
    source_num_samples: u32,
    reference_wav_path: String,
    speaker_id: u32,
}

/// Preload-only request: the bridge extracts and caches the speaker
/// embedding for `reference_wav_path` and replies with a single JSON
/// line, no audio. Used at "Start" time so the first real conversion
/// fires without paying the SE-extraction cost.
#[derive(Serialize)]
struct VcPreloadRequest {
    action: &'static str,
    reference_wav_path: String,
}

#[derive(Deserialize)]
struct VcPreloadResponse {
    status: String,
    #[serde(default)]
    elapsed_ms: u64,
    #[serde(default)]
    reference_wav_path: String,
}

/// Header the bridge emits before the converted PCM. The output sample
/// rate is fixed at the TCC's native rate (22050 Hz for OpenVoice v2);
/// the playback resampler in `crates/audio` lifts it to the device rate.
#[derive(Deserialize)]
struct VcResponseHeader {
    sample_rate: u32,
    num_samples: u32,
}

/// Client for `scripts/voice_convert_bridge.py`. Same architectural
/// pattern as the diarizer / TTS bridges: one persistent Python
/// subprocess, requests serialised through a `Mutex` so the binary
/// payloads on stdout don't interleave between concurrent callers.
///
/// The bridge can fail to start (OpenVoice not vendored, checkpoint
/// missing) without taking the rest of the pipeline down. When that
/// happens, `dead` flips to true and every subsequent `convert()`
/// returns the source audio unchanged. The Rust pipeline then plays
/// raw Kokoro output — no fancy timbre conversion, but no crash.
pub struct ToneColorConverter {
    bridge_script_path: PathBuf,
    process: Option<Mutex<VcBridgeProcess>>,
    dead: Arc<AtomicBool>,
}

struct VcBridgeProcess {
    child: Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

/// Result of a single conversion call. Always at the bridge's native
/// 22050 Hz rate (OpenVoice TCC's training rate); the caller is
/// responsible for resampling to the playback rate.
#[derive(Debug, Clone)]
pub struct ConvertedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl ToneColorConverter {
    pub fn new(bridge_script_path: PathBuf) -> Self {
        Self {
            bridge_script_path,
            process: None,
            dead: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Ask the bridge to extract and cache the target speaker
    /// embedding for `reference_wav_path` *now*, so the first real
    /// `convert()` call doesn't pay the ~150 ms SE-extraction cost.
    /// Safe to call before any conversion request.
    ///
    /// Returns `Ok(false)` if the bridge is dead or the preload
    /// failed — both are non-fatal: the next `convert()` will simply
    /// extract the SE on demand or return the source audio unchanged.
    pub fn preload_reference(
        &self,
        reference_wav_path: &str,
    ) -> Result<bool, VcError> {
        if self.dead.load(Ordering::Acquire) {
            return Ok(false);
        }
        let process_mutex = self
            .process
            .as_ref()
            .ok_or(VcError::BridgeNotStarted)?;
        let mut bridge = process_mutex.lock().map_err(|e| {
            VcError::ConversionFailed(format!("Lock poisoned: {}", e))
        })?;

        let request = VcPreloadRequest {
            action: "preload",
            reference_wav_path: reference_wav_path.to_string(),
        };
        let header_json = serde_json::to_string(&request)
            .map_err(|e| VcError::ConversionFailed(e.to_string()))?;

        let result: Result<VcPreloadResponse, VcError> = (|| {
            writeln!(bridge.stdin, "{}", header_json)
                .map_err(|e| VcError::ConversionFailed(e.to_string()))?;
            bridge
                .stdin
                .flush()
                .map_err(|e| VcError::ConversionFailed(e.to_string()))?;
            let mut header_line = String::new();
            bridge
                .stdout
                .read_line(&mut header_line)
                .map_err(|e| VcError::ConversionFailed(e.to_string()))?;
            let response: VcPreloadResponse = serde_json::from_str(header_line.trim())
                .map_err(|e| VcError::ConversionFailed(format!("Invalid header: {}", e)))?;
            Ok(response)
        })();

        match result {
            Ok(response) => {
                let ok = response.status == "preloaded";
                if ok {
                    tracing::info!(
                        "Voice-convert SE preloaded for {} ({} ms)",
                        response.reference_wav_path,
                        response.elapsed_ms,
                    );
                } else {
                    tracing::warn!(
                        "Voice-convert preload failed for {} (status={})",
                        response.reference_wav_path,
                        response.status,
                    );
                }
                Ok(ok)
            }
            Err(e) => {
                self.dead.store(true, Ordering::Release);
                tracing::error!(
                    "Voice-convert bridge marked dead during preload: {}",
                    e,
                );
                Ok(false)
            }
        }
    }

    /// Convert `source` to the timbre captured by the WAV at
    /// `reference_wav_path`. Returns `Ok(None)` when the bridge has
    /// been marked dead — the caller should fall back to using the
    /// source audio unchanged.
    pub fn convert(
        &self,
        source: &AudioChunk,
        reference_wav_path: &str,
        speaker_id: u32,
    ) -> Result<Option<ConvertedAudio>, VcError> {
        if self.dead.load(Ordering::Acquire) {
            return Ok(None);
        }

        let process_mutex = self
            .process
            .as_ref()
            .ok_or(VcError::BridgeNotStarted)?;

        let mut bridge = process_mutex.lock().map_err(|e| {
            VcError::ConversionFailed(format!("Lock poisoned: {}", e))
        })?;

        let header = VcRequestHeader {
            source_sr: source.sample_rate,
            source_num_samples: source.samples.len() as u32,
            reference_wav_path: reference_wav_path.to_string(),
            speaker_id,
        };

        let header_json = serde_json::to_string(&header)
            .map_err(|e| VcError::ConversionFailed(e.to_string()))?;

        let result: Result<ConvertedAudio, VcError> = (|| {
            writeln!(bridge.stdin, "{}", header_json)
                .map_err(|e| VcError::ConversionFailed(e.to_string()))?;

            let pcm_bytes = encode_f32_as_int16_le(&source.samples);
            bridge
                .stdin
                .write_all(&pcm_bytes)
                .map_err(|e| VcError::ConversionFailed(e.to_string()))?;
            bridge
                .stdin
                .flush()
                .map_err(|e| VcError::ConversionFailed(e.to_string()))?;

            let mut header_line = String::new();
            bridge
                .stdout
                .read_line(&mut header_line)
                .map_err(|e| VcError::ConversionFailed(e.to_string()))?;

            let response: VcResponseHeader = serde_json::from_str(header_line.trim())
                .map_err(|e| VcError::ConversionFailed(format!("Invalid header: {}", e)))?;

            let samples = read_pcm_samples(&mut bridge.stdout, response.num_samples as usize)
                .map_err(|e| VcError::ConversionFailed(format!("PCM read: {}", e)))?;

            tracing::debug!(
                "VC: speaker={} {} samples @ {}Hz → {} samples @ {}Hz",
                speaker_id,
                source.samples.len(),
                source.sample_rate,
                samples.len(),
                response.sample_rate,
            );

            Ok(ConvertedAudio {
                samples,
                sample_rate: response.sample_rate,
            })
        })();

        match result {
            Ok(audio) => Ok(Some(audio)),
            Err(e) => {
                // One bad request kills the bridge for the rest of the
                // session — pyworld errors and CUDA OOMs leave the
                // Python subprocess in a corrupted state, and trying to
                // recover would block downstream forever. Marking dead
                // makes subsequent calls return None instantly so the
                // pipeline keeps moving with raw TTS audio.
                self.dead.store(true, Ordering::Release);
                tracing::error!(
                    "Voice-convert bridge marked dead after failure: {}. \
                     Pipeline continues with unconverted TTS output.",
                    e
                );
                Ok(None)
            }
        }
    }
}

fn encode_f32_as_int16_le(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * BYTES_PER_INT16_SAMPLE);
    for &sample in samples {
        let clipped = sample.clamp(-1.0, 1.0);
        let int16 = (clipped * I16_MAX_AS_F32) as i16;
        bytes.extend_from_slice(&int16.to_le_bytes());
    }
    bytes
}

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

/// Build an `AudioChunk` from a `ConvertedAudio` value. Convenience for
/// callers that want to drop the converted audio straight back into the
/// pipeline's mixer channel.
impl From<ConvertedAudio> for AudioChunk {
    fn from(value: ConvertedAudio) -> Self {
        AudioChunk::new(value.samples, value.sample_rate, MONO_CHANNELS)
    }
}

#[async_trait]
impl PipelineStage for ToneColorConverter {
    fn name(&self) -> &str {
        "openvoice-tcc"
    }

    async fn initialize(&mut self) -> Result<(), StageError> {
        let script_path = self.bridge_script_path.to_string_lossy().to_string();
        tracing::info!("Starting voice-convert bridge: {}", script_path);

        let python = shared::find_python();
        let mut child = Command::new(&python)
            .arg(&script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| StageError::NotInitialized(format!(
                "Failed to start Python (tried '{}'). Error: {}", python, e,
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
            .map_err(|e| StageError::NotInitialized(format!("Bridge startup failed: {}", e)))?;

        if !ready_line.trim().contains("ready") {
            return Err(StageError::NotInitialized(format!(
                "Bridge did not signal ready: {}",
                ready_line
            )));
        }

        self.process = Some(Mutex::new(VcBridgeProcess {
            child,
            stdin,
            stdout: reader,
        }));

        tracing::info!("Voice-convert bridge ready (OpenVoice v2 TCC)");
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<(), StageError> {
        if let Some(process_mutex) = self.process.take() {
            if let Ok(mut bridge) = process_mutex.into_inner() {
                let _ = bridge.child.kill();
                let _ = bridge.child.wait();
            }
        }
        tracing::info!("Voice-convert bridge stopped");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn request_header_serialises_with_all_fields() {
        let header = VcRequestHeader {
            source_sr: 24000,
            source_num_samples: 12000,
            reference_wav_path: "C:/refs/spk0.wav".to_string(),
            speaker_id: 0,
        };
        let json = serde_json::to_string(&header).unwrap();
        assert!(json.contains("24000"));
        assert!(json.contains("spk0.wav"));
        assert!(json.contains("\"speaker_id\":0"));
    }

    #[test]
    fn response_header_deserialises() {
        let json = r#"{"sample_rate": 22050, "num_samples": 11000}"#;
        let header: VcResponseHeader = serde_json::from_str(json).unwrap();
        assert_eq!(header.sample_rate, 22050);
        assert_eq!(header.num_samples, 11000);
    }

    #[test]
    fn encode_then_decode_is_lossless_within_int16() {
        let samples = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let bytes = encode_f32_as_int16_le(&samples);
        let mut reader = BufReader::new(Cursor::new(bytes));
        let decoded = read_pcm_samples(&mut reader, samples.len()).unwrap();
        for (orig, got) in samples.iter().zip(decoded.iter()) {
            assert!((orig - got).abs() < 1e-3, "{} vs {}", orig, got);
        }
    }

    #[test]
    fn convert_errors_when_bridge_not_started() {
        let vc = ToneColorConverter::new(PathBuf::from("nonexistent.py"));
        let source = AudioChunk::new(vec![0.0; 100], 24000, 1);
        assert!(vc.convert(&source, "C:/fake.wav", 0).is_err());
    }
}
