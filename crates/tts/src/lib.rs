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

/// Target voice characteristics for the next synthesis. Resolved per-call
/// from the speaker's running F0 + formant profile maintained by the
/// pipeline; default values mean "use the bridge's default voice".
#[derive(Debug, Clone, Default)]
pub struct VoiceProfile {
    /// Target F0 in Hz (mean of voiced frames in the speaker's recent
    /// audio). Anything <= 0 disables pitch shifting. Used by the
    /// Kokoro engine; XTTS-v2 ignores this and infers prosody from
    /// the reference WAV instead.
    pub target_f0_hz: f32,
    /// Spectral-envelope warp ratio. 1.0 = no formant shift; >1 enlarges
    /// the vocal tract → deeper voice; <1 narrows → thinner.
    /// Kokoro-only, like target_f0_hz.
    pub formant_shift: f32,
    /// Diarised speaker id this synthesis is attributed to. The bridge
    /// uses it to keep voice routing stable across utterances of the
    /// same speaker (sticky voice + F0 hysteresis) — without it, F0
    /// jitter chunk-to-chunk would alternate the same person's
    /// translations between male and female voices.
    pub speaker_id: Option<u32>,
    /// Path to a reference WAV (5-10 s of the target speaker's voice)
    /// used by zero-shot voice cloning engines (XTTS-v2 — ADR 0014).
    /// Kokoro ignores this. Resolved by the pipeline from
    /// `VoiceProfileRegistry::reference_for(speaker_id)` or the mic-
    /// side `mic_voice_profile_path`.
    pub reference_wav_path: Option<String>,
}

impl VoiceProfile {
    pub fn is_active(&self) -> bool {
        self.target_f0_hz > 0.0
    }
}

#[derive(Serialize)]
struct TtsRequest {
    text: String,
    language: String,
    /// Speaker's running mean F0. Omitted from the JSON when zero so the
    /// bridge skips analysis-synthesis entirely (Piper output passes
    /// through, ~150 ms per call instead of ~250-350 ms).
    #[serde(skip_serializing_if = "f32_is_zero_or_negative")]
    target_f0: f32,
    /// Spectral-envelope warp. Always sent — defaults to 1.0 (no warp).
    formant_shift: f32,
    /// Diarised speaker id, when known. The bridge keeps a sticky voice
    /// per id so the same speaker keeps the same voice across utterances.
    #[serde(skip_serializing_if = "Option::is_none")]
    speaker_id: Option<u32>,
    /// Path to a reference WAV for zero-shot voice cloning engines
    /// (XTTS-v2 — ADR 0014). Kokoro ignores this. Both bridges accept
    /// the same JSON shape; engine-specific fields are ignored on the
    /// other side.
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_wav_path: Option<String>,
}

fn f32_is_zero_or_negative(value: &f32) -> bool {
    *value <= 0.0
}

/// Header sent by the Python bridge before the raw int16 PCM bytes
/// for ONE frame. The bridge emits frames in a loop, terminated by a
/// frame with `is_final=true`. Atomic bridges (Kokoro) emit exactly
/// one frame whose body carries the full audio AND `is_final=true`.
/// Streaming bridges (XTTS-v2) emit N body frames with `is_final=false`
/// followed by a single empty terminator frame with `is_final=true`.
#[derive(Deserialize)]
struct TtsResponseHeader {
    sample_rate: u32,
    num_samples: u32,
    /// `final` on the wire (Python keyword-safe, matches both bridges).
    /// Defaults to true so a bridge that doesn't yet emit the field is
    /// treated as legacy single-frame — keeps a half-upgraded checkout
    /// working without a coordinated bump.
    #[serde(rename = "final", default = "default_final")]
    is_final: bool,
}

fn default_final() -> bool {
    true
}

/// Local TTS client with optional pyworld pitch/formant shifting on top.
///
/// The current backend is **Kokoro v1.0** (~82M params, ONNX, 24 kHz —
/// see ADR 0010). The struct kept its `PiperTts` name from the previous
/// backend because the Rust-side protocol is identical (only the
/// Python bridge changed); renaming would churn every downstream import
/// for no functional gain.
///
/// One instance owns a single Python subprocess; the `Mutex` serialises
/// concurrent callers so stdout frames don't interleave. Clone the
/// `Arc<PiperTts>` to share between pipelines.
pub struct PiperTts {
    bridge_script_path: PathBuf,
    process: Option<Mutex<TtsBridgeProcess>>,
}

struct TtsBridgeProcess {
    child: Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl PiperTts {
    /// Construct a TTS bridge client. The `_language` parameter is
    /// kept on the constructor for API stability (callers passed it
    /// for years), but is no longer stored — the synthesise call now
    /// derives language from `TextSegment.language` per request, so
    /// one bridge instance handles both directions.
    pub fn new(bridge_script_path: PathBuf, _language: Language) -> Self {
        Self {
            bridge_script_path,
            process: None,
        }
    }

    /// Streaming synthesis: invokes `on_chunk` for every audio fragment
    /// the bridge emits, in order, until the bridge signals the
    /// terminator frame (`is_final=true`).
    ///
    /// Each emitted `AudioChunk` is constructed with
    /// `AudioChunk::streaming(...)` when the bridge says there are more
    /// frames coming AND with `AudioChunk::new(...)` for the trailing
    /// content frame (if any) — only the LAST content chunk gets the
    /// non-streaming flag so the mixer can apply its phrase-boundary
    /// fade-out there. Mid-phrase chunks skip the envelope entirely
    /// (see `AudioChunk::is_streaming_chunk`).
    ///
    /// Atomic bridges (Kokoro) emit exactly one frame with
    /// `is_final=true` carrying the full audio — `on_chunk` fires once
    /// and the chunk is non-streaming (full envelope applies).
    pub fn synthesize_stream<F>(
        &self,
        segment: &TextSegment,
        voice_profile: VoiceProfile,
        mut on_chunk: F,
    ) -> Result<(), TtsError>
    where
        F: FnMut(AudioChunk),
    {
        let process_mutex = self
            .process
            .as_ref()
            .ok_or(TtsError::BridgeNotStarted)?;

        let mut bridge = process_mutex.lock().map_err(|e| {
            TtsError::SynthesisFailed(format!("Lock poisoned: {}", e))
        })?;

        // Language comes from the segment, not from the bridge instance.
        // Both the Kokoro and XTTS bridges accept any language per call,
        // so a single `PiperTts` can serve both translation directions —
        // critical when XTTS is the engine because each instance burns
        // ~1.8 GB of VRAM (ADR 0014 amendment 2026-05-08: two instances
        // were saturating 6 GB GPUs and producing silent output).
        let request = TtsRequest {
            text: segment.text.clone(),
            language: match segment.language {
                Language::Portuguese => "pt".to_string(),
                Language::English => "en".to_string(),
            },
            target_f0: voice_profile.target_f0_hz,
            formant_shift: if voice_profile.formant_shift > 0.0 {
                voice_profile.formant_shift
            } else {
                1.0
            },
            speaker_id: voice_profile.speaker_id,
            reference_wav_path: voice_profile.reference_wav_path.clone(),
        };

        let request_json =
            serde_json::to_string(&request).map_err(|e| TtsError::SynthesisFailed(e.to_string()))?;

        writeln!(bridge.stdin, "{}", request_json)
            .map_err(|e| TtsError::SynthesisFailed(e.to_string()))?;
        bridge
            .stdin
            .flush()
            .map_err(|e| TtsError::SynthesisFailed(e.to_string()))?;

        // Buffer the previous content frame so we can mark it as the
        // final (atomic-style) one when the terminator arrives. This
        // way the LAST chunk of a streamed phrase carries the
        // boundary fade-out the mixer applies to atomic chunks, while
        // every chunk before it stays mid-phrase (no envelope).
        let mut pending: Option<AudioChunk> = None;
        let mut total_samples: usize = 0;
        let mut frames_seen: usize = 0;
        // Set on the first frame and overwritten unconditionally on
        // every subsequent frame; the initial 0 is never observed by
        // the trace below because the loop runs at least once.
        #[allow(unused_assignments)]
        let mut last_sample_rate: u32 = 0;

        loop {
            let mut header_line = String::new();
            bridge
                .stdout
                .read_line(&mut header_line)
                .map_err(|e| TtsError::SynthesisFailed(e.to_string()))?;

            let header: TtsResponseHeader = serde_json::from_str(header_line.trim())
                .map_err(|e| TtsError::SynthesisFailed(format!("Invalid TTS header: {}", e)))?;

            let samples = read_pcm_samples(&mut bridge.stdout, header.num_samples as usize)
                .map_err(|e| TtsError::SynthesisFailed(format!("Failed to read PCM: {}", e)))?;

            last_sample_rate = header.sample_rate;
            total_samples += samples.len();
            frames_seen += 1;

            if header.is_final {
                // Terminator frame. If the terminator itself carries
                // audio (Kokoro: full payload + final=true), treat it
                // as both the last content frame AND the close —
                // flush the pending mid-phrase chunk first (if any),
                // then emit this one as the boundary-eligible last.
                if let Some(prev) = pending.take() {
                    on_chunk(prev);
                }
                if !samples.is_empty() {
                    on_chunk(AudioChunk::new(samples, header.sample_rate, MONO_CHANNELS));
                }
                break;
            }

            // Non-final frame. The PREVIOUSLY-buffered chunk (if any)
            // is now confirmed to be mid-phrase: emit it tagged as a
            // streaming chunk so the mixer skips its per-chunk fade.
            if let Some(prev) = pending.take() {
                on_chunk(AudioChunk::streaming(
                    prev.samples,
                    prev.sample_rate,
                    prev.channels,
                ));
            }
            if !samples.is_empty() {
                pending = Some(AudioChunk::new(samples, header.sample_rate, MONO_CHANNELS));
            }
        }

        tracing::debug!(
            "TTS streamed {} samples in {} frames at {}Hz (target_f0={:.0}, formant={:.2})",
            total_samples,
            frames_seen,
            last_sample_rate,
            voice_profile.target_f0_hz,
            voice_profile.formant_shift,
        );

        Ok(())
    }

    /// Atomic synthesis. Wrapper around `synthesize_stream` that
    /// concatenates every fragment into one `AudioChunk`. Kept for
    /// callers (and tests) that don't care about per-chunk dispatch
    /// — the streaming pipeline in `crates/pipeline/src/v2.rs` calls
    /// `synthesize_stream` directly to dispatch each chunk to the
    /// playback mixer as it lands.
    pub fn synthesize(
        &self,
        segment: &TextSegment,
        voice_profile: VoiceProfile,
    ) -> Result<AudioChunk, TtsError> {
        let mut combined: Vec<f32> = Vec::new();
        let mut sample_rate: u32 = 0;
        self.synthesize_stream(segment, voice_profile, |chunk| {
            sample_rate = chunk.sample_rate;
            combined.extend(chunk.samples);
        })?;
        Ok(AudioChunk::new(combined, sample_rate, MONO_CHANNELS))
    }
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

#[async_trait]
impl PipelineStage for PiperTts {
    fn name(&self) -> &str {
        "piper-tts"
    }

    async fn initialize(&mut self) -> Result<(), StageError> {
        let script_path = self.bridge_script_path.to_string_lossy().to_string();
        tracing::info!("Starting Piper TTS bridge: {}", script_path);

        let python = shared::find_python();
        let mut child = Command::new(&python)
            .arg(&script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
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

        tracing::info!("Piper TTS bridge ready");
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<(), StageError> {
        if let Some(process_mutex) = self.process.take() {
            if let Ok(mut bridge) = process_mutex.into_inner() {
                let _ = bridge.child.kill();
                let _ = bridge.child.wait();
            }
        }
        tracing::info!("Piper TTS bridge stopped");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn voice_profile_is_active_when_f0_positive() {
        let profile = VoiceProfile {
            target_f0_hz: 180.0,
            formant_shift: 1.0,
            speaker_id: None,
            reference_wav_path: None,
        };
        assert!(profile.is_active());
        let default = VoiceProfile::default();
        assert!(!default.is_active());
    }

    #[test]
    fn tts_request_omits_target_f0_when_zero() {
        let request = TtsRequest {
            text: "olá".to_string(),
            language: "pt".to_string(),
            target_f0: 0.0,
            formant_shift: 1.0,
            speaker_id: None,
            reference_wav_path: None,
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(!json.contains("target_f0"));
        assert!(json.contains("formant_shift"));
    }

    #[test]
    fn tts_request_includes_target_f0_when_set() {
        let request = TtsRequest {
            text: "hi".to_string(),
            language: "en".to_string(),
            target_f0: 220.0,
            formant_shift: 0.95,
            speaker_id: Some(0),
            reference_wav_path: None,
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("target_f0"));
        assert!(json.contains("220"));
    }

    #[test]
    fn tts_request_omits_reference_wav_when_absent() {
        let request = TtsRequest {
            text: "hi".to_string(),
            language: "en".to_string(),
            target_f0: 0.0,
            formant_shift: 1.0,
            speaker_id: None,
            reference_wav_path: None,
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(!json.contains("reference_wav_path"));
    }

    #[test]
    fn tts_request_emits_reference_wav_when_set() {
        let request = TtsRequest {
            text: "hi".to_string(),
            language: "en".to_string(),
            target_f0: 0.0,
            formant_shift: 1.0,
            speaker_id: None,
            reference_wav_path: Some("/tmp/voice/ref.wav".to_string()),
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("reference_wav_path"));
        assert!(json.contains("ref.wav"));
    }

    #[test]
    fn tts_header_deserialises_correctly() {
        let json = r#"{"sample_rate": 22050, "num_samples": 3, "final": true}"#;
        let header: TtsResponseHeader = serde_json::from_str(json).unwrap();
        assert_eq!(header.sample_rate, 22050);
        assert_eq!(header.num_samples, 3);
        assert!(header.is_final);
    }

    #[test]
    fn tts_header_defaults_final_when_absent() {
        // Backward compat with bridges that haven't been upgraded yet:
        // missing `final` is treated as a single-frame atomic response.
        let json = r#"{"sample_rate": 22050, "num_samples": 3}"#;
        let header: TtsResponseHeader = serde_json::from_str(json).unwrap();
        assert!(header.is_final);
    }

    #[test]
    fn tts_header_parses_non_final_frame() {
        let json = r#"{"sample_rate": 24000, "num_samples": 6000, "final": false}"#;
        let header: TtsResponseHeader = serde_json::from_str(json).unwrap();
        assert!(!header.is_final);
    }

    #[test]
    fn read_pcm_samples_converts_int16_to_normalised_f32() {
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
        let bytes: Vec<u8> = vec![0x00, 0x00];
        let mut reader = BufReader::new(Cursor::new(bytes));
        assert!(read_pcm_samples(&mut reader, 2).is_err());
    }

    #[test]
    fn tts_without_bridge_returns_error() {
        let tts = PiperTts::new(PathBuf::from("nonexistent.py"), Language::Portuguese);
        let segment = TextSegment::new("olá".to_string(), Language::Portuguese);
        let result = tts.synthesize(&segment, VoiceProfile::default());
        assert!(result.is_err());
    }
}
