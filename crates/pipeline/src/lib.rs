use tracing;

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub mod v2;
pub use v2::{SpeakerPipelineV2, SubtitleEvent, V2Config};

/// How long to keep recent translations for echo detection. STT feedback
/// typically appears within 2-4 seconds of TTS playback.
const ECHO_WINDOW_SECONDS: f32 = 8.0;

/// Word overlap threshold above which STT text is considered an echo of a
/// recent translation and dropped.
const ECHO_SIMILARITY_THRESHOLD: f32 = 0.4;

/// Smoothing factor for the per-speaker running F0. Each new sample is mixed
/// in with this weight; a higher value reacts faster, a lower value is
/// steadier. 0.2 trades roughly 5 chunks of half-life for stability — a
/// loud cough or breath produces a brief F0 spike that doesn't survive the
/// average, while a sustained pitch change (different speaker, or the same
/// speaker getting excited) updates the profile within ~2 s.
const F0_RUNNING_MEAN_ALPHA: f32 = 0.2;

/// F0 ceilings used to clamp running-mean updates. pyworld occasionally
/// returns absurd values when fed near-silence; clamping prevents one bad
/// chunk from poisoning a speaker's entire profile.
const F0_MIN_HZ: f32 = 70.0;
const F0_MAX_HZ: f32 = 400.0;

/// Seconds of clean speech to collect per speaker before writing a
/// reference WAV. OpenVoice TCC's SE extractor is happiest with 5–8 s
/// of voice; less than ~3 s yields a noisy embedding that produces
/// uneven timbre conversion. 6 s is the empirical sweet spot we used
/// during the CosyVoice experiment too.
const REFERENCE_ENROLL_SECONDS: f32 = 6.0;

/// Minimum RMS for a chunk's samples to be admitted into a speaker's
/// reference buffer. References built from breath, room tone or music
/// poison the SE extractor — better to wait longer for clean speech
/// than ship a polluted reference.
const REFERENCE_INGEST_MIN_RMS: f32 = 0.015;

/// Shared buffer of recent translation outputs, used to detect when the
/// loopback captures our own TTS audio (feedback loop). Both pipelines
/// share one buffer so cross-pipeline echo is also detected.
pub type EchoBuffer = Arc<Mutex<VecDeque<(Instant, Vec<String>)>>>;

/// Create a new shared echo buffer for cross-pipeline echo detection.
pub fn new_echo_buffer() -> EchoBuffer {
    Arc::new(Mutex::new(VecDeque::new()))
}

// ─── Voice profile registry: per-speaker F0 + reference WAV enrolment ──────
//
// Two responsibilities, one struct:
//
// 1. Per-speaker running F0 mean (used by Kokoro voice routing to pick
//    a base voice and by the optional pyworld pitch shift).
// 2. Per-speaker reference-WAV enrolment: as the diariser attributes
//    chunks to a speaker, we accumulate ~6 s of clean audio and write
//    it to disk. The OpenVoice TCC bridge then loads that WAV to
//    extract the speaker's tone-color embedding and rewrites the
//    Kokoro output's timbre to match — distinguishing speakers in
//    documentaries beyond what F0 alone can do.
//
// One registry per pipeline branch, shared with the translate worker
// via `Arc<>` so the TTS stage can read references without crossing
// the tokio select loop.

struct VoiceProfileInner {
    f0_by_speaker: HashMap<u32, f32>,
    /// Audio buffer per speaker; once it reaches `target_samples` we
    /// flush to disk and stop accumulating for that speaker.
    audio_buffers: HashMap<u32, Vec<f32>>,
    /// Disk paths of materialised reference WAVs, keyed by speaker_id.
    references: HashMap<u32, String>,
    /// Speakers we've already enrolled — so subsequent ingests for
    /// them are cheap no-ops instead of growing memory.
    enrolled: HashSet<u32>,
}

pub struct VoiceProfileRegistry {
    inner: Mutex<VoiceProfileInner>,
    pipeline_name: String,
    sample_rate: u32,
    target_samples: usize,
    output_dir: PathBuf,
}

impl VoiceProfileRegistry {
    pub(crate) fn new(pipeline_name: impl Into<String>, sample_rate: u32) -> Self {
        let target_samples = (sample_rate as f32 * REFERENCE_ENROLL_SECONDS) as usize;
        let output_dir = std::env::temp_dir().join("meeting_translator_refs");
        let _ = std::fs::create_dir_all(&output_dir);
        Self {
            inner: Mutex::new(VoiceProfileInner {
                f0_by_speaker: HashMap::new(),
                audio_buffers: HashMap::new(),
                references: HashMap::new(),
                enrolled: HashSet::new(),
            }),
            pipeline_name: pipeline_name.into(),
            sample_rate,
            target_samples,
            output_dir,
        }
    }

    /// Mix `f0_hz` into `speaker_id`'s running mean. Silently skips
    /// implausible values (the diarizer returns 0.0 when no voiced frame
    /// could be detected, and very high/low values are usually pyworld
    /// noise on near-silent audio).
    pub(crate) fn record_f0(&self, speaker_id: u32, f0_hz: f32) {
        if !(F0_MIN_HZ..=F0_MAX_HZ).contains(&f0_hz) {
            return;
        }
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let entry = inner.f0_by_speaker.entry(speaker_id).or_insert(f0_hz);
        *entry = *entry * (1.0 - F0_RUNNING_MEAN_ALPHA)
            + f0_hz * F0_RUNNING_MEAN_ALPHA;
    }

    /// Return the running mean F0 for `speaker_id`, or 0.0 when no F0
    /// has ever been recorded for them. The TTS stage interprets 0.0
    /// as "use the default Kokoro voice unchanged".
    pub(crate) fn f0_for(&self, speaker_id: u32) -> f32 {
        self.inner
            .lock()
            .ok()
            .and_then(|g| g.f0_by_speaker.get(&speaker_id).copied())
            .unwrap_or(0.0)
    }

    /// Append fresh audio for `speaker_id` to the enrolment buffer.
    /// Once we have `REFERENCE_ENROLL_SECONDS` of clean speech the
    /// buffer is flushed to a WAV under the temp directory and the
    /// resulting path becomes available via `reference_for`. Calls
    /// after enrolment for the same speaker are cheap no-ops.
    pub(crate) fn ingest_audio(&self, speaker_id: u32, samples: &[f32]) {
        // Reject low-energy chunks before they hit the buffer — silence
        // and music poison the OpenVoice SE extractor far more than a
        // few extra seconds of waiting cost us.
        let rms = (samples.iter().map(|s| s * s).sum::<f32>()
            / samples.len().max(1) as f32)
            .sqrt();
        if rms < REFERENCE_INGEST_MIN_RMS {
            return;
        }

        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return,
        };

        if inner.enrolled.contains(&speaker_id) {
            return;
        }

        let buffer = inner.audio_buffers.entry(speaker_id).or_default();
        buffer.extend_from_slice(samples);
        if buffer.len() < self.target_samples {
            return;
        }

        let snapshot: Vec<f32> = buffer[..self.target_samples].to_vec();
        let path = self.output_dir.join(format!(
            "ref_{}_{}.wav",
            self.pipeline_name.to_lowercase(),
            speaker_id,
        ));

        match write_mono_pcm_wav(&path, &snapshot, self.sample_rate) {
            Ok(()) => {
                let path_str = path.to_string_lossy().to_string();
                tracing::info!(
                    "[{}] Enrolled speaker {}: {} ({:.1}s)",
                    self.pipeline_name,
                    speaker_id,
                    path_str,
                    REFERENCE_ENROLL_SECONDS,
                );
                inner.references.insert(speaker_id, path_str);
                inner.enrolled.insert(speaker_id);
                inner.audio_buffers.remove(&speaker_id);
            }
            Err(e) => {
                tracing::warn!(
                    "[{}] Failed to write reference for speaker {}: {}",
                    self.pipeline_name,
                    speaker_id,
                    e,
                );
            }
        }
    }

    /// Path of the reference WAV for `speaker_id`, if enrolment has
    /// completed. `None` means "no conversion possible yet — use raw
    /// Kokoro output for this speaker".
    pub(crate) fn reference_for(&self, speaker_id: u32) -> Option<String> {
        self.inner
            .lock()
            .ok()
            .and_then(|g| g.references.get(&speaker_id).cloned())
    }
}

/// Mono 16-bit PCM WAV writer — small enough to keep here instead of
/// pulling a WAV crate just for reference enrolment. Format is fixed
/// (mono, 16-bit signed) because that's what every downstream consumer
/// (OpenVoice's SE extractor, our debug tools) expects.
fn write_mono_pcm_wav(path: &Path, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
    let data_bytes = (samples.len() * 2) as u32;
    let chunk_size = 36 + data_bytes;
    let byte_rate = sample_rate * 2;

    let mut file = File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&chunk_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;          // subchunk1 size
    file.write_all(&1u16.to_le_bytes())?;           // format = PCM
    file.write_all(&1u16.to_le_bytes())?;           // channels = mono
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?;           // block align
    file.write_all(&16u16.to_le_bytes())?;          // bits per sample
    file.write_all(b"data")?;
    file.write_all(&data_bytes.to_le_bytes())?;

    for &sample in samples {
        let clipped = sample.clamp(-1.0, 1.0);
        let int16 = (clipped * 32767.0) as i16;
        file.write_all(&int16.to_le_bytes())?;
    }
    Ok(())
}


// ─── Translation quality guard ──────────────────────────────────────────────

fn is_translation_degenerate(input: &str, output: &str) -> bool {
    let input_words = input.split_whitespace().count().max(1);
    let output_words = output.split_whitespace().count();
    if output_words > input_words * 4 && output_words > 20 {
        return true;
    }

    let words: Vec<&str> = output.split_whitespace().collect();
    if words.len() >= 6 {
        let unique: std::collections::HashSet<&str> = words.iter().copied().collect();
        let ratio = unique.len() as f32 / words.len() as f32;
        if ratio < 0.25 {
            return true;
        }
    }

    false
}

// ─── Echo detection (TTS feedback loop filter) ──────────────────────────────

fn normalize_for_echo(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|w| {
            w.chars()
                .filter_map(|c| {
                    if c.is_ascii_alphanumeric() {
                        Some(c)
                    } else if c.is_alphanumeric() {
                        Some(strip_diacritic(c))
                    } else {
                        None
                    }
                })
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

fn strip_diacritic(c: char) -> char {
    match c {
        'á' | 'à' | 'â' | 'ã' | 'ä' => 'a',
        'é' | 'è' | 'ê' | 'ë' => 'e',
        'í' | 'ì' | 'î' | 'ï' => 'i',
        'ó' | 'ò' | 'ô' | 'õ' | 'ö' => 'o',
        'ú' | 'ù' | 'û' | 'ü' => 'u',
        'ç' => 'c',
        'ñ' => 'n',
        _ => c,
    }
}

fn record_translation(echo_buffer: &EchoBuffer, translated_text: &str) {
    let words = normalize_for_echo(translated_text);
    if words.is_empty() {
        return;
    }
    let mut buf = echo_buffer.lock().unwrap();
    buf.push_back((Instant::now(), words));
    let cutoff = Instant::now() - std::time::Duration::from_secs_f32(ECHO_WINDOW_SECONDS);
    while buf.front().map_or(false, |(t, _)| *t < cutoff) {
        buf.pop_front();
    }
}

fn is_echo(stt_text: &str, echo_buffer: &EchoBuffer) -> bool {
    let stt_words = normalize_for_echo(stt_text);
    if stt_words.is_empty() {
        return false;
    }

    let buf = echo_buffer.lock().unwrap();
    let cutoff = Instant::now() - std::time::Duration::from_secs_f32(ECHO_WINDOW_SECONDS);

    for (timestamp, translation_words) in buf.iter() {
        if *timestamp < cutoff {
            continue;
        }
        let overlap = word_overlap_ratio(&stt_words, translation_words);
        if overlap >= ECHO_SIMILARITY_THRESHOLD {
            return true;
        }
    }
    false
}

fn word_overlap_ratio(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    let b_set: std::collections::HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    let matches = a.iter().filter(|w| b_set.contains(w.as_str())).count();
    matches as f32 / a.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    const WHISPER_SAMPLE_RATE: u32 = 16_000;

    #[test]
    fn voice_profile_registry_records_and_recalls() {
        let reg = VoiceProfileRegistry::new("test", WHISPER_SAMPLE_RATE);
        reg.record_f0(0, 200.0);
        reg.record_f0(0, 200.0);
        reg.record_f0(0, 200.0);
        let f0 = reg.f0_for(0);
        assert!((f0 - 200.0).abs() < 5.0);  // converged towards 200 Hz
    }

    #[test]
    fn voice_profile_registry_returns_zero_for_unknown_speaker() {
        let reg = VoiceProfileRegistry::new("test", WHISPER_SAMPLE_RATE);
        assert_eq!(reg.f0_for(42), 0.0);
    }

    #[test]
    fn voice_profile_registry_clamps_outliers() {
        let reg = VoiceProfileRegistry::new("test", WHISPER_SAMPLE_RATE);
        reg.record_f0(0, 200.0);
        reg.record_f0(0, 5000.0);  // pyworld noise — must be ignored
        let f0 = reg.f0_for(0);
        assert!((f0 - 200.0).abs() < 5.0);
    }

    #[test]
    fn repetitive_translation_is_degenerate() {
        let input = "No, no, no";
        let output = "Não, não, não, não, não, não, não, não, não, não, não, não, não, não, não, não, não, não, não, não, não, não, não";
        assert!(is_translation_degenerate(input, output));
    }

    #[test]
    fn echo_detected_when_stt_matches_recent_translation() {
        let buf: EchoBuffer = Arc::new(Mutex::new(VecDeque::new()));
        record_translation(&buf, "plataforma petrolífera");
        assert!(is_echo("platforma petrolifera", &buf));
    }
}
