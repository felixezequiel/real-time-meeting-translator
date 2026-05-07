//! Speaker pipeline V2 — adaptive sliding-window per ADR 0013.
//!
//! Replaces the streaming local-agreement path (ADR 0004) with a
//! phrase-aligned pipeline:
//!
//! ```text
//! [audio_input] → [VAD] → [PhraseSegmenter] → [STT batch] →
//!     [is_echo? translate? TTS? TCC?] → [audio_output]
//! ```
//!
//! Each window is a complete utterance (closed by silence or by the
//! adaptive max). Whisper, the translator and TTS receive a coherent
//! phrase instead of fragments — which fixes the "translation
//! horrível" problem and removes the need for streaming local-
//! agreement, accumulator state, or punctuation-driven flushes.
//!
//! V2 starts simple: it does **not** do per-speaker F0 tracking or
//! auto-enrolment of voice references. Those are Phase 2 work tracked
//! in ADR 0013. The Mic side (which already had a fixed voice
//! profile) keeps full TCC support; the loopback side uses raw TTS
//! until per-speaker enrolment lands.

use audio::phrase_segmenter::{PhraseSegmenter, PhraseSegmenterConfig};
use audio::SileroVad;
use diarization::OnlineDiarizer;
use shared::{AudioChunk, Language, PipelineCommand, PipelineMetrics};
use stt::WhisperStt;
use tokio::sync::mpsc;
use tracing;
use translation::OpusMtTranslator;
use tts::{PiperTts, VoiceProfile};
use voice_convert::ToneColorConverter;

use std::sync::{Arc, Mutex};
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};

/// How long the accumulator may hold a not-yet-final phrase before
/// forcing a flush. Bounds the worst-case wait when a speaker never
/// produces a clean punctuation boundary.
const ACCUMULATOR_MAX_HOLD_MS: u128 = 3000;

/// Word count at which the accumulator force-flushes even without
/// punctuation. Long monologues without commas would otherwise just
/// keep growing.
const ACCUMULATOR_MAX_WORDS: usize = 35;

/// Per-pipeline accumulator state. Shared between concurrent
/// `process_segment` tasks via `Arc<Mutex<>>` so they serialise on
/// the merge/flush decision.
#[derive(Default)]
struct Accumulator {
    pending_text: String,
    pending_speaker_id: Option<u32>,
    pending_started_at: Option<Instant>,
    /// Diariser-supplied F0 of the most recent contributor — passed
    /// to TTS at flush time so the voice picker has a target.
    pending_f0_hz: f32,
    /// Cached reference WAV path for TCC: for the mic side this is
    /// the user's profile, for the speaker side it's whatever the
    /// last identified speaker has enrolled (or none).
    pending_reference_path: Option<String>,
}

fn ends_with_punctuation(text: &str) -> bool {
    matches!(
        text.trim().chars().last(),
        Some('.' | '!' | '?' | ';' | '。' | '；'),
    )
}

use crate::{is_echo, is_translation_degenerate, record_translation, EchoBuffer, VoiceProfileRegistry};

const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Energy floor used when Silero VAD is unavailable. Mirrors the
/// fallback in ADR 0008 / the legacy pipeline so behaviour stays
/// consistent on a fresh checkout.
const FALLBACK_MIN_RMS_FOR_STT: f32 = 0.003;

#[derive(Debug, Clone, Copy)]
pub struct V2Config {
    pub max_window: Duration,
    pub silence_tail: Duration,
    pub min_window: Duration,
}

impl Default for V2Config {
    fn default() -> Self {
        Self {
            max_window: Duration::from_millis(5000),
            silence_tail: Duration::from_millis(400),
            min_window: Duration::from_millis(600),
        }
    }
}

impl From<V2Config> for PhraseSegmenterConfig {
    fn from(c: V2Config) -> Self {
        PhraseSegmenterConfig {
            max_window: c.max_window,
            silence_tail: c.silence_tail,
            min_window: c.min_window,
        }
    }
}

pub struct SpeakerPipelineV2 {
    pub name: String,
    pub stt: Arc<WhisperStt>,
    pub translator: Arc<OpusMtTranslator>,
    pub tts: Arc<PiperTts>,
    pub source_language: Language,
    pub echo_buffer: EchoBuffer,
    pub vad: Option<Arc<SileroVad>>,
    pub diarizer: Option<Arc<OnlineDiarizer>>,
    pub voice_convert: Option<Arc<ToneColorConverter>>,
    pub fixed_voice_reference: Option<String>,
    pub config: V2Config,
    /// Optional channel for subtitle text events. When present, every
    /// translated phrase is forwarded to the overlay UI before TTS.
    /// Uses `std::sync::mpsc` because the send happens inside
    /// `spawn_blocking` (no async context).
    pub subtitle_tx: Option<std_mpsc::Sender<SubtitleEvent>>,
}

#[derive(Debug, Clone)]
pub struct SubtitleEvent {
    pub pipeline_name: String,
    pub source_text: String,
    pub translated_text: String,
    pub language: Language,
    pub timestamp: Instant,
}

impl SpeakerPipelineV2 {
    pub fn new(
        name: impl Into<String>,
        stt: Arc<WhisperStt>,
        translator: Arc<OpusMtTranslator>,
        tts: Arc<PiperTts>,
        source_language: Language,
        echo_buffer: EchoBuffer,
    ) -> Self {
        Self {
            name: name.into(),
            stt,
            translator,
            tts,
            source_language,
            echo_buffer,
            vad: None,
            diarizer: None,
            voice_convert: None,
            fixed_voice_reference: None,
            config: V2Config::default(),
            subtitle_tx: None,
        }
    }

    pub fn with_vad(mut self, vad: Arc<SileroVad>) -> Self {
        self.vad = Some(vad);
        self
    }

    pub fn with_diarizer(mut self, diarizer: Arc<OnlineDiarizer>) -> Self {
        self.diarizer = Some(diarizer);
        self
    }

    pub fn with_voice_convert(mut self, vc: Arc<ToneColorConverter>) -> Self {
        self.voice_convert = Some(vc);
        self
    }

    pub fn with_fixed_voice_reference(mut self, path: impl Into<String>) -> Self {
        self.fixed_voice_reference = Some(path.into());
        self
    }

    pub fn with_subtitle_channel(
        mut self,
        tx: std_mpsc::Sender<SubtitleEvent>,
    ) -> Self {
        self.subtitle_tx = Some(tx);
        self
    }

    pub fn with_config(mut self, config: V2Config) -> Self {
        self.config = config;
        self
    }

    pub async fn run(
        self,
        mut audio_input: mpsc::UnboundedReceiver<AudioChunk>,
        audio_output: mpsc::UnboundedSender<AudioChunk>,
        mut command_rx: mpsc::Receiver<PipelineCommand>,
        metrics_tx: mpsc::Sender<PipelineMetrics>,
    ) {
        let pipeline_name = self.name;
        let mut segmenter = PhraseSegmenter::new(
            WHISPER_SAMPLE_RATE,
            self.config.into(),
        );
        let min_speech_samples = segmenter.min_window_samples();
        // Voice profile registry: collects ~6 s of clean audio per
        // diarised speaker, writes a reference WAV, and exposes its
        // path so OpenVoice TCC can convert subsequent TTS output to
        // that speaker's timbre. Without diariser this stays empty
        // and the pipeline falls back to raw Kokoro output.
        let voice_profiles = Arc::new(VoiceProfileRegistry::new(
            pipeline_name.clone(),
            WHISPER_SAMPLE_RATE,
        ));
        // Inter-window phrase accumulator (ADR 0013 follow-up,
        // 2026-05-07): each closed segment's transcription appends
        // here, and translation+TTS only fire when the accumulated
        // text ends in punctuation, the speaker changes, or it has
        // been held longer than ACCUMULATOR_MAX_HOLD_MS. Solves the
        // "frase cortada no meio" quality regression that came from
        // shrinking silence_tail to 280 ms for latency.
        let accumulator = Arc::new(Mutex::new(Accumulator::default()));
        let mut is_running = false;

        tracing::info!(
            "[{}] V2 ready (max_window={:?}, silence_tail={:?}, min_window={:?}, diariser={}, vc={})",
            pipeline_name,
            self.config.max_window,
            self.config.silence_tail,
            self.config.min_window,
            self.diarizer.is_some(),
            self.voice_convert.is_some(),
        );

        loop {
            tokio::select! {
                cmd = command_rx.recv() => {
                    match cmd {
                        Some(PipelineCommand::Start) => {
                            tracing::info!(
                                "[{}] V2 pipeline started (source={})",
                                pipeline_name,
                                self.source_language.display_name(),
                            );
                            is_running = true;
                            if let Some(v) = self.vad.as_ref() {
                                v.reset_state();
                            }
                            let _ = segmenter.flush();
                        }
                        Some(PipelineCommand::Stop) => {
                            tracing::info!("[{}] V2 pipeline stopped", pipeline_name);
                            is_running = false;
                            let _ = segmenter.flush();
                            if let Ok(mut acc) = accumulator.lock() {
                                *acc = Accumulator::default();
                            }
                        }
                        None => return,
                    }
                }
                chunk = audio_input.recv() => {
                    let chunk = match chunk {
                        Some(c) => c,
                        None => return,
                    };
                    if !is_running { continue; }

                    let vad_start = Instant::now();
                    let is_speech = match self.vad.as_ref() {
                        Some(v) => v.has_speech(&chunk.samples),
                        None => rms(&chunk.samples) >= FALLBACK_MIN_RMS_FOR_STT,
                    };
                    let _ = metrics_tx.try_send(PipelineMetrics::new(
                        "vad".to_string(),
                        vad_start.elapsed(),
                    ));

                    let segment = segmenter.ingest(&chunk.samples, is_speech);
                    if let Some(segment) = segment {
                        if segment.speech_samples < min_speech_samples {
                            tracing::debug!(
                                "[{}] dropping short window ({} speech samples < {} min)",
                                pipeline_name,
                                segment.speech_samples,
                                min_speech_samples,
                            );
                            continue;
                        }

                        let stt = Arc::clone(&self.stt);
                        let translator = Arc::clone(&self.translator);
                        let tts_engine = Arc::clone(&self.tts);
                        let voice_convert = self.voice_convert.clone();
                        let diarizer = self.diarizer.clone();
                        let voice_profiles = Arc::clone(&voice_profiles);
                        let echo_buffer = Arc::clone(&self.echo_buffer);
                        let audio_output = audio_output.clone();
                        let metrics_tx = metrics_tx.clone();
                        let pipeline_name = pipeline_name.clone();
                        let source_language = self.source_language;
                        let fixed_voice_reference = self.fixed_voice_reference.clone();
                        let subtitle_tx = self.subtitle_tx.clone();
                        let segment_samples = segment.samples;
                        let segment_sample_rate = segment.sample_rate;
                        let accumulator = Arc::clone(&accumulator);

                        tokio::task::spawn_blocking(move || {
                            process_segment(
                                pipeline_name,
                                segment_samples,
                                segment_sample_rate,
                                stt,
                                translator,
                                tts_engine,
                                voice_convert,
                                diarizer,
                                voice_profiles,
                                echo_buffer,
                                source_language,
                                fixed_voice_reference,
                                audio_output,
                                metrics_tx,
                                subtitle_tx,
                                accumulator,
                            );
                        });
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn process_segment(
    pipeline_name: String,
    samples: Vec<f32>,
    sample_rate: u32,
    stt: Arc<WhisperStt>,
    translator: Arc<OpusMtTranslator>,
    tts: Arc<PiperTts>,
    voice_convert: Option<Arc<ToneColorConverter>>,
    diarizer: Option<Arc<OnlineDiarizer>>,
    voice_profiles: Arc<VoiceProfileRegistry>,
    echo_buffer: EchoBuffer,
    source_language: Language,
    fixed_voice_reference: Option<String>,
    audio_output: mpsc::UnboundedSender<AudioChunk>,
    metrics_tx: mpsc::Sender<PipelineMetrics>,
    subtitle_tx: Option<std_mpsc::Sender<SubtitleEvent>>,
    accumulator: Arc<Mutex<Accumulator>>,
) {
    let total_start = Instant::now();

    // ─── Diariser ─────────────────────────────────────────────────────
    // Run BEFORE STT so we can route the right voice reference into TCC.
    // When fixed_voice_reference is set (mic side), skip diariser entirely
    // — only one person ever speaks into the mic, no need to identify.
    let diar_start = Instant::now();
    let identification = if fixed_voice_reference.is_some() {
        None
    } else {
        diarizer.as_ref().and_then(|d| match d.identify(&samples, sample_rate) {
            Ok(opt) => opt,
            Err(e) => {
                tracing::warn!("[{}] V2 diariser failed: {}", pipeline_name, e);
                None
            }
        })
    };
    let _ = metrics_tx.try_send(PipelineMetrics::new(
        "diarize".to_string(),
        diar_start.elapsed(),
    ));

    let speaker_id = identification.as_ref().map(|s| s.speaker_id);
    if let Some(id) = identification.as_ref() {
        voice_profiles.record_f0(id.speaker_id, id.f0_hz);
        // Auto-enrolment: feed the segment audio into this speaker's
        // reference buffer. Once it has 6 s of clean speech a WAV
        // reference is written and TCC can convert future TTS for the
        // same speaker.
        voice_profiles.ingest_audio(id.speaker_id, &samples);
    }

    let chunk = AudioChunk::new(samples, sample_rate, 1);

    // ─── STT ──────────────────────────────────────────────────────────
    let stt_start = Instant::now();
    let transcribed = match stt.transcribe(&chunk, source_language) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!("[{}] V2 STT failed: {}", pipeline_name, e);
            return;
        }
    };
    let _ = metrics_tx.try_send(PipelineMetrics::new(
        "stt".to_string(),
        stt_start.elapsed(),
    ));

    let source_text = transcribed.text.trim();
    if source_text.is_empty() {
        return;
    }

    if is_echo(source_text, &echo_buffer) {
        tracing::info!(
            "[{}] V2 echo dropped: \"{}\"",
            pipeline_name,
            preview(source_text, 60),
        );
        return;
    }

    tracing::info!(
        "[{}] V2 STT: \"{}\"",
        pipeline_name,
        preview(source_text, 80),
    );

    // ─── Update accumulator and decide what to flush ─────────────────
    // The accumulator holds a sentence-in-progress across multiple
    // closed phrase windows. We translate + TTS only when we hit a
    // real sentence boundary (punctuation), the speaker changes, the
    // text grows past a hard cap, or it has been held longer than
    // ACCUMULATOR_MAX_HOLD_MS.
    let now = Instant::now();
    let resolved_reference = fixed_voice_reference
        .clone()
        .or_else(|| speaker_id.and_then(|id| voice_profiles.reference_for(id)));
    let f0_for_tts = match speaker_id {
        Some(id) => voice_profiles.f0_for(id),
        None => 0.0,
    };

    // Up to two flushes can come out of a single ingest:
    //   1. an "early flush" when the speaker changed and we have a
    //      pending phrase from the previous speaker;
    //   2. a "main flush" when the (possibly extended) accumulator
    //      now ends in punctuation, has aged out, or is long enough.
    let (early_flush, main_flush) = {
        let mut acc = accumulator.lock().expect("accumulator poisoned");
        let speaker_changed = matches!(
            (acc.pending_speaker_id, speaker_id),
            (Some(old), Some(new)) if old != new,
        );

        let early = if speaker_changed && !acc.pending_text.is_empty() {
            let drained = std::mem::take(&mut acc.pending_text);
            let snapshot = (
                drained,
                acc.pending_speaker_id,
                acc.pending_f0_hz,
                acc.pending_reference_path.clone(),
            );
            acc.pending_speaker_id = None;
            acc.pending_started_at = None;
            acc.pending_f0_hz = 0.0;
            acc.pending_reference_path = None;
            Some(snapshot)
        } else {
            None
        };

        if !acc.pending_text.is_empty() {
            acc.pending_text.push(' ');
        }
        acc.pending_text.push_str(source_text);
        if acc.pending_speaker_id.is_none() {
            acc.pending_speaker_id = speaker_id;
        }
        if acc.pending_started_at.is_none() {
            acc.pending_started_at = Some(now);
        }
        if f0_for_tts > 0.0 {
            acc.pending_f0_hz = f0_for_tts;
        }
        if resolved_reference.is_some() {
            acc.pending_reference_path = resolved_reference.clone();
        }

        let aged_out = acc
            .pending_started_at
            .map(|t| now.duration_since(t).as_millis() > ACCUMULATOR_MAX_HOLD_MS)
            .unwrap_or(false);
        let punct = ends_with_punctuation(&acc.pending_text);
        let long_enough = acc.pending_text.split_whitespace().count() >= ACCUMULATOR_MAX_WORDS;
        let main = if aged_out || punct || long_enough {
            let drained = std::mem::take(&mut acc.pending_text);
            let snapshot = (
                drained,
                acc.pending_speaker_id,
                acc.pending_f0_hz,
                acc.pending_reference_path.clone(),
            );
            acc.pending_speaker_id = None;
            acc.pending_started_at = None;
            acc.pending_f0_hz = 0.0;
            acc.pending_reference_path = None;
            Some(snapshot)
        } else {
            None
        };

        (early, main)
    };

    // The phrase wasn't closed — log STT progress and bail without
    // translating. The text stays in the accumulator for the next
    // segment to extend.
    if early_flush.is_none() && main_flush.is_none() {
        tracing::info!(
            "[{}] V2 STT (held): \"{}\"",
            pipeline_name,
            preview(source_text, 80),
        );
        let _ = metrics_tx.try_send(PipelineMetrics::new(
            "total".to_string(),
            total_start.elapsed(),
        ));
        return;
    }

    if let Some((text, sid, f0, reference)) = early_flush {
        flush_phrase(
            &pipeline_name, &text, sid, f0, reference,
            source_language,
            &translator, &tts, voice_convert.as_ref(),
            &echo_buffer,
            &audio_output, &metrics_tx,
            subtitle_tx.as_ref(),
        );
    }
    if let Some((text, sid, f0, reference)) = main_flush {
        flush_phrase(
            &pipeline_name, &text, sid, f0, reference,
            source_language,
            &translator, &tts, voice_convert.as_ref(),
            &echo_buffer,
            &audio_output, &metrics_tx,
            subtitle_tx.as_ref(),
        );
    }

    let _ = metrics_tx.try_send(PipelineMetrics::new(
        "total".to_string(),
        total_start.elapsed(),
    ));
}

#[allow(clippy::too_many_arguments)]
fn flush_phrase(
    pipeline_name: &str,
    source_text: &str,
    speaker_id: Option<u32>,
    target_f0_hz: f32,
    reference_path: Option<String>,
    source_language: Language,
    translator: &OpusMtTranslator,
    tts: &PiperTts,
    voice_convert: Option<&Arc<ToneColorConverter>>,
    echo_buffer: &EchoBuffer,
    audio_output: &mpsc::UnboundedSender<AudioChunk>,
    metrics_tx: &mpsc::Sender<PipelineMetrics>,
    subtitle_tx: Option<&std_mpsc::Sender<SubtitleEvent>>,
) {
    use shared::TextSegment;
    let trimmed = source_text.trim();
    if trimmed.is_empty() {
        return;
    }

    let segment = TextSegment::new(trimmed.to_string(), source_language);

    let translate_start = Instant::now();
    let translated = match translator.translate(&segment) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!("[{}] V2 translation failed: {}", pipeline_name, e);
            return;
        }
    };
    let _ = metrics_tx.try_send(PipelineMetrics::new(
        "translate".to_string(),
        translate_start.elapsed(),
    ));

    let translated_text = translated.text.trim();
    if translated_text.is_empty() {
        return;
    }
    if is_translation_degenerate(trimmed, translated_text) {
        tracing::info!(
            "[{}] V2 degenerate translation dropped: \"{}\" → \"{}\"",
            pipeline_name,
            preview(trimmed, 40),
            preview(translated_text, 40),
        );
        return;
    }

    record_translation(echo_buffer, translated_text);

    if let Some(tx) = subtitle_tx {
        let _ = tx.send(SubtitleEvent {
            pipeline_name: pipeline_name.to_string(),
            source_text: trimmed.to_string(),
            translated_text: translated_text.to_string(),
            language: translated.language,
            timestamp: Instant::now(),
        });
    }

    tracing::info!(
        "[{}] V2 → \"{}\"",
        pipeline_name,
        preview(translated_text, 80),
    );

    let tts_start = Instant::now();
    let voice_profile = match speaker_id {
        Some(id) => VoiceProfile {
            target_f0_hz,
            formant_shift: 1.0,
            speaker_id: Some(id),
        },
        None => VoiceProfile::default(),
    };
    let tts_audio = match tts.synthesize(&translated, voice_profile) {
        Ok(a) => a,
        Err(e) => {
            tracing::warn!("[{}] V2 TTS failed: {}", pipeline_name, e);
            return;
        }
    };
    let _ = metrics_tx.try_send(PipelineMetrics::new(
        "tts".to_string(),
        tts_start.elapsed(),
    ));

    const TCC_MIN_DURATION_MS: u64 = 500;
    let tts_duration_ms =
        (tts_audio.samples.len() as u64 * 1000) / tts_audio.sample_rate.max(1) as u64;
    let speaker_for_tcc = speaker_id.unwrap_or(0);
    let final_audio = match (voice_convert, reference_path.as_deref()) {
        (Some(vc), Some(reference)) if tts_duration_ms >= TCC_MIN_DURATION_MS => {
            let vc_start = Instant::now();
            let converted = match vc.convert(&tts_audio, reference, speaker_for_tcc) {
                Ok(Some(c)) => Some(AudioChunk::new(c.samples, c.sample_rate, 1)),
                Ok(None) => None,
                Err(e) => {
                    tracing::warn!("[{}] V2 voice convert failed: {}", pipeline_name, e);
                    None
                }
            };
            let _ = metrics_tx.try_send(PipelineMetrics::new(
                "voice_convert".to_string(),
                vc_start.elapsed(),
            ));
            converted.unwrap_or(tts_audio)
        }
        (Some(_), Some(_)) => {
            tracing::debug!(
                "[{}] V2 TCC skipped: phrase too short ({}ms < {}ms)",
                pipeline_name,
                tts_duration_ms,
                TCC_MIN_DURATION_MS,
            );
            tts_audio
        }
        _ => tts_audio,
    };

    let _ = audio_output.send(final_audio);
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// UTF-8-safe truncation for log preview strings. Byte slicing
/// (`&s[..n]`) panics when `n` lands inside a multi-byte char — and
/// Portuguese has plenty (á, ã, ç, é, …). This walks codepoints
/// instead so 80 means "up to 80 characters", not "up to 80 bytes".
fn preview(text: &str, max_chars: usize) -> &str {
    match text.char_indices().nth(max_chars) {
        Some((idx, _)) => &text[..idx],
        None => text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ends_with_punctuation_recognises_standard_marks() {
        assert!(ends_with_punctuation("Hello world."));
        assert!(ends_with_punctuation("Are you sure?"));
        assert!(ends_with_punctuation("Wait!"));
        assert!(ends_with_punctuation("First item;"));
    }

    #[test]
    fn ends_with_punctuation_handles_trailing_whitespace() {
        assert!(ends_with_punctuation("Hello world.   "));
        assert!(ends_with_punctuation("\nDone.\n"));
    }

    #[test]
    fn ends_with_punctuation_returns_false_on_open_clauses() {
        assert!(!ends_with_punctuation("the market has been"));
        assert!(!ends_with_punctuation("we're getting started"));
        assert!(!ends_with_punctuation(""));
    }

    #[test]
    fn ends_with_punctuation_handles_portuguese_accents() {
        // Trailing 'á' must NOT confuse byte-vs-char logic.
        assert!(!ends_with_punctuation("você está"));
        assert!(ends_with_punctuation("você está aqui."));
    }

    #[test]
    fn preview_clamps_by_chars_not_bytes() {
        // 'á' is 2 bytes; "fala você" is 9 chars / 10 bytes.
        let s = "fala você está aqui";
        let p = preview(s, 9);
        // 9 chars from "fala você está aqui" = "fala você"
        assert_eq!(p, "fala você");
    }

    #[test]
    fn preview_returns_full_string_when_shorter_than_limit() {
        assert_eq!(preview("oi", 80), "oi");
    }
}
