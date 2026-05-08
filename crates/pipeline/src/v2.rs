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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};

/// Monotonically-increasing id for each phrase the pipeline ships
/// to the subtitle overlay. The streaming translator emits multiple
/// `SubtitleEvent`s per phrase (one per clause boundary) — they all
/// share the same `phrase_id` so the UI can update a single line
/// in place instead of stacking N copies of the growing text.
static PHRASE_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_phrase_id() -> u64 {
    PHRASE_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// How long the accumulator may hold a not-yet-final phrase before
/// forcing a flush. Bounds the worst-case wait when a speaker never
/// produces a clean punctuation boundary.
///
/// Lowered from 3000 → 1200 ms on 2026-05-08 (ADR 0013 amendment
/// "Latency-first accumulator"): the previous 3 s ceiling was the
/// dominant TTFA contributor on punctuation-less Whisper output —
/// typical case in spoken speech. 1200 ms keeps the worst-case
/// end-to-end latency under ~2.2 s while the soft-flush rule below
/// catches the *common* case much earlier.
const ACCUMULATOR_MAX_HOLD_MS: u128 = 1200;

/// Word count at which the accumulator force-flushes even without
/// punctuation. Long monologues without commas would otherwise just
/// keep growing.
const ACCUMULATOR_MAX_WORDS: usize = 35;

/// Soft-flush trigger: once the accumulator has at least this many
/// words AND has been held for `ACCUMULATOR_SOFT_FLUSH_HOLD_MS`,
/// release it even without punctuation. This is the lever that
/// brought TTFA from "wait for the dot" to "wait for a phrase-shaped
/// chunk" — Whisper rarely emits commas mid-utterance, so the old
/// punctuation-only flush rule paid the full MAX_HOLD on most spoken
/// input. Six words is roughly the size of a self-contained clause
/// in conversational speech ("we should refactor this module"),
/// large enough that the streaming translator and the interpreter-
/// style prompt still produce coherent output.
const ACCUMULATOR_SOFT_FLUSH_WORDS: usize = 6;
const ACCUMULATOR_SOFT_FLUSH_HOLD_MS: u128 = 800;

/// Pure flush decision used by `process_segment` and unit-tested in
/// isolation. Returns `true` when the accumulator has a forced reason
/// to release its content even when the buffer doesn't end on a
/// sentence boundary. Three triggers, in increasing latency cost:
///
/// 1. **Soft flush** — `word_count ≥ ACCUMULATOR_SOFT_FLUSH_WORDS`
///    AND `age_ms ≥ ACCUMULATOR_SOFT_FLUSH_HOLD_MS`. Catches the
///    common case where the speaker said one full clause but Whisper
///    didn't punctuate it.
/// 2. **Long enough** — `word_count ≥ ACCUMULATOR_MAX_WORDS`.
///    Comma-less monologue.
/// 3. **Aged out** — `age_ms > ACCUMULATOR_MAX_HOLD_MS`. Hard ceiling
///    on how long any phrase may sit pending.
fn should_force_flush(word_count: usize, age_ms: u128) -> bool {
    let aged_out = age_ms > ACCUMULATOR_MAX_HOLD_MS;
    let long_enough = word_count >= ACCUMULATOR_MAX_WORDS;
    let soft_flush =
        word_count >= ACCUMULATOR_SOFT_FLUSH_WORDS
            && age_ms >= ACCUMULATOR_SOFT_FLUSH_HOLD_MS;
    aged_out || long_enough || soft_flush
}

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

/// Pull all complete sentences from the front of `text`, leaving any
/// in-progress trailing fragment in the second return value. Used by
/// the accumulator so a buffer like *"sentence one. sentence two. open"*
/// flushes the first two sentences IMMEDIATELY instead of waiting for
/// the trailing fragment to finally close — without this the listener
/// can wait 6+ s for a single sentence's audio.
///
/// Boundary rule: ASCII sentence-ending punctuation (`.!?;`) followed
/// by whitespace, or by end-of-string. Periods inside acronyms or
/// numbers ("U.S.A", "3.14") are NOT boundaries because the next byte
/// is alphanumeric, not whitespace. Safe on UTF-8 because we only
/// index by single-byte ASCII positions.
fn split_complete_sentences(text: &str) -> (String, String) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return (String::new(), String::new());
    }
    let bytes = trimmed.as_bytes();
    let mut last_boundary: Option<usize> = None;
    for i in 0..bytes.len() {
        if !matches!(bytes[i], b'.' | b'!' | b'?' | b';') {
            continue;
        }
        let next = i + 1;
        let is_boundary = match bytes.get(next) {
            None => true,
            Some(b) if b.is_ascii_whitespace() => true,
            // Run of consecutive punctuation ("?!", "..."): keep
            // scanning, the LATER position wins.
            Some(b) if matches!(*b, b'.' | b'!' | b'?' | b';') => false,
            _ => false,
        };
        if is_boundary {
            last_boundary = Some(next);
        }
    }
    match last_boundary {
        Some(idx) => {
            let complete = trimmed[..idx].trim().to_string();
            let rest = trimmed[idx..].trim_start().to_string();
            (complete, rest)
        }
        None => (String::new(), trimmed.to_string()),
    }
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
    /// All events emitted within a single flushed phrase share the
    /// same `phrase_id` — the UI replaces (rather than stacks) the
    /// existing line when it sees the same id. New phrases get a
    /// fresh id and become a new line in the overlay.
    pub phrase_id: u64,
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

        let pending_age_ms = acc
            .pending_started_at
            .map(|t| now.duration_since(t).as_millis())
            .unwrap_or(0);
        let pending_word_count = acc.pending_text.split_whitespace().count();
        let force_flush = should_force_flush(pending_word_count, pending_age_ms);

        // Try to peel off any complete sentences that have already
        // accumulated, leaving the still-in-progress trailing
        // fragment in `acc.pending_text`. This is the win that
        // dropped TTFA dramatically on multi-sentence speech: e.g.
        // a buffer like "Sentence one. Sentence two. fragment"
        // flushes the first two sentences NOW instead of waiting
        // until the final fragment finally closes (which in field
        // logs took 6+ s).
        let (complete, remaining) = split_complete_sentences(&acc.pending_text);

        let main = if !complete.is_empty() {
            let snapshot = (
                complete,
                acc.pending_speaker_id,
                acc.pending_f0_hz,
                acc.pending_reference_path.clone(),
            );
            acc.pending_text = remaining;
            if acc.pending_text.is_empty() {
                acc.pending_speaker_id = None;
                acc.pending_started_at = None;
                acc.pending_f0_hz = 0.0;
                acc.pending_reference_path = None;
            } else {
                // Trailing fragment continues — keep speaker/F0/ref
                // but reset the started_at clock so MAX_HOLD applies
                // to the fragment alone, not to the original phrase.
                acc.pending_started_at = Some(now);
            }
            Some(snapshot)
        } else if force_flush {
            // No complete sentences but a force-flush trigger fired
            // (soft-flush at 6 words / 800 ms, MAX_WORDS, or
            // MAX_HOLD). Drain the accumulator so the listener
            // doesn't wait forever on a comma-less utterance.
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

/// Streaming-friendly minimum TTS duration before TCC is invoked.
/// OpenVoice preprocessing fails on very short audio (< 500 ms),
/// so we skip TCC for those fragments and play raw Kokoro instead.
const TCC_MIN_DURATION_MS: u64 = 500;

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
    let target_language = match source_language {
        Language::English => Language::Portuguese,
        Language::Portuguese => Language::English,
    };

    // ─── Streaming translation + per-fragment TTS dispatch ────────────
    // The translator emits one TranslationFragment per commit-eligible
    // clause (comma, period, semicolon). We synthesise each clause as
    // soon as it lands and push it into the playback mixer — so TTS
    // for clause A starts playing while the translator is still
    // generating clause B. This drops time-to-first-audio from
    // ~translate_full + tts_full (~900-1500 ms) to ~first_token +
    // first_clause_tts (~300-500 ms), which is the metric the user
    // perceives as "delay".
    let translate_start = Instant::now();
    let mut full_translation = String::new();
    let mut first_fragment_seen = false;

    // VoiceProfile carries everything either engine might want:
    //   - Kokoro reads target_f0_hz / formant_shift / speaker_id and
    //     ignores reference_wav_path (its sticky-voice + pitch-shift
    //     pipeline is fully self-contained — ADR 0010).
    //   - XTTS-v2 reads reference_wav_path and ignores the rest
    //     (zero-shot cloning conditioned on the reference audio —
    //     ADR 0014).
    // We populate every field; each engine picks what it needs.
    let voice_profile = VoiceProfile {
        target_f0_hz,
        formant_shift: 1.0,
        speaker_id,
        reference_wav_path: reference_path.clone(),
    };

    // One id per flushed phrase. All streaming subtitle events
    // emitted by this call share it so the overlay updates a
    // single line in place instead of stacking N copies of the
    // text growing fragment-by-fragment.
    let phrase_id = next_phrase_id();

    let stream_result = translator.translate_stream(&segment, |fragment| {
        let frag_text = fragment.text.trim();
        if frag_text.is_empty() && !fragment.is_final {
            return;
        }

        if !fragment.is_final {
            if !full_translation.is_empty() && !full_translation.ends_with(' ') {
                full_translation.push(' ');
            }
            full_translation.push_str(&fragment.text);
            if !first_fragment_seen {
                let _ = metrics_tx.try_send(PipelineMetrics::new(
                    "translate_first_fragment".to_string(),
                    translate_start.elapsed(),
                ));
                first_fragment_seen = true;
            }
        }

        // Live subtitle update — UI reflects accumulated translation
        // even before the full sentence finishes. Same phrase_id
        // every fragment → overlay replaces the line in place.
        if let Some(tx) = subtitle_tx {
            let _ = tx.send(SubtitleEvent {
                phrase_id,
                pipeline_name: pipeline_name.to_string(),
                source_text: trimmed.to_string(),
                translated_text: full_translation.trim().to_string(),
                language: target_language,
                timestamp: Instant::now(),
            });
        }

        // Synthesise + dispatch THIS fragment. The TTS bridge can
        // emit several PCM chunks per fragment (XTTS streaming) — we
        // forward each one to audio_output as it arrives so playback
        // starts on the first chunk instead of waiting for the whole
        // fragment to finish synthesising. Time-to-first-audio per
        // fragment drops from ~`fragment_inference_ms` (~700-2000 ms
        // on RTX 3050) to ~`first_chunk_ms` (~250-500 ms).
        //
        // Atomic bridges (Kokoro) emit exactly one chunk; the loop
        // executes once and the per-chunk path is equivalent to the
        // previous atomic dispatch.
        if !fragment.is_final && !frag_text.is_empty() {
            let frag_segment =
                TextSegment::new(frag_text.to_string(), target_language);
            let tts_start = Instant::now();
            let mut tts_first_chunk_logged = false;
            let stream_result = tts.synthesize_stream(
                &frag_segment,
                voice_profile.clone(),
                |chunk| {
                    if !tts_first_chunk_logged {
                        let _ = metrics_tx.try_send(PipelineMetrics::new(
                            "tts_first_chunk".to_string(),
                            tts_start.elapsed(),
                        ));
                        tts_first_chunk_logged = true;
                    }
                    // TCC needs a whole utterance to compute its
                    // tone-color transfer; running it per mid-stream
                    // chunk gives garbage at the edges. We only apply
                    // it to chunks that carry the boundary marker
                    // (atomic chunks — the LAST one of a streamed
                    // fragment OR a single Kokoro chunk). Mid-stream
                    // chunks (`is_streaming_chunk == true`) bypass
                    // TCC and play raw — the bridge already cloned
                    // the speaker's voice when XTTS is the engine,
                    // so TCC is redundant there anyway (ADR 0014).
                    let final_audio = if chunk.is_streaming_chunk {
                        chunk
                    } else {
                        apply_tcc_if_eligible(
                            pipeline_name,
                            chunk,
                            speaker_id,
                            reference_path.as_deref(),
                            voice_convert,
                            metrics_tx,
                        )
                    };
                    let _ = audio_output.send(final_audio);
                },
            );
            let _ = metrics_tx.try_send(PipelineMetrics::new(
                "tts".to_string(),
                tts_start.elapsed(),
            ));
            if let Err(e) = stream_result {
                tracing::warn!(
                    "[{}] V2 TTS fragment failed: {}",
                    pipeline_name, e,
                );
            }
        }
    });

    if let Err(e) = stream_result {
        tracing::warn!(
            "[{}] V2 streaming translation failed: {}",
            pipeline_name, e,
        );
        return;
    }

    let _ = metrics_tx.try_send(PipelineMetrics::new(
        "translate".to_string(),
        translate_start.elapsed(),
    ));

    let final_translation = full_translation.trim();
    if final_translation.is_empty() {
        return;
    }

    // Degenerate check runs AFTER the stream because we want the full
    // translated text. By this point the audio already played, so
    // dropping is too late — the check now only suppresses the echo
    // record so the bad text doesn't pollute the buffer.
    if is_translation_degenerate(trimmed, final_translation) {
        tracing::info!(
            "[{}] V2 degenerate translation (post-stream): \"{}\" → \"{}\"",
            pipeline_name,
            preview(trimmed, 40),
            preview(final_translation, 40),
        );
        return;
    }

    record_translation(echo_buffer, final_translation);

    tracing::info!(
        "[{}] V2 → \"{}\"",
        pipeline_name,
        preview(final_translation, 80),
    );
}

fn apply_tcc_if_eligible(
    pipeline_name: &str,
    tts_audio: AudioChunk,
    speaker_id: Option<u32>,
    reference_path: Option<&str>,
    voice_convert: Option<&Arc<ToneColorConverter>>,
    metrics_tx: &mpsc::Sender<PipelineMetrics>,
) -> AudioChunk {
    let tts_duration_ms =
        (tts_audio.samples.len() as u64 * 1000) / tts_audio.sample_rate.max(1) as u64;
    if tts_duration_ms < TCC_MIN_DURATION_MS {
        return tts_audio;
    }
    let (vc, reference) = match (voice_convert, reference_path) {
        (Some(vc), Some(reference)) => (vc, reference),
        _ => return tts_audio,
    };
    let vc_start = Instant::now();
    let speaker_for_tcc = speaker_id.unwrap_or(0);
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

    #[test]
    fn split_extracts_single_complete_sentence() {
        let (complete, rest) = split_complete_sentences("Hello world.");
        assert_eq!(complete, "Hello world.");
        assert_eq!(rest, "");
    }

    #[test]
    fn split_keeps_fragment_when_no_punctuation() {
        let (complete, rest) = split_complete_sentences("Hello world");
        assert_eq!(complete, "");
        assert_eq!(rest, "Hello world");
    }

    #[test]
    fn split_separates_complete_from_in_progress() {
        // The bug from 2026-05-07 logs: "Sentence one. Sentence two. fragment"
        // used to be held entirely until "fragment" got punctuation.
        let (complete, rest) =
            split_complete_sentences("First done. Second done. open clause");
        assert_eq!(complete, "First done. Second done.");
        assert_eq!(rest, "open clause");
    }

    #[test]
    fn split_handles_multiple_complete_sentences_no_remainder() {
        let (complete, rest) = split_complete_sentences("Yes. No. Maybe!");
        assert_eq!(complete, "Yes. No. Maybe!");
        assert_eq!(rest, "");
    }

    #[test]
    fn split_does_not_break_on_acronyms() {
        // "U.S.A. is" — the periods inside U.S.A are not sentence
        // boundaries (next byte is letter, not whitespace). Only the
        // final period before " is" qualifies.
        let (complete, rest) = split_complete_sentences("U.S.A. is great");
        assert_eq!(complete, "U.S.A.");
        assert_eq!(rest, "is great");
    }

    #[test]
    fn split_handles_question_and_exclamation() {
        let (complete, rest) =
            split_complete_sentences("Are you sure? Yes! we are. starting now");
        assert_eq!(complete, "Are you sure? Yes! we are.");
        assert_eq!(rest, "starting now");
    }

    // ─── should_force_flush: latency-first accumulator (2026-05-08) ─────────

    #[test]
    fn force_flush_holds_short_quiet_buffer() {
        // 3 words, 200 ms held — well below every trigger. Stay
        // pending so the speaker can finish the clause.
        assert!(!should_force_flush(3, 200));
    }

    #[test]
    fn force_flush_releases_on_soft_trigger() {
        // 6 words + 800 ms = soft-flush threshold met. This is the
        // common case the rule was added for: Whisper produced a
        // full clause without punctuation; we no longer wait the
        // full MAX_HOLD ceiling for it.
        assert!(should_force_flush(6, 800));
    }

    #[test]
    fn force_flush_holds_when_age_below_soft_threshold() {
        // 8 words but only 400 ms — age gate of soft-flush not yet
        // satisfied. Still no aged_out or long_enough trigger
        // either, so we hold.
        assert!(!should_force_flush(8, 400));
    }

    #[test]
    fn force_flush_holds_when_word_count_below_soft_threshold() {
        // 5 words, 1000 ms — under SOFT_WORDS (6) so soft-flush
        // doesn't fire; under MAX_HOLD (1200) so aged_out doesn't
        // fire either. Wait for either more words or more time.
        assert!(!should_force_flush(5, 1000));
    }

    #[test]
    fn force_flush_releases_on_age_ceiling() {
        // 4 words but held > MAX_HOLD. The hard ceiling fires even
        // when the buffer is small — short utterances shouldn't be
        // held forever just because they're brief.
        assert!(should_force_flush(4, 1300));
    }

    #[test]
    fn force_flush_releases_on_word_ceiling() {
        // 35 words, 0 ms — comma-less monologue, blow the lid.
        assert!(should_force_flush(ACCUMULATOR_MAX_WORDS, 0));
    }

    #[test]
    fn force_flush_releases_when_both_thresholds_met() {
        // 40 words AND 5000 ms — every trigger fires. Just verifies
        // the disjunction doesn't suppress one when another applies.
        assert!(should_force_flush(40, 5000));
    }

    #[test]
    fn split_handles_portuguese_accents() {
        // The text after the period has multi-byte chars — the byte-
        // based scan must not panic and must still find the boundary
        // because the byte right after the period IS an ASCII space.
        let (complete, rest) =
            split_complete_sentences("Você está aqui. À tarde começa");
        assert_eq!(complete, "Você está aqui.");
        assert_eq!(rest, "À tarde começa");
    }
}
