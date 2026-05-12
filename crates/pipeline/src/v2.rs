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
use sbd::SbdService;
use shared::{AudioChunk, Language, PipelineCommand, PipelineMetrics};
use stt::{StreamingSession, WhisperStt};
use tokio::sync::mpsc;
use tracing;
use translation::OpusMtTranslator;
use tts::{PiperTts, VoiceProfile};
use voice_convert::ToneColorConverter;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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
/// "Latency-first accumulator").
///
/// Raised from 1200 → 2000 ms on 2026-05-11 (Phase 0 follow-up). The
/// 1200 ms ceiling was cutting documentary narration mid-clause:
/// `"and two and a half times the height of the"` (1843 ms old)
/// flushed alone, and `"Statue of Liberty"` arrived as an orphan
/// phrase that the translator couldn't connect back. 2000 ms covers
/// the typical "noun phrase + subordinate clause" arc in English
/// without giving back the soft-flush gains on short conversational
/// speech (those still flush at the 800 ms soft trigger).
const ACCUMULATOR_MAX_HOLD_MS: u128 = 2000;

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
/// input.
///
/// Raised from 6 → 10 on 2026-05-11 (Phase 0 / XTTS backlog fix). At
/// 6 the accumulator would flush mid-clause on documentary narration
/// — e.g. "and two and a half times the height of the Statue of
/// Liberty" got cut at "...the height" and "Statue of Liberty"
/// arrived as a separate phrase, defeating context. 10 covers the
/// typical English subordinate clause without losing the soft-flush
/// benefit on conversational speech (still well under the 1200 ms
/// hard MAX_HOLD that catches monologues).
const ACCUMULATOR_SOFT_FLUSH_WORDS: usize = 10;
const ACCUMULATOR_SOFT_FLUSH_HOLD_MS: u128 = 800;

/// Maximum number of phrases allowed in-flight at the TTS engine. When
/// `tts.synthesize_stream` already has this many concurrent calls
/// (counting both ongoing inference and audio still queued for the
/// mixer to consume), the next phrase enters the wait loop instead
/// of joining an unbounded XTTS queue.
const MAX_TTS_INFLIGHT: usize = 2;

/// Maximum time a phrase will busy-wait for a TTS slot before being
/// dropped. The previous design dropped on the first conflict, which
/// killed short phrases that would have synthesised in 1-2 s simply
/// because a long phrase ahead held a slot. The wait turns "drop"
/// into "queue with a fairness ceiling" — content survives moderate
/// XTTS spikes; only truly sustained overload (> ~5 s of solid
/// backlog) loses anything.
const TTS_QUEUE_TIMEOUT_MS: u64 = 5000;

/// Poll cadence while a phrase is waiting for a TTS slot. 200 ms is a
/// trade-off: shorter wakes up faster when a slot opens (lower added
/// latency) but spends more thread time spinning. Two checks per
/// second is plenty given XTTS phrase durations are 1-15 s.
const TTS_QUEUE_POLL_MS: u64 = 200;

/// Duplicate-suppression window. When the same final translation has
/// just been sent to the TTS within this window, skip the second
/// invocation. Catches the common case where streaming-partial and
/// closed-segment paths both commit the same phrase a few hundred ms
/// apart ("Huge Conversations." synthesised twice in 2026-05-11
/// capture). Set wider than typical phrase playback time so that the
/// dedup also covers the case where the duplicate would interrupt
/// the still-playing original.
const PHRASE_DEDUP_WINDOW_MS: u128 = 4000;

/// How many recent phrases we remember for dedup. Documentary-style
/// narration alternates 3-5 distinct phrases in a 4 s window — a ring
/// of 8 covers every reasonable case without growing unbounded.
const PHRASE_DEDUP_RING_SIZE: usize = 8;

/// How many recent diariser identifications we keep to stabilise the
/// effective speaker_id by majority vote. ECAPA-TDNN occasionally
/// returns a noisy id on a single chunk (similar voices, short clip);
/// without smoothing, the Kokoro voice router and the per-speaker
/// reference path flip every time that happens — the listener hears
/// the same person "change voice" mid-paragraph. With K=5, a single
/// bad call gets out-voted by four neighbours and the assigned voice
/// stays stable; a genuine speaker change still wins after 3 of 5
/// agreements (~9-15 s of speech in narration), which is the right
/// trade-off between responsiveness and stability.
const SPEAKER_HISTORY_K: usize = 5;

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
/// Pure majority-vote over the last `SPEAKER_HISTORY_K` diariser
/// identifications. Pushes `new_id` into `history`, trims the ring,
/// and returns the most frequent id. Ties are broken by recency
/// (the most recently observed id among the tied wins) — that's
/// what keeps the speaker locked instead of oscillating when a new
/// id appears and the ring is split 50/50 between the old and the
/// new.
fn stable_speaker_id(new_id: u32, history: &mut VecDeque<u32>) -> u32 {
    history.push_back(new_id);
    while history.len() > SPEAKER_HISTORY_K {
        history.pop_front();
    }
    let mut counts: std::collections::HashMap<u32, usize> =
        std::collections::HashMap::new();
    for &id in history.iter() {
        *counts.entry(id).or_insert(0) += 1;
    }
    let max_count = counts.values().copied().max().unwrap_or(0);
    // Walk the ring back-to-front so the most recent id wins on ties.
    for &id in history.iter().rev() {
        if counts.get(&id).copied().unwrap_or(0) == max_count {
            return id;
        }
    }
    new_id
}

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
    /// Cached reference WAV path for TCC / XTTS in the CURRENT phrase.
    /// Cleared on flush along with the rest of the per-phrase state.
    pending_reference_path: Option<String>,
    /// Sticky "last identified speaker" reference, kept across flushes
    /// so that streaming-partial flushes — which never carry a
    /// `speaker_id` (diariser only runs on closed segments) — can
    /// still pick the right voice instead of falling back to the
    /// generic `user.wav` fallback. Updated whenever a closed-segment
    /// flush brings in a non-None resolved reference; never cleared
    /// unless the pipeline is restarted (`Stop` → `Accumulator::default`).
    /// Captured on 2026-05-11: in continuous narration most flushes
    /// fire from streaming partials before the next closed segment
    /// lands, and without this sticky field every other phrase played
    /// in the default voice even though the diariser had already
    /// identified the speaker on the previous closed segment.
    last_known_reference_path: Option<String>,
    /// Ring of recently-synthesised translations with their dispatch
    /// timestamp. Used to suppress duplicates: the streaming-partial
    /// path and the closed-segment path can both commit the same
    /// phrase a few hundred ms apart (e.g. "Huge Conversations."
    /// shipped twice in 2026-05-11 capture). Bounded to
    /// `PHRASE_DEDUP_RING_SIZE` entries; older entries fall off the
    /// front when full. Lookups use `PHRASE_DEDUP_WINDOW_MS` to ignore
    /// entries that are too old to overlap with new audio anyway.
    recent_synths: VecDeque<(String, Instant)>,
    /// Ring of recent raw speaker ids returned by the diariser, in
    /// closed-segment order. The pipeline routes by the MAJORITY of
    /// this ring (`stable_speaker_id`) rather than by each diariser
    /// call directly — a noisy ECAPA-TDNN identification on a single
    /// chunk no longer flips the Kokoro voice / reference path. Ring
    /// is bounded to `SPEAKER_HISTORY_K`.
    recent_speaker_ids: VecDeque<u32>,
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

/// Decide where to cut the accumulator buffer between "ready to flush"
/// and "still pending". When the SBD service is available, it consults
/// spaCy's dependency parser (via `scripts/sbd_bridge.py`) for a
/// semantic-aware decision: a span only counts as complete if it has
/// terminal punctuation AND a finite verb with a subject (or an
/// imperative). When SBD fails or isn't configured, falls back to the
/// regex `split_complete_sentences` rule so the pipeline keeps running.
///
/// Why both layers: SBD is a Python subprocess and can hang, time out,
/// or be intentionally disabled. The regex fallback is local, in-memory,
/// and never blocks — it's a worse decision but a guaranteed one.
fn decide_complete_split(
    text: &str,
    language: Language,
    sbd: Option<&SbdService>,
) -> (String, String) {
    if let Some(service) = sbd {
        match service.split(text, language) {
            Ok(result) => return (result.complete, result.rest),
            Err(e) => {
                tracing::warn!(
                    "SBD split failed ({}), falling back to regex split_complete_sentences",
                    e
                );
            }
        }
    }
    split_complete_sentences(text)
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
    /// Semantic sentence boundary detector. `Some` when the SBD bridge
    /// initialised cleanly at startup; `None` when SBD was disabled
    /// or its Python subprocess refused to start. The accumulator's
    /// flush decision falls back to regex `split_complete_sentences`
    /// in the None case — same external behaviour as pre-SBD builds.
    pub sbd: Option<Arc<SbdService>>,
    pub fixed_voice_reference: Option<String>,
    pub config: V2Config,
    /// Optional channel for subtitle text events. When present, every
    /// translated phrase is forwarded to the overlay UI before TTS.
    /// Uses `std::sync::mpsc` because the send happens inside
    /// `spawn_blocking` (no async context).
    pub subtitle_tx: Option<std_mpsc::Sender<SubtitleEvent>>,
    /// ADR 0015 — when true, partial Whisper passes run inside the
    /// open phrase window and commit stable LA-2 prefixes into the
    /// accumulator before the window closes. Default false to match
    /// pre-streaming behaviour.
    pub streaming_stt_enabled: bool,
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
            sbd: None,
            fixed_voice_reference: None,
            config: V2Config::default(),
            subtitle_tx: None,
            streaming_stt_enabled: false,
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

    pub fn with_sbd(mut self, sbd: Arc<SbdService>) -> Self {
        self.sbd = Some(sbd);
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

    /// Enable ADR 0015 streaming STT. When set, partial Whisper passes
    /// run periodically while the phrase window is open and commit
    /// stable LA-2 prefixes into the accumulator before the window
    /// closes. Trades ~3-4× more Whisper inference per phrase for a
    /// big TTFA win.
    pub fn with_streaming_stt(mut self, enabled: bool) -> Self {
        self.streaming_stt_enabled = enabled;
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
        // Phase 0 backlog cap (2026-05-11): every flush increments this
        // before invoking TTS and decrements after. When the counter is
        // already at MAX_TTS_INFLIGHT the new flush is dropped instead
        // of joining an unbounded XTTS queue.
        let tts_inflight = Arc::new(AtomicUsize::new(0));
        // ADR 0015 — optional streaming session. When present, partial
        // Whisper passes run periodically while the phrase window is
        // open and commit stable LA-2 prefixes into the accumulator
        // before the window closes.
        let streaming_session: Option<Arc<Mutex<StreamingSession>>> =
            if self.streaming_stt_enabled {
                Some(Arc::new(Mutex::new(StreamingSession::new(
                    Arc::clone(&self.stt),
                    self.source_language,
                ))))
            } else {
                None
            };
        let mut is_running = false;

        tracing::info!(
            "[{}] V2 ready (max_window={:?}, silence_tail={:?}, min_window={:?}, diariser={}, vc={}, streaming_stt={})",
            pipeline_name,
            self.config.max_window,
            self.config.silence_tail,
            self.config.min_window,
            self.diarizer.is_some(),
            self.voice_convert.is_some(),
            streaming_session.is_some(),
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
                            if let Some(ss) = streaming_session.as_ref() {
                                ss.lock().expect("streaming session lock").reset();
                            }
                        }
                        Some(PipelineCommand::Stop) => {
                            tracing::info!("[{}] V2 pipeline stopped", pipeline_name);
                            is_running = false;
                            let _ = segmenter.flush();
                            if let Ok(mut acc) = accumulator.lock() {
                                *acc = Accumulator::default();
                            }
                            if let Some(ss) = streaming_session.as_ref() {
                                ss.lock().expect("streaming session lock").reset();
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
                        let audio_output_clone = audio_output.clone();
                        let metrics_tx_clone = metrics_tx.clone();
                        let pipeline_name_clone = pipeline_name.clone();
                        let source_language = self.source_language;
                        let fixed_voice_reference = self.fixed_voice_reference.clone();
                        let subtitle_tx = self.subtitle_tx.clone();
                        let segment_samples = segment.samples;
                        let segment_sample_rate = segment.sample_rate;
                        let accumulator_clone = Arc::clone(&accumulator);
                        let streaming_for_segment = streaming_session.clone();
                        let tts_inflight_clone = Arc::clone(&tts_inflight);
                        let sbd_clone = self.sbd.clone();

                        tokio::task::spawn_blocking(move || {
                            process_segment(
                                pipeline_name_clone,
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
                                audio_output_clone,
                                metrics_tx_clone,
                                subtitle_tx,
                                accumulator_clone,
                                streaming_for_segment,
                                tts_inflight_clone,
                                sbd_clone,
                            );
                        });
                    } else if let Some(ss) = streaming_session.as_ref() {
                        // ADR 0015 — phrase still open. Schedule a
                        // partial pass on the in-progress buffer if
                        // there's enough audio. The session's internal
                        // throttle (`PARTIAL_INTERVAL_MS`) and minimum
                        // length gate (`MIN_PARTIAL_SECONDS`) make this
                        // a no-op when called too frequently, so we can
                        // dispatch on every chunk safely. The Mutex
                        // around the session also serialises with the
                        // close-time `finalize` call, so a chunk
                        // arriving exactly as the window closes can't
                        // race against the finalize.
                        if !segmenter.is_phrase_open() {
                            continue;
                        }
                        let open_buffer = segmenter.peek_open_buffer().to_vec();
                        if open_buffer.is_empty() {
                            continue;
                        }
                        let session = Arc::clone(ss);
                        let translator = Arc::clone(&self.translator);
                        let tts_engine = Arc::clone(&self.tts);
                        let voice_convert = self.voice_convert.clone();
                        let echo_buffer = Arc::clone(&self.echo_buffer);
                        let audio_output_clone = audio_output.clone();
                        let metrics_tx_clone = metrics_tx.clone();
                        let pipeline_name_clone = pipeline_name.clone();
                        let source_language = self.source_language;
                        let fixed_voice_reference = self.fixed_voice_reference.clone();
                        let subtitle_tx = self.subtitle_tx.clone();
                        let accumulator_clone = Arc::clone(&accumulator);
                        let tts_inflight_clone = Arc::clone(&tts_inflight);
                        let sbd_clone = self.sbd.clone();

                        tokio::task::spawn_blocking(move || {
                            process_streaming_partial(
                                pipeline_name_clone,
                                open_buffer,
                                session,
                                translator,
                                tts_engine,
                                voice_convert,
                                echo_buffer,
                                source_language,
                                fixed_voice_reference,
                                audio_output_clone,
                                metrics_tx_clone,
                                subtitle_tx,
                                accumulator_clone,
                                tts_inflight_clone,
                                sbd_clone,
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
    // ADR 0015 — when present, partial Whisper passes already
    // committed some words during the open phrase. We call
    // `finalize(final_text)` on close to compute the suffix that
    // hadn't been emitted yet, and only push that suffix into the
    // accumulator. None means streaming was disabled for this
    // pipeline; the full transcribed text is pushed.
    streaming_session: Option<Arc<Mutex<StreamingSession>>>,
    tts_inflight: Arc<AtomicUsize>,
    sbd: Option<Arc<SbdService>>,
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

    // Stabilise the diariser's per-chunk identification by majority
    // vote over the last few closed segments. F0 history and audio
    // enrolment still use the RAW id (we want the running F0 and the
    // 6-second WAV reference to belong to the actual voice the
    // diariser thought it heard, not to a smoothed alias), but the
    // id we propagate to the rest of the pipeline (Kokoro voice
    // routing, reference path lookup) is the stable one.
    let raw_speaker_id = identification.as_ref().map(|s| s.speaker_id);
    if let Some(id) = identification.as_ref() {
        voice_profiles.record_f0(id.speaker_id, id.f0_hz);
        // Auto-enrolment: feed the segment audio into this speaker's
        // reference buffer. Once it has 6 s of clean speech a WAV
        // reference is written and TCC can convert future TTS for the
        // same speaker.
        voice_profiles.ingest_audio(id.speaker_id, &samples);
    }
    let speaker_id = raw_speaker_id.map(|raw| {
        let mut acc = accumulator.lock().expect("accumulator poisoned");
        let stable = stable_speaker_id(raw, &mut acc.recent_speaker_ids);
        if stable != raw {
            tracing::debug!(
                "[{}] V2 speaker id smoothed: diariser said {} → stable {}",
                pipeline_name,
                raw,
                stable,
            );
        }
        stable
    });

    let chunk = AudioChunk::new(samples, sample_rate, 1);

    // ─── STT (final pass on the closed segment) ───────────────────────
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

    let full_text = transcribed.text.trim().to_string();
    if full_text.is_empty() {
        // Even on empty STT, give the streaming session a chance to
        // reset its per-phrase state — otherwise the next phrase's
        // first partial would compute LA against this phrase's
        // history.
        if let Some(ss) = streaming_session.as_ref() {
            ss.lock().expect("streaming session lock").reset();
        }
        return;
    }

    // ─── Streaming reconciliation ─────────────────────────────────────
    // When streaming was active, words were already pushed to the
    // accumulator during the open window. The closed-segment
    // transcribe is the authoritative read; finalize() gives us the
    // suffix that wasn't yet committed (and a flag if the streaming
    // commit diverged from what the final pass produces). We keep the
    // streaming commit regardless — audio already played — and feed
    // only the suffix downstream.
    let source_text: String = match streaming_session.as_ref() {
        Some(ss) => {
            let finalised = ss.lock().expect("streaming session lock").finalize(&full_text);
            if finalised.committed_diverged {
                tracing::info!(
                    "[{}] V2 streaming finalize: committed prefix diverged from final transcribe — keeping committed audio (suffix={} words)",
                    pipeline_name,
                    finalised.uncommitted_suffix.len(),
                );
            }
            if finalised.uncommitted_suffix.is_empty() {
                // Nothing left to push — streaming committed
                // everything. Skip the rest, but still emit total.
                tracing::debug!(
                    "[{}] V2 streaming: all words already committed, no suffix to push",
                    pipeline_name,
                );
                let _ = metrics_tx.try_send(PipelineMetrics::new(
                    "total".to_string(),
                    total_start.elapsed(),
                ));
                return;
            }
            finalised.uncommitted_suffix.join(" ")
        }
        None => full_text,
    };
    let source_text = source_text.trim();
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

    let resolved_reference = fixed_voice_reference
        .clone()
        .or_else(|| speaker_id.and_then(|id| voice_profiles.reference_for(id)));
    let f0_for_tts = match speaker_id {
        Some(id) => voice_profiles.f0_for(id),
        None => 0.0,
    };

    ingest_text_and_maybe_flush(
        &pipeline_name,
        source_text,
        speaker_id,
        f0_for_tts,
        resolved_reference,
        source_language,
        &translator,
        &tts,
        voice_convert.as_ref(),
        &echo_buffer,
        &audio_output,
        &metrics_tx,
        subtitle_tx.as_ref(),
        &accumulator,
        &tts_inflight,
        sbd.as_deref(),
    );

    let _ = metrics_tx.try_send(PipelineMetrics::new(
        "total".to_string(),
        total_start.elapsed(),
    ));
}

/// ADR 0015 — process a single streaming partial pass while a phrase
/// window is still open. Runs `StreamingSession::run_partial` on the
/// open buffer; if LA-2 newly-committed any words, joins them and
/// pushes through `ingest_text_and_maybe_flush`. The accumulator's
/// existing flush rules (punctuation, soft-flush, force-flush) decide
/// whether to actually translate + TTS now or keep buffering.
///
/// Skips the diariser and echo check that the closed-segment path
/// runs — partials don't have a stable speaker_id yet (diariser runs
/// per closed window in V2), and echo detection needs the full
/// closed-segment audio. The voice profile falls back to whatever
/// `fixed_voice_reference` provides (mic side has the user's profile;
/// loopback side may be `None` and TTS uses the engine default).
#[allow(clippy::too_many_arguments)]
fn process_streaming_partial(
    pipeline_name: String,
    open_buffer: Vec<f32>,
    streaming_session: Arc<Mutex<StreamingSession>>,
    translator: Arc<OpusMtTranslator>,
    tts: Arc<PiperTts>,
    voice_convert: Option<Arc<ToneColorConverter>>,
    echo_buffer: EchoBuffer,
    source_language: Language,
    fixed_voice_reference: Option<String>,
    audio_output: mpsc::UnboundedSender<AudioChunk>,
    metrics_tx: mpsc::Sender<PipelineMetrics>,
    subtitle_tx: Option<std_mpsc::Sender<SubtitleEvent>>,
    accumulator: Arc<Mutex<Accumulator>>,
    tts_inflight: Arc<AtomicUsize>,
    sbd: Option<Arc<SbdService>>,
) {
    let partial_start = Instant::now();
    let new_words = {
        let mut session = streaming_session.lock().expect("streaming session lock");
        session.run_partial(&open_buffer)
    };
    let _ = metrics_tx.try_send(PipelineMetrics::new(
        "streaming_partial".to_string(),
        partial_start.elapsed(),
    ));
    if new_words.is_empty() {
        return;
    }

    let source_text = new_words.join(" ");
    let source_text = source_text.trim();
    if source_text.is_empty() {
        return;
    }

    tracing::info!(
        "[{}] V2 streaming committed: \"{}\"",
        pipeline_name,
        preview(source_text, 80),
    );

    ingest_text_and_maybe_flush(
        &pipeline_name,
        source_text,
        None,
        0.0,
        fixed_voice_reference,
        source_language,
        &translator,
        &tts,
        voice_convert.as_ref(),
        &echo_buffer,
        &audio_output,
        &metrics_tx,
        subtitle_tx.as_ref(),
        &accumulator,
        &tts_inflight,
        sbd.as_deref(),
    );
}

/// Push `source_text` into the per-pipeline accumulator and decide
/// whether any portion of it should flush now (translate + TTS) or
/// keep waiting. Shared by the closed-segment path
/// (`process_segment`) and the streaming-partial path
/// (`process_streaming_partial`). The accumulator's flush rules
/// (punctuation, soft-flush, MAX_HOLD, MAX_WORDS, speaker change)
/// are invariant across both callers.
///
/// Up to two flushes can fire from a single ingest:
///   1. **Early flush** when the speaker changed and the previous
///      speaker had pending text — that text releases first, with
///      the previous speaker's voice profile.
///   2. **Main flush** when the (possibly extended) accumulator now
///      ends in punctuation, has aged past hard caps, or hit the
///      soft-flush threshold.
#[allow(clippy::too_many_arguments)]
fn ingest_text_and_maybe_flush(
    pipeline_name: &str,
    source_text: &str,
    speaker_id: Option<u32>,
    f0_for_tts: f32,
    resolved_reference: Option<String>,
    source_language: Language,
    translator: &OpusMtTranslator,
    tts: &PiperTts,
    voice_convert: Option<&Arc<ToneColorConverter>>,
    echo_buffer: &EchoBuffer,
    audio_output: &mpsc::UnboundedSender<AudioChunk>,
    metrics_tx: &mpsc::Sender<PipelineMetrics>,
    subtitle_tx: Option<&std_mpsc::Sender<SubtitleEvent>>,
    accumulator: &Arc<Mutex<Accumulator>>,
    tts_inflight: &Arc<AtomicUsize>,
    sbd: Option<&SbdService>,
) {
    let now = Instant::now();
    let (early_flush, main_flush) = {
        let mut acc = accumulator.lock().expect("accumulator poisoned");
        let speaker_changed = matches!(
            (acc.pending_speaker_id, speaker_id),
            (Some(old), Some(new)) if old != new,
        );

        let early = if speaker_changed && !acc.pending_text.is_empty() {
            let drained = std::mem::take(&mut acc.pending_text);
            let snapshot_reference = acc
                .pending_reference_path
                .clone()
                .or_else(|| acc.last_known_reference_path.clone());
            let snapshot = (
                drained,
                acc.pending_speaker_id,
                acc.pending_f0_hz,
                snapshot_reference,
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
            acc.last_known_reference_path = resolved_reference.clone();
        }

        let pending_age_ms = acc
            .pending_started_at
            .map(|t| now.duration_since(t).as_millis())
            .unwrap_or(0);
        let pending_word_count = acc.pending_text.split_whitespace().count();
        let force_flush = should_force_flush(pending_word_count, pending_age_ms);

        // Ask the SBD service (or fall back to the regex rule) whether
        // the current accumulator content has reached a semantic
        // boundary. Holds the lock briefly across the subprocess
        // round-trip — typically <50 ms with spaCy's small models;
        // every other producer (streaming partials, closed segments)
        // serialises on this same mutex anyway, so we don't lose
        // parallelism by waiting here.
        let (complete, remaining) =
            decide_complete_split(&acc.pending_text, source_language, sbd);

        // The snapshot reference picks the per-phrase reference if set,
        // otherwise falls back to the last-known reference from prior
        // closed segments. This is what makes streaming-partial
        // flushes — which never carry a speaker_id — speak in the
        // right voice instead of the generic fallback. Inlined twice
        // (instead of a closure) so the immutable borrow of `acc`
        // ends before the per-branch mutable updates below.
        let main = if !complete.is_empty() {
            let snapshot_ref = acc
                .pending_reference_path
                .clone()
                .or_else(|| acc.last_known_reference_path.clone());
            let snapshot = (
                complete,
                acc.pending_speaker_id,
                acc.pending_f0_hz,
                snapshot_ref,
            );
            acc.pending_text = remaining;
            if acc.pending_text.is_empty() {
                acc.pending_speaker_id = None;
                acc.pending_started_at = None;
                acc.pending_f0_hz = 0.0;
                acc.pending_reference_path = None;
            } else {
                acc.pending_started_at = Some(now);
            }
            Some(snapshot)
        } else if force_flush {
            let snapshot_ref = acc
                .pending_reference_path
                .clone()
                .or_else(|| acc.last_known_reference_path.clone());
            let drained = std::mem::take(&mut acc.pending_text);
            let snapshot = (
                drained,
                acc.pending_speaker_id,
                acc.pending_f0_hz,
                snapshot_ref,
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

    if early_flush.is_none() && main_flush.is_none() {
        tracing::info!(
            "[{}] V2 held: \"{}\"",
            pipeline_name,
            preview(source_text, 80),
        );
        return;
    }

    if let Some((text, sid, f0, reference)) = early_flush {
        flush_phrase(
            pipeline_name,
            &text,
            sid,
            f0,
            reference,
            source_language,
            translator,
            tts,
            voice_convert,
            echo_buffer,
            audio_output,
            metrics_tx,
            subtitle_tx,
            tts_inflight,
            accumulator,
        );
    }
    if let Some((text, sid, f0, reference)) = main_flush {
        flush_phrase(
            pipeline_name,
            &text,
            sid,
            f0,
            reference,
            source_language,
            translator,
            tts,
            voice_convert,
            echo_buffer,
            audio_output,
            metrics_tx,
            subtitle_tx,
            tts_inflight,
            accumulator,
        );
    }
}

/// Streaming-friendly minimum TTS duration before TCC is invoked.
/// OpenVoice preprocessing fails on very short audio (< 500 ms),
/// so we skip TCC for those fragments and play raw Kokoro instead.
const TCC_MIN_DURATION_MS: u64 = 500;

/// Number of XTTS streaming chunks to accumulate before releasing the
/// first chunk to the mixer. XTTS-v2 emits ~250 ms PCM frames at RTF
/// ~1.5-1.9 on RTX 3050 — slower than the player consumes. Without a
/// pre-buffer, the mixer plays the first chunk in ~700 ms but the
/// second chunk lands ~1.5 s later, leaving a silence gap mid-phrase
/// that listeners described as "audio falhando no meio da fala".
/// Three chunks ≈ 750 ms of safety cushion: enough that consecutive
/// chunks arrive while the player is still draining the buffered
/// audio, eliminating the underrun without converting back to atomic
/// (which would push TTFA from ~1 s to the full phrase synthesis time
/// of 3-6 s).
const TTS_PRE_BUFFER_CHUNKS: usize = 3;

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
    tts_inflight: &Arc<AtomicUsize>,
    accumulator: &Arc<Mutex<Accumulator>>,
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

    // ─── Phase 0 (2026-05-11): TTS receives the whole translation ─────
    // Previous behaviour: each clause-bound fragment from Qwen went
    // straight into XTTS as a separate inference call. With XTTS-v2 at
    // RTF ~1-2 on RTX 3050, N fragments per phrase → N sequential
    // inference calls → backlog that grew unbounded (51 s end-to-end
    // latency captured 2026-05-11). Now we keep the streaming
    // translation (Qwen still token-streams, subtitle still updates
    // live), but TTS is invoked ONCE per phrase with the assembled
    // translation. Cost: ~1 s of TTFA we gave up by waiting for the
    // last token. Benefit: XTTS sees a complete sentence with proper
    // prosodic arc, fragments < 15 chars (the regime where XTTS
    // mis-vocalises punctuation / emits silence) disappear, and the
    // queue depth caps at 1 inflight per phrase instead of N.
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
        // every fragment → overlay replaces the line in place. This
        // is the only consumer of the streaming fragments now; TTS
        // waits for the complete phrase below.
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

    // Degenerate check runs BEFORE TTS now that the audio hasn't been
    // dispatched yet. Dropping degenerate translations early saves the
    // XTTS inference and prevents bogus echo records.
    if is_translation_degenerate(trimmed, final_translation) {
        tracing::info!(
            "[{}] V2 degenerate translation: \"{}\" → \"{}\"",
            pipeline_name,
            preview(trimmed, 40),
            preview(final_translation, 40),
        );
        return;
    }

    // Phase 0 dedup (2026-05-11): the streaming-partial path and the
    // closed-segment path can commit the same final phrase a few
    // hundred ms apart. Without dedup the listener heard "Huge
    // Conversations." twice back-to-back. Compare against the recent
    // ring before reserving a TTS slot.
    {
        let mut acc = accumulator.lock().expect("accumulator poisoned");
        let now = Instant::now();
        let window = Duration::from_millis(PHRASE_DEDUP_WINDOW_MS as u64);
        let is_duplicate = acc
            .recent_synths
            .iter()
            .any(|(text, ts)| text == final_translation && now.duration_since(*ts) <= window);
        if is_duplicate {
            tracing::info!(
                "[{}] V2 duplicate phrase suppressed: \"{}\"",
                pipeline_name,
                preview(final_translation, 60),
            );
            return;
        }
        // Drop entries past the dedup window while we hold the lock —
        // keeps the ring's working set small.
        while let Some((_, ts)) = acc.recent_synths.front() {
            if now.duration_since(*ts) > window {
                acc.recent_synths.pop_front();
            } else {
                break;
            }
        }
    }

    // Phase 0 backlog cap with queue timeout (2026-05-11): try to grab
    // a TTS slot, falling back to a bounded busy-wait loop so short
    // phrases don't get dropped just because a long phrase is still
    // synthesising. `fetch_add` returns the count BEFORE this slot was
    // added, so a returned value ≥ MAX_TTS_INFLIGHT means we crossed
    // the cap. Using fetch_add (not load + add) closes the race where
    // two threads both observe `n < MAX` and both add, exceeding the
    // cap. After the wait deadline the phrase is dropped — bounded
    // wait converts "drop on first conflict" into "queue with a
    // fairness ceiling" without unbounded backlog growth.
    let wait_deadline = Instant::now() + Duration::from_millis(TTS_QUEUE_TIMEOUT_MS);
    loop {
        let previous_inflight = tts_inflight.fetch_add(1, Ordering::SeqCst);
        if previous_inflight < MAX_TTS_INFLIGHT {
            break;
        }
        tts_inflight.fetch_sub(1, Ordering::SeqCst);
        if Instant::now() >= wait_deadline {
            tracing::warn!(
                "[{}] V2 TTS queue timeout after {}ms, dropping phrase: \"{}\"",
                pipeline_name,
                TTS_QUEUE_TIMEOUT_MS,
                preview(final_translation, 60),
            );
            return;
        }
        std::thread::sleep(Duration::from_millis(TTS_QUEUE_POLL_MS));
    }

    // Atomic TTS: one inference per phrase. XTTS still emits chunks
    // internally (~250 ms PCM frames), but we hold the first
    // `TTS_PRE_BUFFER_CHUNKS` before releasing them as a batch — see
    // the const docs for the underrun fix this addresses.
    let tts_segment = TextSegment::new(final_translation.to_string(), target_language);
    let tts_start = Instant::now();
    let mut tts_first_chunk_logged = false;
    let mut pre_buffer: Vec<AudioChunk> = Vec::with_capacity(TTS_PRE_BUFFER_CHUNKS);
    let mut pre_buffer_released = false;
    let tts_result = tts.synthesize_stream(
        &tts_segment,
        voice_profile.clone(),
        |chunk| {
            if !tts_first_chunk_logged {
                let _ = metrics_tx.try_send(PipelineMetrics::new(
                    "tts_first_chunk".to_string(),
                    tts_start.elapsed(),
                ));
                tts_first_chunk_logged = true;
            }
            // TCC needs a whole utterance to compute its tone-color
            // transfer; running it per mid-stream chunk gives garbage
            // at the edges. We only apply it to chunks that carry the
            // boundary marker (atomic chunks — the LAST one of a
            // streamed fragment OR a single Kokoro chunk). Mid-stream
            // chunks (`is_streaming_chunk == true`) bypass TCC and
            // play raw — the bridge already cloned the speaker's
            // voice when XTTS is the engine, so TCC is redundant
            // there anyway (ADR 0014).
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
            // Hold the opening chunks until the pre-buffer threshold is
            // met; once released, every subsequent chunk goes straight
            // through. `pre_buffer_released` short-circuits the Vec
            // operations for the rest of the phrase.
            if pre_buffer_released {
                let _ = audio_output.send(final_audio);
                return;
            }
            pre_buffer.push(final_audio);
            if pre_buffer.len() >= TTS_PRE_BUFFER_CHUNKS {
                for buffered in pre_buffer.drain(..) {
                    let _ = audio_output.send(buffered);
                }
                pre_buffer_released = true;
            }
        },
    );
    // Drain any chunks left in the pre-buffer (phrase ended with fewer
    // than TTS_PRE_BUFFER_CHUNKS chunks total).
    for buffered in pre_buffer.drain(..) {
        let _ = audio_output.send(buffered);
    }
    let _ = metrics_tx.try_send(PipelineMetrics::new(
        "tts".to_string(),
        tts_start.elapsed(),
    ));
    // Release the TTS slot whether the synthesis succeeded or failed.
    // A panic in tts.synthesize_stream would skip this; if XTTS panics
    // become a real failure mode we can wrap the call in a guard
    // struct that decrements on Drop. Until then we trust the bridge
    // to return Err instead of panicking.
    tts_inflight.fetch_sub(1, Ordering::SeqCst);
    if let Err(e) = tts_result {
        tracing::warn!("[{}] V2 TTS failed: {}", pipeline_name, e);
        return;
    }

    // Record the successful dispatch for the dedup ring. Only on success
    // — a failed synth shouldn't suppress a retry of the same phrase
    // (e.g. if XTTS hit a transient error).
    {
        let mut acc = accumulator.lock().expect("accumulator poisoned");
        if acc.recent_synths.len() >= PHRASE_DEDUP_RING_SIZE {
            acc.recent_synths.pop_front();
        }
        acc.recent_synths.push_back((final_translation.to_string(), Instant::now()));
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
        // 10 words + 800 ms = soft-flush threshold met. The threshold
        // was raised from 6 → 10 on 2026-05-11 because shorter cuts
        // fragmented subordinate clauses ("of the Statue of Liberty"
        // arrived as a separate phrase). 10 words is enough that
        // typical English subordinate clauses ship as one unit.
        assert!(should_force_flush(10, 800));
    }

    #[test]
    fn force_flush_holds_subordinate_clause_below_new_threshold() {
        // 6 words + 1000 ms — this used to trip soft-flush at the
        // 6-word threshold and fragment narration mid-clause. Now
        // below the 10-word floor, we keep accumulating.
        assert!(!should_force_flush(6, 1000));
    }

    #[test]
    fn force_flush_holds_when_age_below_soft_threshold() {
        // 12 words but only 400 ms — age gate of soft-flush not yet
        // satisfied. Still no aged_out or long_enough trigger
        // either, so we hold.
        assert!(!should_force_flush(12, 400));
    }

    #[test]
    fn force_flush_holds_when_word_count_below_soft_threshold() {
        // 5 words, 1000 ms — under SOFT_WORDS (10) so soft-flush
        // doesn't fire; under MAX_HOLD (1200) so aged_out doesn't
        // fire either. Wait for either more words or more time.
        assert!(!should_force_flush(5, 1000));
    }

    #[test]
    fn force_flush_releases_on_age_ceiling() {
        // 4 words but held > MAX_HOLD (2000 ms). The hard ceiling fires
        // even when the buffer is small — short utterances shouldn't be
        // held forever just because they're brief.
        assert!(should_force_flush(4, 2100));
    }

    #[test]
    fn force_flush_holds_subordinate_clause_under_new_max_hold() {
        // Regression for the 2026-05-11 "Statue of Liberty" cut: a
        // partial clause held 1843 ms with 9 words used to trip the
        // old 1200 ms aged_out trigger and ship without the trailing
        // noun phrase. Under the new 2000 ms ceiling and 10-word soft
        // threshold, the accumulator keeps waiting.
        assert!(!should_force_flush(9, 1843));
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

    // ─── stable_speaker_id: majority vote over recent diariser ids ─────

    #[test]
    fn stable_id_returns_input_with_empty_history() {
        let mut h = VecDeque::new();
        assert_eq!(stable_speaker_id(3, &mut h), 3);
    }

    #[test]
    fn stable_id_holds_against_single_noisy_call() {
        // Four calls returning 0, one noisy 5 in the middle.
        // Majority is 0; the 5 should be out-voted.
        let mut h = VecDeque::new();
        stable_speaker_id(0, &mut h);
        stable_speaker_id(0, &mut h);
        let result = stable_speaker_id(5, &mut h); // noise
        assert_eq!(result, 0);
        let result = stable_speaker_id(0, &mut h);
        assert_eq!(result, 0);
    }

    #[test]
    fn stable_id_switches_after_sustained_change() {
        // After three consecutive calls with the new id, it has to
        // win — a real speaker change should not be held back forever.
        let mut h = VecDeque::new();
        stable_speaker_id(0, &mut h);
        stable_speaker_id(0, &mut h);
        assert_eq!(stable_speaker_id(1, &mut h), 0); // still 0 (2 vs 1)
        assert_eq!(stable_speaker_id(1, &mut h), 1); // 2 vs 2 → tie, recency wins
        assert_eq!(stable_speaker_id(1, &mut h), 1); // 2 vs 3 → 1 wins
    }

    #[test]
    fn stable_id_ring_is_bounded() {
        let mut h = VecDeque::new();
        for _ in 0..20 {
            stable_speaker_id(7, &mut h);
        }
        assert!(h.len() <= SPEAKER_HISTORY_K);
    }
}
