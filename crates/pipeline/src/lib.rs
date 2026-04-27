use audio::resampler;
use audio::SileroVad;
use diarization::OnlineDiarizer;
use shared::{AudioChunk, Language, PipelineCommand, PipelineMetrics, TextSegment};
use stt::{CommittedWords, StreamingSession, WhisperStt};
use tokio::sync::mpsc;
use tracing;
use translation::OpusMtTranslator;
use tts::{PiperTts, VoiceProfile};

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const PLAYBACK_SAMPLE_RATE: u32 = 48_000;
const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Energy floor used only when Silero VAD is unavailable (model file
/// missing or ORT init failure). Same threshold the legacy gate used —
/// the neural VAD is the primary path; this keeps the pipeline alive on
/// a fresh checkout that hasn't run install.ps1 yet.
const FALLBACK_MIN_RMS_FOR_STT: f32 = 0.003;

/// Maximum concurrent translate+TTS tasks in flight. Limits GPU/CPU pressure
/// while still allowing pipeline overlap between sentences.
const MAX_CONCURRENT_TRANSLATE: usize = 3;

/// Minimum word count required before a timeout flush is allowed. Acts as a
/// floor so we never ship a fragment like "The market has been" to the
/// translator.
const MIN_WORDS_FOR_TIMEOUT_FLUSH: usize = 4;

/// Maximum time the text accumulator may hold without flushing. A simultaneous
/// interpreter waits through long clauses rather than cut mid-thought; we
/// bound the wait so a stuck accumulator eventually drains.
const MAX_HOLD_SECONDS: f32 = 6.0;

/// How long to keep recent translations for echo detection. STT feedback
/// typically appears within 2-4 seconds of TTS playback.
const ECHO_WINDOW_SECONDS: f32 = 8.0;

/// Word overlap threshold above which STT text is considered an echo of a
/// recent translation and dropped.
const ECHO_SIMILARITY_THRESHOLD: f32 = 0.4;

/// Number of consecutive chunks the diarizer must identify as a different
/// speaker before we accept the change and flush the accumulator. Hides
/// single-chunk wobble that used to fragment sentences.
const MIN_CHUNKS_FOR_SPEAKER_CHANGE: u32 = 2;

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

/// Shared buffer of recent translation outputs, used to detect when the
/// loopback captures our own TTS audio (feedback loop). Both pipelines
/// share one buffer so cross-pipeline echo is also detected.
pub type EchoBuffer = Arc<Mutex<VecDeque<(Instant, Vec<String>)>>>;

/// Create a new shared echo buffer for cross-pipeline echo detection.
pub fn new_echo_buffer() -> EchoBuffer {
    Arc::new(Mutex::new(VecDeque::new()))
}

// ─── Voice profile registry: per-speaker running F0 ─────────────────────────
//
// Replaces the file-based `SpeakerRegistry` from the CosyVoice era. Instead
// of materialising a reference WAV on disk, we just track a running F0 mean
// per speaker_id. The TTS stage reads the profile to bend Piper's output
// pitch towards the original speaker — same "voices sound different per
// person" UX the cloned-voice path was meant to provide, at a fraction of
// the cost (no model, no GPU, ~10 ms of pyworld DSP per chunk).
//
// One registry per pipeline branch, shared with the translate worker via
// `Arc<Mutex<>>` so the TTS stage can read it without crossing the tokio
// select loop.

#[derive(Default)]
struct VoiceProfileInner {
    f0_by_speaker: HashMap<u32, f32>,
}

pub struct VoiceProfileRegistry {
    inner: Mutex<VoiceProfileInner>,
}

impl VoiceProfileRegistry {
    fn new() -> Self {
        Self {
            inner: Mutex::new(VoiceProfileInner::default()),
        }
    }

    /// Mix `f0_hz` into `speaker_id`'s running mean. Silently skips
    /// implausible values (the diarizer returns 0.0 when no voiced frame
    /// could be detected, and very high/low values are usually pyworld
    /// noise on near-silent audio).
    fn record_f0(&self, speaker_id: u32, f0_hz: f32) {
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
    /// as "use Piper's default voice unchanged".
    fn f0_for(&self, speaker_id: u32) -> f32 {
        self.inner
            .lock()
            .ok()
            .and_then(|g| g.f0_by_speaker.get(&speaker_id).copied())
            .unwrap_or(0.0)
    }
}

/// Pipeline modelled after a human simultaneous interpreter, driven by
/// streaming STT (Local Agreement-style boundary detection) and Piper TTS
/// with per-speaker pitch shifting.
///
/// Stages (fully pipelined):
///
/// ```text
/// [Audio capture] → [STT worker] → [Text accumulator] → [Translate+TTS] → [Ordered playback]
///   (parallel)      (single,        (in-line, sentence    (up to 3 in       (reorder buffer)
///                    serial)         boundary detect)      flight)
/// ```
pub struct SpeakerPipeline {
    pub name: String,
    pub stt: Arc<WhisperStt>,
    pub translator: Arc<OpusMtTranslator>,
    pub tts: Arc<PiperTts>,
    pub source_language: Language,
    pub echo_buffer: EchoBuffer,
    /// Optional online diariser. When present the pipeline identifies a
    /// speaker_id per chunk and tracks per-speaker F0 to drive the TTS
    /// pitch shifter — without it the TTS uses the default Piper voice.
    pub diarizer: Option<Arc<OnlineDiarizer>>,
    /// Optional neural VAD (ADR 0008). When present, decides per-chunk
    /// whether audio carries speech; when absent the pipeline falls back
    /// to a simple RMS energy gate. Each pipeline owns its own instance
    /// because the model is stateful (LSTM) and sharing one would mix
    /// contexts between mic and loopback streams.
    pub vad: Option<Arc<SileroVad>>,
}

impl SpeakerPipeline {
    pub fn new(
        name: impl Into<String>,
        stt: Arc<WhisperStt>,
        translator: Arc<OpusMtTranslator>,
        tts: Arc<PiperTts>,
        source_language: Language,
        echo_buffer: EchoBuffer,
    ) -> Self {
        Self {
            name: name.into(), stt, translator, tts,
            source_language, echo_buffer,
            diarizer: None,
            vad: None,
        }
    }

    pub fn with_diarizer(mut self, diarizer: Arc<OnlineDiarizer>) -> Self {
        self.diarizer = Some(diarizer);
        self
    }

    pub fn with_vad(mut self, vad: Arc<SileroVad>) -> Self {
        self.vad = Some(vad);
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
        let stt = self.stt;
        let translator = self.translator;
        let tts = self.tts;
        let source_language = self.source_language;
        let echo_buffer = self.echo_buffer;
        let diarizer = self.diarizer;
        let vad = self.vad;
        let voice_profiles = Arc::new(VoiceProfileRegistry::new());

        // ── STT worker channel ────────────────────────────────────────────
        let (raw_audio_tx, raw_audio_rx) = mpsc::unbounded_channel::<Vec<f32>>();
        let (committed_tx, mut committed_rx) =
            mpsc::unbounded_channel::<(CommittedWords, Option<u32>)>();
        start_stt_worker(
            pipeline_name.clone(),
            stt.clone(),
            source_language,
            raw_audio_rx,
            committed_tx,
            diarizer.clone(),
            voice_profiles.clone(),
            metrics_tx.clone(),
        );

        // ── Translate+TTS worker channel ──────────────────────────────────
        // The Instant carried alongside (seq, text, speaker) is the moment
        // the accumulator flushed — used to compute the end-to-end "total"
        // metric (text-flush → first audio sample sent to playback).
        let (text_tx, text_rx) =
            mpsc::unbounded_channel::<(u64, String, Option<u32>, Instant)>();
        start_translate_worker(
            text_rx,
            translator.clone(),
            tts.clone(),
            audio_output.clone(),
            echo_buffer.clone(),
            voice_profiles.clone(),
            metrics_tx.clone(),
        );

        let mut is_running = false;
        let mut accumulator = String::new();
        let mut accumulator_start: Option<Instant> = None;
        let mut accumulator_speaker: Option<u32> = None;
        let mut translate_seq: u64 = 0;

        loop {
            tokio::select! {
                Some(command) = command_rx.recv() => {
                    match command {
                        PipelineCommand::Start => {
                            tracing::info!("[{}] Pipeline started (source={})",
                                pipeline_name, source_language.display_name());
                            is_running = true;
                        }
                        PipelineCommand::Stop => {
                            tracing::info!("[{}] Pipeline stopped", pipeline_name);
                            is_running = false;
                            accumulator.clear();
                            accumulator_start = None;
                            accumulator_speaker = None;
                        }
                    }
                }

                Some(chunk) = audio_input.recv() => {
                    if !is_running { continue; }

                    let vad_start = Instant::now();
                    let has_speech = match vad.as_ref() {
                        Some(v) => v.has_speech(&chunk.samples),
                        None => {
                            let rms = (chunk.samples.iter().map(|s| s * s).sum::<f32>()
                                / chunk.samples.len().max(1) as f32).sqrt();
                            rms >= FALLBACK_MIN_RMS_FOR_STT
                        }
                    };
                    let _ = metrics_tx.try_send(PipelineMetrics::new(
                        "vad".to_string(),
                        vad_start.elapsed(),
                    ));

                    if !has_speech {
                        let _ = raw_audio_tx.send(Vec::new());
                        continue;
                    }
                    let _ = raw_audio_tx.send(chunk.samples);
                }

                Some((committed, speaker_id)) = committed_rx.recv() => {
                    let text_fragment = committed.words.join(" ");
                    if text_fragment.trim().is_empty() {
                        continue;
                    }

                    tracing::info!(
                        "[{}] committed (speaker={:?}, lang={:?}, tail_silent={}): \"{}\"",
                        pipeline_name, speaker_id, committed.language,
                        committed.tail_silent,
                        &text_fragment[..text_fragment.len().min(80)],
                    );

                    if is_echo(&text_fragment, &echo_buffer) {
                        tracing::info!(
                            "[{}] Echo detected, dropping: \"{}\"",
                            pipeline_name,
                            &text_fragment[..text_fragment.len().min(60)]
                        );
                        continue;
                    }

                    if let Some(new_sid) = speaker_id {
                        match accumulator_speaker {
                            Some(current) if current != new_sid && !accumulator.is_empty() => {
                                let text = std::mem::take(&mut accumulator);
                                accumulator_start = None;
                                tracing::info!(
                                    "→ flush (speaker change {} → {}): \"{}\"",
                                    current, new_sid,
                                    &text[..text.len().min(80)]
                                );
                                let seq = translate_seq;
                                translate_seq += 1;
                                let _ = text_tx.send((seq, text, Some(current), Instant::now()));
                            }
                            _ => {}
                        }
                        accumulator_speaker = Some(new_sid);
                    }

                    if !accumulator.is_empty() { accumulator.push(' '); }
                    accumulator.push_str(text_fragment.trim());
                    if accumulator_start.is_none() {
                        accumulator_start = Some(Instant::now());
                    }

                    let has_punctuation = ends_with_punctuation(&accumulator);
                    let word_count = accumulator.split_whitespace().count();
                    let looks_incomplete = ends_with_continuation_word(&accumulator);
                    let held_too_long = accumulator_start
                        .map(|t| t.elapsed().as_secs_f32() >= MAX_HOLD_SECONDS)
                        .unwrap_or(false);
                    let speaker_paused = committed.tail_silent && !looks_incomplete;

                    let should_flush = has_punctuation
                        || speaker_paused
                        || (held_too_long && word_count >= MIN_WORDS_FOR_TIMEOUT_FLUSH);

                    if should_flush {
                        let text = std::mem::take(&mut accumulator);
                        let flushed_speaker = accumulator_speaker;
                        accumulator_start = None;
                        let reason = if has_punctuation { "punctuation" }
                            else if speaker_paused { "pause" }
                            else { "timeout" };
                        tracing::info!("→ flush ({}): \"{}\"", reason, &text[..text.len().min(80)]);
                        let seq = translate_seq;
                        translate_seq += 1;
                        let _ = text_tx.send((seq, text, flushed_speaker, Instant::now()));
                    }
                }

                else => break,
            }
        }

        tracing::info!("[{}] Pipeline loop ended", pipeline_name);
    }
}

// ─── STT worker (single long-lived spawn_blocking task) ──────────────────────

fn start_stt_worker(
    pipeline_name: String,
    stt: Arc<WhisperStt>,
    language: Language,
    mut raw_audio_rx: mpsc::UnboundedReceiver<Vec<f32>>,
    committed_tx: mpsc::UnboundedSender<(CommittedWords, Option<u32>)>,
    diarizer: Option<Arc<OnlineDiarizer>>,
    voice_profiles: Arc<VoiceProfileRegistry>,
    metrics_tx: mpsc::Sender<PipelineMetrics>,
) {
    tokio::task::spawn_blocking(move || {
        let mut session = StreamingSession::new(stt, language);
        let mut stable_speaker_id: Option<u32> = None;
        let mut pending_speaker_id: Option<u32> = None;
        let mut pending_count: u32 = 0;

        while let Some(samples) = raw_audio_rx.blocking_recv() {
            if samples.is_empty() {
                if let Some(committed) = session.flush_tentative() {
                    let _ = committed_tx.send((committed, stable_speaker_id));
                }
                continue;
            }

            // Diarisation + F0 (if enabled). Both come from the same
            // bridge so we pay one round-trip per chunk for both pieces
            // of metadata.
            let (raw_speaker_id, f0_hz) = match diarizer.as_ref() {
                Some(d) => match d.identify(&samples, WHISPER_SAMPLE_RATE) {
                    Ok(Some(id)) => (Some(id.speaker_id), id.f0_hz),
                    Ok(None) => (None, 0.0),
                    Err(e) => {
                        tracing::warn!("[{}] diarisation failed: {}", pipeline_name, e);
                        (None, 0.0)
                    }
                },
                None => (None, 0.0),
            };

            stable_speaker_id = match (raw_speaker_id, stable_speaker_id) {
                (Some(new), Some(curr)) if new != curr => {
                    if pending_speaker_id == Some(new) {
                        pending_count += 1;
                        if pending_count >= MIN_CHUNKS_FOR_SPEAKER_CHANGE {
                            pending_speaker_id = None;
                            pending_count = 0;
                            Some(new)
                        } else {
                            Some(curr)
                        }
                    } else {
                        pending_speaker_id = Some(new);
                        pending_count = 1;
                        Some(curr)
                    }
                }
                (Some(new), None) => {
                    pending_speaker_id = None;
                    pending_count = 0;
                    Some(new)
                }
                (Some(_new), Some(curr)) => {
                    pending_speaker_id = None;
                    pending_count = 0;
                    Some(curr)
                }
                (None, current) => current,
            };

            // Update the per-speaker F0 profile. We attribute the F0
            // measurement to the *stable* speaker, not the raw one, so
            // a single-chunk identification glitch doesn't pollute a
            // confirmed speaker's profile.
            if let (Some(sid), true) = (stable_speaker_id, f0_hz > 0.0) {
                voice_profiles.record_f0(sid, f0_hz);
            }

            // The streaming session may run zero or one Whisper inference
            // depending on its internal throttle (`MIN_INFERENCE_INTERVAL_MS`).
            // We always tag the stage time, even when the throttle skipped
            // the inference — that makes the metric "wall time of the STT
            // worker iteration", which is what the UI panel cares about.
            let stt_start = Instant::now();
            let committed = session.push_audio(&samples);
            let _ = metrics_tx.try_send(PipelineMetrics::new(
                "stt".to_string(),
                stt_start.elapsed(),
            ));
            if let Some(committed) = committed {
                let _ = committed_tx.send((committed, stable_speaker_id));
            }
        }

        if let Some(committed) = session.flush_tentative() {
            let _ = committed_tx.send((committed, stable_speaker_id));
        }
        session.reset();
    });
}

// ─── Concurrent translate + TTS worker with ordered delivery ─────────────────

fn start_translate_worker(
    mut text_rx: mpsc::UnboundedReceiver<(u64, String, Option<u32>, Instant)>,
    translator: Arc<OpusMtTranslator>,
    tts: Arc<PiperTts>,
    audio_tx: mpsc::UnboundedSender<AudioChunk>,
    echo_buffer: EchoBuffer,
    voice_profiles: Arc<VoiceProfileRegistry>,
    metrics_tx: mpsc::Sender<PipelineMetrics>,
) {
    tokio::spawn(async move {
        let semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_TRANSLATE));
        let (result_tx, mut result_rx) =
            mpsc::unbounded_channel::<(u64, Option<AudioChunk>)>();

        let mut expected_seq: u64 = 0;
        let mut pending: std::collections::BTreeMap<u64, Option<AudioChunk>> =
            std::collections::BTreeMap::new();
        // Maps each in-flight seq to the moment its source text flushed.
        // We use it to compute the end-to-end "total" metric exactly once,
        // when the corresponding audio chunk reaches `audio_tx.send()`.
        let mut flush_instants: std::collections::HashMap<u64, Instant> =
            std::collections::HashMap::new();

        loop {
            tokio::select! {
                Some((seq, text, speaker_id, flushed_at)) = text_rx.recv() => {
                    flush_instants.insert(seq, flushed_at);
                    let translator = translator.clone();
                    let tts = tts.clone();
                    let tx = result_tx.clone();
                    let metrics = metrics_tx.clone();
                    let permit = semaphore.clone().acquire_owned().await.unwrap();

                    let echo_buf = echo_buffer.clone();
                    // Resolve the voice profile for this utterance: pick
                    // the speaker's running F0, or VoiceProfile::default()
                    // (zero F0 → bridge skips the analysis-synthesis pass)
                    // when no profile has been recorded yet.
                    let voice_profile = match speaker_id {
                        Some(sid) => {
                            let f0 = voice_profiles.f0_for(sid);
                            VoiceProfile {
                                target_f0_hz: f0,
                                formant_shift: formant_shift_for_f0(f0),
                            }
                        }
                        None => VoiceProfile::default(),
                    };

                    tokio::task::spawn_blocking(move || {
                        let _permit = permit;
                        let translate_start = Instant::now();

                        let segment = TextSegment::new(text.clone(), Language::English);

                        let translated = match translator.translate(&segment) {
                            Ok(t) => t,
                            Err(e) => {
                                tracing::warn!("Translation failed: {}", e);
                                let _ = tx.send((seq, None));
                                return;
                            }
                        };
                        let translate_elapsed = translate_start.elapsed();
                        let _ = metrics.try_send(PipelineMetrics::new(
                            "translate".to_string(),
                            translate_elapsed,
                        ));

                        if is_translation_degenerate(&text, &translated.text) {
                            tracing::warn!("Translation degenerate, dropping: \"{}\" → \"{}\"",
                                &text[..text.len().min(60)],
                                &translated.text[..translated.text.len().min(60)]);
                            let _ = tx.send((seq, None));
                            return;
                        }

                        tracing::info!("← \"{}\" ({}ms)", translated.text, translate_elapsed.as_millis());
                        record_translation(&echo_buf, &translated.text);

                        let tts_start = Instant::now();
                        let audio_out = match tts.synthesize(&translated, voice_profile) {
                            Ok(a) => a,
                            Err(e) => {
                                tracing::warn!("TTS failed: {}", e);
                                let _ = tx.send((seq, None));
                                return;
                            }
                        };
                        let tts_elapsed = tts_start.elapsed();
                        let _ = metrics.try_send(PipelineMetrics::new(
                            "tts".to_string(),
                            tts_elapsed,
                        ));
                        tracing::info!(
                            "TTS (speaker={:?}, target_f0={:.0}, formant={:.2}): {} samples ({}ms)",
                            speaker_id,
                            voice_profile.target_f0_hz,
                            voice_profile.formant_shift,
                            audio_out.samples.len(),
                            tts_elapsed.as_millis(),
                        );

                        let audio_out = if audio_out.sample_rate != PLAYBACK_SAMPLE_RATE {
                            resampler::resample_to_target(
                                &audio_out.samples, audio_out.sample_rate, PLAYBACK_SAMPLE_RATE,
                            )
                            .map(|r| AudioChunk::new(r, PLAYBACK_SAMPLE_RATE, 1))
                            .unwrap_or(audio_out)
                        } else {
                            audio_out
                        };

                        let _ = tx.send((seq, Some(audio_out)));
                    });
                }

                Some((seq, audio)) = result_rx.recv() => {
                    pending.insert(seq, audio);
                    while let Some(audio) = pending.remove(&expected_seq) {
                        let total = flush_instants
                            .remove(&expected_seq)
                            .map(|t| t.elapsed());
                        expected_seq += 1;
                        if let Some(chunk) = audio {
                            let _ = audio_tx.send(chunk);
                            if let Some(elapsed) = total {
                                let _ = metrics_tx.try_send(PipelineMetrics::new(
                                    "total".to_string(),
                                    elapsed,
                                ));
                            }
                        }
                    }
                }

                else => break,
            }
        }
    });
}

/// Heuristic mapping from running mean F0 to a formant-shift ratio.
/// Higher F0 (typical female / child voices) → narrower vocal tract →
/// formant_shift slightly below 1. Lower F0 → wider tract → above 1.
/// The ratio is bounded conservatively (0.85–1.15) because aggressive
/// shifts produce unnatural artefacts in the analysis-synthesis output.
fn formant_shift_for_f0(f0_hz: f32) -> f32 {
    if f0_hz <= 0.0 {
        return 1.0;
    }
    // Anchor: a ~120 Hz adult-male voice maps to formant_shift = 1.0,
    // a ~220 Hz typical adult-female voice maps to ~0.92.
    let normalised = (f0_hz - 120.0) / 100.0;
    (1.0 - normalised * 0.08).clamp(0.85, 1.15)
}

// ─── Punctuation detection ───────────────────────────────────────────────────

fn ends_with_punctuation(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?')
}

fn ends_with_continuation_word(text: &str) -> bool {
    const CONTINUATIONS: &[&str] = &[
        "and", "but", "or", "nor", "so", "because", "though", "although",
        "if", "unless", "while", "whereas", "that", "which", "who", "whom",
        "whose", "when", "where", "why", "how", "however", "therefore",
        "thus", "since", "as", "of", "to", "for", "with", "by", "in", "on",
        "at", "the", "a", "an",
        "e", "mas", "ou", "porque", "porém", "porem", "embora", "se",
        "enquanto", "que", "qual", "quais", "quando", "onde", "como",
        "então", "entao", "portanto", "pois", "de", "da", "do", "das", "dos",
        "para", "pra", "com", "por", "em", "no", "na", "nos", "nas", "ao",
        "aos", "à", "às",
    ];

    let last = text
        .trim_end_matches(|c: char| c.is_ascii_punctuation())
        .split_whitespace()
        .next_back();

    match last {
        Some(word) => {
            let normalized: String = word
                .chars()
                .filter(|c| c.is_alphabetic())
                .flat_map(char::to_lowercase)
                .collect();
            !normalized.is_empty() && CONTINUATIONS.contains(&normalized.as_str())
        }
        None => false,
    }
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

    #[test]
    fn ends_with_period() {
        assert!(ends_with_punctuation("Hello world."));
    }

    #[test]
    fn no_punctuation() {
        assert!(!ends_with_punctuation("Hello world"));
    }

    #[test]
    fn voice_profile_registry_records_and_recalls() {
        let reg = VoiceProfileRegistry::new();
        reg.record_f0(0, 200.0);
        reg.record_f0(0, 200.0);
        reg.record_f0(0, 200.0);
        let f0 = reg.f0_for(0);
        assert!((f0 - 200.0).abs() < 5.0);  // converged towards 200 Hz
    }

    #[test]
    fn voice_profile_registry_returns_zero_for_unknown_speaker() {
        let reg = VoiceProfileRegistry::new();
        assert_eq!(reg.f0_for(42), 0.0);
    }

    #[test]
    fn voice_profile_registry_clamps_outliers() {
        let reg = VoiceProfileRegistry::new();
        reg.record_f0(0, 200.0);
        reg.record_f0(0, 5000.0);  // pyworld noise — must be ignored
        let f0 = reg.f0_for(0);
        assert!((f0 - 200.0).abs() < 5.0);
    }

    #[test]
    fn formant_shift_is_unity_for_unknown_f0() {
        assert!((formant_shift_for_f0(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn formant_shift_decreases_for_higher_f0() {
        let male = formant_shift_for_f0(120.0);
        let female = formant_shift_for_f0(220.0);
        assert!(female < male);
    }

    #[test]
    fn formant_shift_is_clamped() {
        let extreme_low = formant_shift_for_f0(50.0);
        let extreme_high = formant_shift_for_f0(500.0);
        assert!((0.85..=1.15).contains(&extreme_low));
        assert!((0.85..=1.15).contains(&extreme_high));
    }

    #[test]
    fn min_words_timeout_floor_is_reasonable() {
        assert!(MIN_WORDS_FOR_TIMEOUT_FLUSH >= 3);
    }

    #[test]
    fn ends_with_conjunction_english() {
        assert!(ends_with_continuation_word("the market crashed because"));
        assert!(ends_with_continuation_word("we arrived and"));
    }

    #[test]
    fn complete_sentence_is_not_continuation() {
        assert!(!ends_with_continuation_word("the market crashed"));
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
