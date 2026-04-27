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
/// speaker before we accept the change. Lowered to 1 — i.e. we accept
/// the new speaker_id immediately — because the TTS-bridge voice
/// hysteresis already guards against "voice flapping" downstream
/// (`pick_voice_with_hysteresis`). The pipeline now reflects who's
/// talking on the FIRST chunk where the diariser sees a different
/// person, important for documentaries where speakers alternate
/// every few seconds. Earlier value of 2 added ~280 ms of dead-air
/// before short interjections were attributed correctly.
const MIN_CHUNKS_FOR_SPEAKER_CHANGE: u32 = 1;

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

// ─── Streaming translate + TTS worker (sequential, fragment-pipelined) ──────
//
// The translation engine (Qwen 2.5 1.5B via translation_bridge.py) emits
// the translation as a sequence of *fragments* — pieces that ended on a
// punctuation mark or a 25-char word boundary. We synthesise each
// fragment as soon as it arrives and push it onto `audio_tx` in order.
// The mixer drains the channel as samples become available, so the
// listener hears the start of the translation while the LLM is still
// generating the rest.
//
// Why sequential (no semaphore, no reorder buffer):
//   - Within an utterance, fragments are inherently ordered (the bridge
//     emits them in order on the same Python subprocess stdout).
//   - Across utterances, the speaker pipeline produces them sequentially
//     too (the accumulator commits at sentence boundaries; the next
//     utterance can't be flushed before the current one is). So a
//     worker that processes one utterance fully before pulling the next
//     preserves the play order without any seq tracking.
//   - The earlier concurrent design was an artefact of atomic
//     translate-then-TTS — useful when each call took 500ms and you
//     wanted to overlap them. With streaming inside a single utterance
//     the parallelism is already there, just not at the utterance level.

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
        // Maximum age (in seconds) of a queued utterance before we
        // drop it as stale. When the LLM can't keep up with the speaker
        // (most common cause: llama-cpp-python is the CPU wheel,
        // running at ~15 tok/s instead of 70 tok/s on the GPU), text
        // arriving every ~2 s queues behind translations that take ~5 s
        // to complete. Without dropping, the perceived latency grows
        // linearly without bound — by minute 3 the listener hears
        // utterance 1. We'd rather skip the older utterances than
        // play a translation 30 s out of date.
        const MAX_QUEUE_AGE_SECONDS: f32 = 4.0;

        while let Some((mut seq, mut text, mut speaker_id, mut flushed_at)) =
            text_rx.recv().await
        {
            // Drain stale utterances ahead of this one. We pull from
            // the queue (try_recv, non-blocking) and skip anything
            // older than MAX_QUEUE_AGE_SECONDS. This keeps the worker
            // anchored to the latest meaningful utterance instead of
            // marching through an ever-growing backlog.
            let mut dropped = 0u32;
            loop {
                if flushed_at.elapsed().as_secs_f32() <= MAX_QUEUE_AGE_SECONDS {
                    break;
                }
                match text_rx.try_recv() {
                    Ok((next_seq, next_text, next_speaker, next_flushed)) => {
                        tracing::warn!(
                            "Dropping stale utterance {} (age {:.1}s): \"{}\"",
                            seq,
                            flushed_at.elapsed().as_secs_f32(),
                            &text[..text.len().min(60)]
                        );
                        dropped += 1;
                        seq = next_seq;
                        text = next_text;
                        speaker_id = next_speaker;
                        flushed_at = next_flushed;
                    }
                    Err(_) => break,
                }
            }
            if dropped > 0 {
                tracing::warn!(
                    "Worker fell behind, skipped {} utterance(s). \
                     Now processing seq {} ({:.1}s old).",
                    dropped, seq, flushed_at.elapsed().as_secs_f32(),
                );
            }

            let translator = translator.clone();
            let tts = tts.clone();
            let audio_tx_inner = audio_tx.clone();
            let metrics_inner = metrics_tx.clone();
            let echo_buf = echo_buffer.clone();

            // Resolve the voice profile for this utterance. With
            // streaming we resolve once per utterance — the profile
            // doesn't change between fragments of the same speaker.
            let voice_profile = match speaker_id {
                Some(sid) => {
                    let f0 = voice_profiles.f0_for(sid);
                    VoiceProfile {
                        target_f0_hz: f0,
                        formant_shift: formant_shift_for_f0(f0),
                        speaker_id: Some(sid),
                    }
                }
                None => VoiceProfile::default(),
            };

            // Sequential per-utterance, but inside the utterance the LLM
            // and TTS run in *parallel*: the LLM stream pushes each
            // fragment to a channel; a worker thread pulls fragments and
            // synthesises them, pushing audio to the playback channel
            // as each one finishes. This decoupling cuts the total
            // utterance latency by roughly the time of N-1 TTS calls
            // (for an utterance with N fragments) — the LLM finishes
            // generating while the early fragments are still being
            // spoken.
            let _ = tokio::task::spawn_blocking(move || {
                let segment = TextSegment::new(text.clone(), Language::English);
                let translate_start = Instant::now();

                // std::sync::mpsc — blocking send/recv, exactly what we
                // want here: the LLM stream callback is sync, the TTS
                // worker is sync, both run on OS threads.
                let (frag_tx, frag_rx) = std::sync::mpsc::channel::<String>();

                // TTS worker thread. Parallel to the LLM stream below;
                // both run independently on their own native threads
                // until the LLM closes its sender by dropping it.
                let tts_clone = tts.clone();
                let audio_tx_clone = audio_tx_inner.clone();
                let metrics_clone = metrics_inner.clone();
                let voice_profile_clone = voice_profile;
                let speaker_id_clone = speaker_id;
                let utterance_start = flushed_at;
                let tts_handle = std::thread::spawn(move || {
                    let mut first_audio_emitted = false;
                    let mut fragment_idx: u32 = 0;
                    while let Ok(fragment_text) = frag_rx.recv() {
                        fragment_idx += 1;
                        let fragment_segment = TextSegment::new(
                            fragment_text.clone(),
                            Language::English,
                        );
                        let tts_start = Instant::now();
                        let audio_out = match tts_clone.synthesize(&fragment_segment, voice_profile_clone) {
                            Ok(a) => a,
                            Err(e) => {
                                tracing::warn!("TTS failed (fragment {}): {}", fragment_idx, e);
                                continue;
                            }
                        };
                        let tts_elapsed = tts_start.elapsed();
                        let _ = metrics_clone.try_send(PipelineMetrics::new(
                            "tts".to_string(),
                            tts_elapsed,
                        ));
                        tracing::info!(
                            "TTS (speaker={:?}, target_f0={:.0}, formant={:.2}, fragment {}): {} samples ({}ms)",
                            speaker_id_clone,
                            voice_profile_clone.target_f0_hz,
                            voice_profile_clone.formant_shift,
                            fragment_idx,
                            audio_out.samples.len(),
                            tts_elapsed.as_millis(),
                        );

                        let audio_out = if audio_out.sample_rate != PLAYBACK_SAMPLE_RATE {
                            resampler::resample_to_target(
                                &audio_out.samples,
                                audio_out.sample_rate,
                                PLAYBACK_SAMPLE_RATE,
                            )
                            .map(|r| AudioChunk::new(r, PLAYBACK_SAMPLE_RATE, 1))
                            .unwrap_or(audio_out)
                        } else {
                            audio_out
                        };

                        let _ = audio_tx_clone.send(audio_out);

                        if !first_audio_emitted {
                            // Time to first audio (TTFA): the metric the
                            // listener actually feels — from the moment
                            // the source utterance flushed to the moment
                            // we hand the first sample to the mixer.
                            let _ = metrics_clone.try_send(PipelineMetrics::new(
                                "ttfa".to_string(),
                                utterance_start.elapsed(),
                            ));
                            first_audio_emitted = true;
                        }
                    }
                });

                let mut accumulated = String::new();
                let mut first_fragment = true;
                let mut fragment_count: u32 = 0;

                let stream_result = translator.translate_stream(&segment, |fragment| {
                    if fragment.is_final {
                        return;
                    }
                    let fragment_text = fragment.text.trim();
                    if fragment_text.is_empty() {
                        accumulated.push_str(&fragment.text);
                        return;
                    }
                    if first_fragment {
                        let _ = metrics_inner.try_send(PipelineMetrics::new(
                            "translate_first_fragment".to_string(),
                            translate_start.elapsed(),
                        ));
                        first_fragment = false;
                    }
                    fragment_count += 1;
                    accumulated.push_str(&fragment.text);
                    tracing::info!("← fragment {}: \"{}\"", fragment_count, fragment_text);
                    // Hand the fragment off to the TTS worker; the LLM
                    // stream continues immediately.
                    let _ = frag_tx.send(fragment.text.clone());
                });

                // Closing the sender lets the TTS worker drain its
                // queue and exit naturally.
                drop(frag_tx);
                let _ = tts_handle.join();

                let translate_elapsed = translate_start.elapsed();
                let _ = metrics_inner.try_send(PipelineMetrics::new(
                    "translate".to_string(),
                    translate_elapsed,
                ));

                if let Err(e) = stream_result {
                    tracing::warn!("Translation stream failed: {}", e);
                    return;
                }
                if accumulated.trim().is_empty() {
                    return;
                }

                tracing::info!(
                    "← \"{}\" ({}ms, {} fragments)",
                    accumulated.trim(),
                    translate_elapsed.as_millis(),
                    fragment_count,
                );

                // Echo registry is kept tight: only record translations
                // that look sane. Degenerate output (the LLM fell into a
                // repetition loop) would falsely match anything in the
                // STT stream, and we'd start dropping legitimate input.
                if is_translation_degenerate(&text, accumulated.trim()) {
                    tracing::warn!(
                        "Translation looks degenerate, skipping echo record: \"{}\" → \"{}\"",
                        &text[..text.len().min(60)],
                        &accumulated[..accumulated.len().min(60)],
                    );
                } else {
                    record_translation(&echo_buf, accumulated.trim());
                }

                let _ = metrics_inner.try_send(PipelineMetrics::new(
                    "total".to_string(),
                    flushed_at.elapsed(),
                ));
            }).await;
            let _ = seq; // sequence number is no longer used (no reorder buffer).
        }
    });
}

/// Formant-shift ratio for the TTS bridge. We deliberately return 1.0
/// (= no warp) for every speaker now. Earlier revisions ran a heuristic
/// that warped the spectral envelope by ±15 % based on the speaker's
/// F0, but the user reported the result as "extremely robotic": even
/// small spectral-envelope warps push the WORLD analysis-synthesis
/// output away from natural-sounding human speech, because the phase
/// reconstruction has no idea about the new envelope and produces
/// audible artefacts. Differentiation by **pitch alone** (F0 swap)
/// keeps voices distinguishable without sounding fanha. If a deeper
/// "vocal weight" cue ever becomes worth the artefact, this is the
/// single function to flip.
fn formant_shift_for_f0(_f0_hz: f32) -> f32 {
    1.0
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
    fn formant_shift_always_returns_one() {
        // Formant warping is currently disabled across the board because
        // even small spectral-envelope shifts produced audible WORLD
        // analysis-synthesis artefacts ("robotic colour"). If a future
        // experiment re-enables it, this test will fail and force the
        // owner to update the contract intentionally.
        for f0 in [0.0, 50.0, 120.0, 220.0, 500.0] {
            assert!(
                (formant_shift_for_f0(f0) - 1.0).abs() < 1e-6,
                "formant_shift_for_f0({f0}) should be exactly 1.0",
            );
        }
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
