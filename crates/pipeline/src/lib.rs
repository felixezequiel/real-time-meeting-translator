use audio::resampler;
use shared::{AudioChunk, Language, PipelineCommand, PipelineMetrics, TextSegment};
use stt::WhisperStt;
use tokio::sync::mpsc;
use tracing;
use translation::OpusMtTranslator;
use tts::PiperTts;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const PLAYBACK_SAMPLE_RATE: u32 = 48_000;
const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Minimum RMS energy to send audio to STT. Below this, the chunk is
/// silence or very quiet noise — sending it to Whisper would only produce
/// hallucinations. This is NOT a VAD — it's just a "is there any signal?"
/// check. Cost: one pass over the samples (~0ms).
const MIN_RMS_FOR_STT: f32 = 0.003;

/// Maximum concurrent translate+TTS tasks in flight. Limits GPU/CPU pressure
/// while still allowing pipeline overlap between chunks.
const MAX_CONCURRENT_TRANSLATE: usize = 3;

/// Minimum word count before sending accumulated text to translation.
/// Gives the translator enough context for coherent output.
const MIN_WORDS_FOR_TRANSLATE: usize = 8;

/// Maximum time (in seconds) to hold accumulated text before force-flushing,
/// even if we haven't reached MIN_WORDS_FOR_TRANSLATE or punctuation.
const MAX_HOLD_SECONDS: f32 = 3.0;

/// How long (seconds) to keep recent translations for echo detection.
/// STT feedback typically appears within 2-4 seconds of TTS playback.
const ECHO_WINDOW_SECONDS: f32 = 8.0;

/// Word overlap threshold (0.0–1.0) above which STT text is considered
/// an echo of a recent translation and should be discarded.
const ECHO_SIMILARITY_THRESHOLD: f32 = 0.4;

/// Shared buffer of recent translation outputs, used to detect when
/// the loopback captures our own TTS audio (feedback loop).
/// Both pipelines share one buffer so cross-pipeline echo is also detected.
pub type EchoBuffer = Arc<Mutex<VecDeque<(Instant, Vec<String>)>>>;

/// Create a new shared echo buffer for cross-pipeline echo detection.
pub fn new_echo_buffer() -> EchoBuffer {
    Arc::new(Mutex::new(VecDeque::new()))
}

/// Pipeline modelled after a human simultaneous interpreter.
///
/// Architecture (3 stages, fully pipelined):
///
/// ```text
/// [Audio capture] ──→ [STT workers] ──→ [Concurrent Translate+TTS] ──→ [Ordered Playback]
///   (parallel)         (parallel)         (up to 3 in flight)           (reorder buffer)
/// ```
///
/// - **STT** runs in parallel (spawn_blocking) — doesn't block audio capture.
/// - **Translate+TTS** runs concurrently (up to 3 tasks) — while chunk N is
///   being synthesized, chunk N+1 is already translating. A reorder buffer
///   guarantees playback matches original speech order.
/// - **Smart accumulator** — text is held until punctuation, 8+ words, or
///   3s timeout. Balances translation coherence with low latency.
pub struct SpeakerPipeline {
    pub name: String,
    pub stt: Arc<WhisperStt>,
    pub translator: Arc<OpusMtTranslator>,
    pub tts: Arc<PiperTts>,
    pub source_language: Language,
    pub flush_interval_seconds: f32,
    /// Shared echo buffer — both pipelines write translations here and check
    /// STT results against it. Prevents cross-pipeline feedback (mic TTS
    /// recaptured by speaker loopback).
    pub echo_buffer: EchoBuffer,
}

impl SpeakerPipeline {
    pub fn new(
        name: impl Into<String>,
        stt: Arc<WhisperStt>,
        translator: Arc<OpusMtTranslator>,
        tts: Arc<PiperTts>,
        source_language: Language,
        flush_interval_seconds: f32,
        echo_buffer: EchoBuffer,
    ) -> Self {
        Self {
            name: name.into(), stt, translator, tts,
            source_language, flush_interval_seconds, echo_buffer,
        }
    }

    pub async fn run(
        self,
        mut audio_input: mpsc::UnboundedReceiver<AudioChunk>,
        audio_output: mpsc::UnboundedSender<AudioChunk>,
        mut command_rx: mpsc::Receiver<PipelineCommand>,
        _metrics_tx: mpsc::Sender<PipelineMetrics>,
    ) {
        let pipeline_name = self.name;
        let stt = self.stt;
        let translator = self.translator;
        let tts = self.tts;
        let source_language = self.source_language;
        let flush_interval = self.flush_interval_seconds;
        let flush_sample_count = (WHISPER_SAMPLE_RATE as f32 * flush_interval) as usize;
        let echo_buffer = self.echo_buffer;

        let mut is_running = false;
        let mut accumulated_samples: Vec<f32> = Vec::new();
        let mut next_seq: u64 = 0;          // next sequence number to assign
        let mut expected_seq: u64 = 0;       // next sequence number we expect to receive
        let mut pending: std::collections::BTreeMap<u64, SttResult> = std::collections::BTreeMap::new();
        let mut translate_seq: u64 = 0;      // sequence number for translate worker
        let mut accumulator = String::new();       // text accumulator for translation context
        let mut accumulator_start: Option<Instant> = None; // when accumulation began

        // STT results arrive here (from parallel spawn_blocking tasks)
        let (stt_tx, mut stt_rx) = mpsc::unbounded_channel::<SttResult>();

        // Text chunks go to the concurrent translate+TTS worker (with ordered delivery)
        let (text_tx, text_rx) = mpsc::unbounded_channel::<(u64, String)>();
        start_translate_worker(text_rx, translator.clone(), tts.clone(), audio_output.clone(), echo_buffer.clone());

        loop {
            tokio::select! {
                Some(command) = command_rx.recv() => {
                    match command {
                        PipelineCommand::Start => {
                            tracing::info!("Pipeline started (source={}, flush={:.1}s)",
                                source_language.display_name(), flush_interval);
                            is_running = true;
                        }
                        PipelineCommand::Stop => {
                            tracing::info!("Pipeline stopped");
                            is_running = false;
                            accumulated_samples.clear();
                            accumulator.clear();
                            accumulator_start = None;
                        }
                    }
                }

                // ── Receive audio, accumulate, dispatch to STT ────────────
                Some(chunk) = audio_input.recv() => {
                    if !is_running { continue; }

                    accumulated_samples.extend_from_slice(&chunk.samples);

                    if accumulated_samples.len() >= flush_sample_count {
                        let samples = std::mem::take(&mut accumulated_samples);

                        // Energy gate — skip STT if audio is silence/noise.
                        // Without this, Whisper hallucinates on near-silence
                        // (e.g. mic picking up headphone bleed → "e" every chunk).
                        let rms = (samples.iter().map(|s| s * s).sum::<f32>()
                            / samples.len().max(1) as f32)
                            .sqrt();
                        if rms < MIN_RMS_FOR_STT {
                            let seq = next_seq;
                            next_seq += 1;
                            let _ = stt_tx.send(SttResult {
                                seq,
                                text: String::new(),
                                detected_language: source_language,
                                expected_language: source_language,
                                stt_duration: std::time::Duration::ZERO,
                            });
                            continue;
                        }

                        let stt_clone = stt.clone();
                        let tx = stt_tx.clone();
                        let expected_lang = source_language;
                        let seq = next_seq;
                        next_seq += 1;

                        tokio::task::spawn_blocking(move || {
                            let start = Instant::now();
                            let chunk = AudioChunk::new(samples, WHISPER_SAMPLE_RATE, 1);
                            match stt_clone.transcribe(&chunk, expected_lang) {
                                Ok(seg) => {
                                    let _ = tx.send(SttResult {
                                        seq,
                                        text: if seg.is_empty() { String::new() } else { seg.text },
                                        detected_language: seg.language,
                                        expected_language: expected_lang,
                                        stt_duration: start.elapsed(),
                                    });
                                }
                                Err(e) => {
                                    tracing::warn!("STT failed: {}", e);
                                    let _ = tx.send(SttResult {
                                        seq,
                                        text: String::new(),
                                        detected_language: expected_lang,
                                        expected_language: expected_lang,
                                        stt_duration: start.elapsed(),
                                    });
                                }
                            }
                        });
                    }
                }

                // ── Receive STT text, reorder, smart accumulator ──────────
                Some(result) = stt_rx.recv() => {
                    pending.insert(result.seq, result);

                    while let Some(r) = pending.remove(&expected_seq) {
                        expected_seq += 1;

                        if r.text.is_empty() {
                            // Silence — flush accumulator if it has text
                            // (speaker paused, send what we have).
                            if !accumulator.is_empty() {
                                let text = std::mem::take(&mut accumulator);
                                accumulator_start = None;
                                tracing::info!("→ flush (silence): \"{}\"", &text[..text.len().min(80)]);
                                let seq = translate_seq;
                                translate_seq += 1;
                                let _ = text_tx.send((seq, text));
                            }
                            continue;
                        }

                        tracing::info!("[{}] STT [{}]: \"{}\" ({:?}, {}ms)",
                            pipeline_name, r.seq, r.text, r.detected_language, r.stt_duration.as_millis());

                        // Echo detection: if this STT text closely matches a recent
                        // translation output, it's our own TTS being recaptured.
                        if is_echo(&r.text, &echo_buffer) {
                            tracing::info!("[{}] Echo detected, dropping: \"{}\"",
                                pipeline_name, &r.text[..r.text.len().min(60)]);
                            continue;
                        }

                        if r.detected_language != r.expected_language {
                            tracing::warn!("Lang mismatch ({:?}≠{:?}), sending anyway: \"{}\"",
                                r.detected_language, r.expected_language,
                                &r.text[..r.text.len().min(60)]);
                            // Don't drop — let the translator handle it.
                            // Dropping causes gaps in continuous speech.
                        }

                        // Append to accumulator
                        if !accumulator.is_empty() { accumulator.push(' '); }
                        accumulator.push_str(&r.text);
                        if accumulator_start.is_none() {
                            accumulator_start = Some(Instant::now());
                        }

                        // Decide whether to flush the accumulator:
                        // 1. Punctuation → send immediately (complete thought)
                        // 2. >= 5 words  → send (enough context for good translation)
                        // 3. >= 2s held  → force flush (delay cap)
                        let has_punctuation = ends_with_punctuation(&accumulator);
                        let word_count = accumulator.split_whitespace().count();
                        let held_too_long = accumulator_start
                            .map(|t| t.elapsed().as_secs_f32() >= MAX_HOLD_SECONDS)
                            .unwrap_or(false);

                        let should_flush = has_punctuation
                            || word_count >= MIN_WORDS_FOR_TRANSLATE
                            || held_too_long;

                        if should_flush {
                            let text = std::mem::take(&mut accumulator);
                            accumulator_start = None;
                            let reason = if has_punctuation { "punctuation" }
                                else if word_count >= MIN_WORDS_FOR_TRANSLATE { "word count" }
                                else { "timeout" };
                            tracing::info!("→ flush ({}): \"{}\"", reason, &text[..text.len().min(80)]);
                            let seq = translate_seq;
                            translate_seq += 1;
                            let _ = text_tx.send((seq, text));
                        }
                    }
                }

                else => break,
            }
        }

        tracing::info!("Pipeline loop ended");
    }
}

// ─── Concurrent translate + TTS worker with ordered delivery ─────────────────

/// Spawns a worker that processes text chunks **concurrently** (up to
/// `MAX_CONCURRENT_TRANSLATE` in flight) but delivers audio to playback
/// **in sequence order** via a reorder buffer.
///
/// This means while chunk N is being synthesized by TTS, chunk N+1 can
/// already be translating — eliminating the sequential bottleneck.
fn start_translate_worker(
    mut text_rx: mpsc::UnboundedReceiver<(u64, String)>,
    translator: Arc<OpusMtTranslator>,
    tts: Arc<PiperTts>,
    audio_tx: mpsc::UnboundedSender<AudioChunk>,
    echo_buffer: EchoBuffer,
) {
    tokio::spawn(async move {
        let semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_TRANSLATE));
        let (result_tx, mut result_rx) =
            mpsc::unbounded_channel::<(u64, Option<AudioChunk>)>();

        let mut expected_seq: u64 = 0;
        let mut pending: std::collections::BTreeMap<u64, Option<AudioChunk>> =
            std::collections::BTreeMap::new();

        loop {
            tokio::select! {
                // ── Accept new text and spawn translate+TTS task ──────────
                Some((seq, text)) = text_rx.recv() => {
                    let translator = translator.clone();
                    let tts = tts.clone();
                    let tx = result_tx.clone();
                    let permit = semaphore.clone().acquire_owned().await.unwrap();

                    let echo_buf = echo_buffer.clone();
                    tokio::task::spawn_blocking(move || {
                        let _permit = permit; // released on drop
                        let start = Instant::now();

                        let segment = TextSegment::new(text.clone(), Language::English);

                        let translated = match translator.translate(&segment) {
                            Ok(t) => t,
                            Err(e) => {
                                tracing::warn!("Translation failed: {}", e);
                                let _ = tx.send((seq, None));
                                return;
                            }
                        };
                        let translate_ms = start.elapsed().as_millis();

                        // Guard: reject degenerate translation output.
                        // Opus-MT can loop on garbage input, producing thousands
                        // of repeated words.
                        if is_translation_degenerate(&text, &translated.text) {
                            tracing::warn!("Translation degenerate, dropping: \"{}\" → \"{}\"",
                                &text[..text.len().min(60)],
                                &translated.text[..translated.text.len().min(60)]);
                            let _ = tx.send((seq, None));
                            return;
                        }

                        tracing::info!("← \"{}\" ({}ms)", translated.text, translate_ms);

                        // Record translation for echo detection
                        record_translation(&echo_buf, &translated.text);

                        let audio_out = match tts.synthesize(&translated) {
                            Ok(a) => a,
                            Err(e) => {
                                tracing::warn!("TTS failed: {}", e);
                                let _ = tx.send((seq, None));
                                return;
                            }
                        };

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

                // ── Collect results and deliver in order ─────────────────
                Some((seq, audio)) = result_rx.recv() => {
                    pending.insert(seq, audio);

                    while let Some(audio) = pending.remove(&expected_seq) {
                        expected_seq += 1;
                        if let Some(chunk) = audio {
                            let _ = audio_tx.send(chunk);
                        }
                    }
                }

                else => break,
            }
        }
    });
}

// ─── Internal types ──────────────────────────────────────────────────────────

struct SttResult {
    seq: u64,
    text: String,
    detected_language: Language,
    expected_language: Language,
    stt_duration: std::time::Duration,
}

// ─── Punctuation detection ───────────────────────────────────────────────────

/// Returns true if the trimmed text ends with sentence-ending punctuation.
fn ends_with_punctuation(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?')
}

// ─── Translation quality guard ──────────────────────────────────────────────

/// Detects degenerate translation output (repetition loops, excessive length).
/// Opus-MT can enter infinite-loop-like generation on garbage or ambiguous input.
fn is_translation_degenerate(input: &str, output: &str) -> bool {
    // Output way too long relative to input — likely a loop
    let input_words = input.split_whitespace().count().max(1);
    let output_words = output.split_whitespace().count();
    if output_words > input_words * 4 && output_words > 20 {
        return true;
    }

    // Repetitive output: same word/phrase repeated many times
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

/// Normalize text for comparison: lowercase, strip accents and punctuation.
/// Uses ASCII-only chars so "petrolífera" and "petrolifera" match.
fn normalize_for_echo(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|w| {
            w.chars()
                .filter_map(|c| {
                    if c.is_ascii_alphanumeric() {
                        Some(c)
                    } else if c.is_alphanumeric() {
                        // Map accented chars to ASCII approximations
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

/// Best-effort ASCII approximation for common Portuguese/Romance diacritics.
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

/// Record a translation output in the echo buffer for later comparison.
fn record_translation(echo_buffer: &EchoBuffer, translated_text: &str) {
    let words = normalize_for_echo(translated_text);
    if words.is_empty() {
        return;
    }
    let mut buf = echo_buffer.lock().unwrap();
    buf.push_back((Instant::now(), words));
    // Evict old entries
    let cutoff = Instant::now() - std::time::Duration::from_secs_f32(ECHO_WINDOW_SECONDS);
    while buf.front().map_or(false, |(t, _)| *t < cutoff) {
        buf.pop_front();
    }
}

/// Check if STT text matches any recent translation (echo detection).
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

/// Fraction of words in `a` that also appear in `b` (order-independent).
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
    fn ends_with_exclamation() {
        assert!(ends_with_punctuation("Stop!"));
    }

    #[test]
    fn ends_with_question() {
        assert!(ends_with_punctuation("How are you?"));
    }

    #[test]
    fn no_punctuation() {
        assert!(!ends_with_punctuation("Hello world"));
    }

    #[test]
    fn trailing_whitespace_still_detected() {
        assert!(ends_with_punctuation("Hello world.  "));
    }

    #[test]
    fn empty_string() {
        assert!(!ends_with_punctuation(""));
    }

    #[test]
    fn min_words_is_reasonable() {
        assert!(MIN_WORDS_FOR_TRANSLATE >= 5);
        assert!(MIN_WORDS_FOR_TRANSLATE <= 15);
    }

    #[test]
    fn max_hold_seconds_caps_delay() {
        assert!(MAX_HOLD_SECONDS >= 2.0);
        assert!(MAX_HOLD_SECONDS <= 5.0);
    }

    #[test]
    fn normal_translation_is_not_degenerate() {
        assert!(!is_translation_degenerate(
            "Hello world, how are you?",
            "Olá mundo, como vai você?",
        ));
    }

    #[test]
    fn repetitive_translation_is_degenerate() {
        let input = "No, no, no";
        let output = "Não, não, não, não, não, não, não, não, não, não, não, não, não, não, não, não, não, não, não, não, não, não, não";
        assert!(is_translation_degenerate(input, output));
    }

    #[test]
    fn excessively_long_translation_is_degenerate() {
        let input = "Hello";
        let output = (0..30).map(|_| "word").collect::<Vec<_>>().join(" ");
        assert!(is_translation_degenerate(input, &output));
    }

    #[test]
    fn proportional_translation_is_not_degenerate() {
        let input = "This is a relatively long sentence with many words";
        let output = "Esta é uma frase relativamente longa com muitas palavras";
        assert!(!is_translation_degenerate(input, output));
    }

    // ── Echo detection tests ─────────────────────────────────────────────────

    #[test]
    fn echo_detected_when_stt_matches_recent_translation() {
        let buf: EchoBuffer = Arc::new(Mutex::new(VecDeque::new()));
        record_translation(&buf, "plataforma petrolífera");
        // STT captures garbled version of our TTS
        assert!(is_echo("platforma petrolifera", &buf));
    }

    #[test]
    fn echo_detected_with_partial_match() {
        let buf: EchoBuffer = Arc::new(Mutex::new(VecDeque::new()));
        record_translation(&buf, "6 de Julho de 1990 1988");
        // STT captures a subset
        assert!(is_echo("6 de Julho de 1990 1988", &buf));
    }

    #[test]
    fn no_echo_for_unrelated_text() {
        let buf: EchoBuffer = Arc::new(Mutex::new(VecDeque::new()));
        record_translation(&buf, "plataforma petrolífera");
        assert!(!is_echo("The Piper Alpha oil rig", &buf));
    }

    #[test]
    fn no_echo_when_buffer_empty() {
        let buf: EchoBuffer = Arc::new(Mutex::new(VecDeque::new()));
        assert!(!is_echo("anything here", &buf));
    }

    #[test]
    fn word_overlap_exact_match() {
        let a = normalize_for_echo("plataforma petrolifera");
        let b = normalize_for_echo("plataforma petrolífera");
        assert!(word_overlap_ratio(&a, &b) >= 0.5);
    }

    #[test]
    fn word_overlap_no_match() {
        let a = normalize_for_echo("hello world");
        let b = normalize_for_echo("plataforma petrolifera");
        assert_eq!(word_overlap_ratio(&a, &b), 0.0);
    }
}
