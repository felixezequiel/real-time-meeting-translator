use audio::resampler;
use shared::{AudioChunk, Language, PipelineCommand, PipelineMetrics, TextSegment};
use stt::WhisperStt;
use tokio::sync::mpsc;
use tracing;
use translation::OpusMtTranslator;
use tts::PiperTts;

use std::sync::Arc;
use std::time::Instant;

const PLAYBACK_SAMPLE_RATE: u32 = 48_000;
const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Minimum RMS energy to send audio to STT. Below this, the chunk is
/// silence or very quiet noise — sending it to Whisper would only produce
/// hallucinations. This is NOT a VAD — it's just a "is there any signal?"
/// check. Cost: one pass over the samples (~0ms).
const MIN_RMS_FOR_STT: f32 = 0.005;

/// Force-flush the sentence buffer after this many words without punctuation.
const MAX_WORDS_BEFORE_FORCE_FLUSH: usize = 15;

/// Pipeline modelled after a human simultaneous interpreter.
///
/// Architecture (3 stages, pipelined):
///
/// ```text
/// [Audio capture] ──→ [STT workers]  ──→ [Sentence buffer] ──→ [Translate+TTS worker] ──→ [Playback]
///   (parallel)         (parallel)          (ordered)              (sequential, ordered)
/// ```
///
/// - **STT** runs in parallel (spawn_blocking) — doesn't block audio capture.
/// - **Sentence buffer** accumulates text until a complete thought is detected.
/// - **Translate+TTS** runs in a SINGLE sequential worker — guarantees playback order.
pub struct SpeakerPipeline {
    pub stt: Arc<WhisperStt>,
    pub translator: Arc<OpusMtTranslator>,
    pub tts: Arc<PiperTts>,
    pub source_language: Language,
    pub flush_interval_seconds: f32,
}

impl SpeakerPipeline {
    pub fn new(
        stt: Arc<WhisperStt>,
        translator: Arc<OpusMtTranslator>,
        tts: Arc<PiperTts>,
        source_language: Language,
        flush_interval_seconds: f32,
    ) -> Self {
        Self { stt, translator, tts, source_language, flush_interval_seconds }
    }

    pub async fn run(
        self,
        mut audio_input: mpsc::UnboundedReceiver<AudioChunk>,
        audio_output: mpsc::UnboundedSender<AudioChunk>,
        mut command_rx: mpsc::Receiver<PipelineCommand>,
        _metrics_tx: mpsc::Sender<PipelineMetrics>,
    ) {
        let stt = self.stt;
        let translator = self.translator;
        let tts = self.tts;
        let source_language = self.source_language;
        let flush_interval = self.flush_interval_seconds;
        let flush_sample_count = (WHISPER_SAMPLE_RATE as f32 * flush_interval) as usize;

        let mut is_running = false;
        let mut accumulated_samples: Vec<f32> = Vec::new();
        let mut sentence_buffer = String::new();
        let mut next_seq: u64 = 0;          // next sequence number to assign
        let mut expected_seq: u64 = 0;       // next sequence number we expect to receive
        let mut pending: std::collections::BTreeMap<u64, SttResult> = std::collections::BTreeMap::new();

        // STT results arrive here (from parallel spawn_blocking tasks)
        let (stt_tx, mut stt_rx) = mpsc::unbounded_channel::<SttResult>();

        // Sentences go to the sequential translate+TTS worker (guarantees order)
        let (sentence_tx, sentence_rx) = mpsc::unbounded_channel::<String>();
        start_translate_worker(sentence_rx, translator.clone(), tts.clone(), audio_output.clone());

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
                            sentence_buffer.clear();
                        }
                    }
                }

                // ── Receive audio, accumulate, dispatch to STT ────────────
                Some(chunk) = audio_input.recv() => {
                    if !is_running { continue; }

                    accumulated_samples.extend_from_slice(&chunk.samples);

                    if accumulated_samples.len() >= flush_sample_count {
                        let samples = std::mem::take(&mut accumulated_samples);

                        // Quick energy check — skip STT if audio is silence/noise.
                        // Much cheaper than running Whisper on empty audio.
                        let rms = (samples.iter().map(|s| s * s).sum::<f32>()
                            / samples.len().max(1) as f32)
                            .sqrt();
                        if rms < MIN_RMS_FOR_STT {
                            // Still need to send empty result to keep sequence flowing
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
                            match stt_clone.transcribe(&chunk) {
                                Ok(seg) => {
                                    // Always send a result (even empty) so the reorder
                                    // buffer doesn't stall waiting for a missing sequence.
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
                                    // Send empty result to keep sequence flowing
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

                // ── Receive STT text, reorder, detect sentences ───────────
                Some(result) = stt_rx.recv() => {
                    // Insert into reorder buffer keyed by sequence number
                    pending.insert(result.seq, result);

                    // Drain all consecutive results starting from expected_seq.
                    // This guarantees sentence_buffer receives text in the same
                    // order the audio was captured, even if STT tasks finish
                    // out of order.
                    while let Some(r) = pending.remove(&expected_seq) {
                        expected_seq += 1;

                        if r.text.is_empty() { continue; }

                        tracing::info!("STT [{}]: \"{}\" ({:?}, {}ms)",
                            r.seq, r.text, r.detected_language, r.stt_duration.as_millis());

                        if r.detected_language != r.expected_language {
                            tracing::debug!("Lang guard: drop ({:?}≠{:?})",
                                r.detected_language, r.expected_language);
                            continue;
                        }

                        if !sentence_buffer.is_empty() { sentence_buffer.push(' '); }
                        sentence_buffer.push_str(&r.text);
                    }

                    // Check for complete sentences
                    if let Some((sentences, remainder)) = extract_complete_sentences(&sentence_buffer) {
                        sentence_buffer = remainder;
                        tracing::info!("→ \"{}\"", sentences);
                        let _ = sentence_tx.send(sentences);
                    } else {
                        let word_count = sentence_buffer.split_whitespace().count();
                        if word_count >= MAX_WORDS_BEFORE_FORCE_FLUSH {
                            let text = std::mem::take(&mut sentence_buffer);
                            tracing::info!("→ flush: \"{}\"", &text[..text.len().min(80)]);
                            let _ = sentence_tx.send(text);
                        }
                    }
                }

                else => break,
            }
        }

        tracing::info!("Pipeline loop ended");
    }
}

// ─── Sequential translate + TTS worker ───────────────────────────────────────

/// Spawns a single worker task that processes sentences **one at a time, in order**.
/// This guarantees that playback order matches the original speech order.
fn start_translate_worker(
    mut sentence_rx: mpsc::UnboundedReceiver<String>,
    translator: Arc<OpusMtTranslator>,
    tts: Arc<PiperTts>,
    audio_tx: mpsc::UnboundedSender<AudioChunk>,
) {
    tokio::spawn(async move {
        while let Some(text) = sentence_rx.recv().await {
            let translator = translator.clone();
            let tts = tts.clone();

            // Run blocking translate+TTS on a dedicated thread, then send audio.
            // We AWAIT the result before processing the next sentence — this is
            // what guarantees the order.
            let result = tokio::task::spawn_blocking(move || {
                let start = Instant::now();

                let segment = TextSegment::new(text, Language::English);

                let translated = match translator.translate(&segment) {
                    Ok(t) => t,
                    Err(e) => { tracing::warn!("Translation failed: {}", e); return None; }
                };
                let translate_ms = start.elapsed().as_millis();

                tracing::info!("← \"{}\" ({}ms)", translated.text, translate_ms);

                let audio_out = match tts.synthesize(&translated) {
                    Ok(a) => a,
                    Err(e) => { tracing::warn!("TTS failed: {}", e); return None; }
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

                Some(audio_out)
            })
            .await;

            if let Ok(Some(audio)) = result {
                let _ = audio_tx.send(audio);
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

// ─── Sentence detection ──────────────────────────────────────────────────────

fn extract_complete_sentences(text: &str) -> Option<(String, String)> {
    let trimmed = text.trim();
    if trimmed.is_empty() { return None; }

    let mut last_boundary = None;
    for (i, c) in trimmed.char_indices() {
        if c == '.' || c == '!' || c == '?' {
            last_boundary = Some(i);
        }
    }

    let pos = last_boundary?;
    let end = pos + 1;
    let sentences = trimmed[..end].trim().to_string();
    let remainder = trimmed[end..].trim().to_string();

    if sentences.is_empty() { None } else { Some((sentences, remainder)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_sentences_finds_period() {
        let (s, r) = extract_complete_sentences("Hello world. How are").unwrap();
        assert_eq!(s, "Hello world.");
        assert_eq!(r, "How are");
    }

    #[test]
    fn extract_sentences_multiple() {
        let (s, r) = extract_complete_sentences("First. Second! Third").unwrap();
        assert_eq!(s, "First. Second!");
        assert_eq!(r, "Third");
    }

    #[test]
    fn extract_sentences_no_boundary() {
        assert!(extract_complete_sentences("Hello world").is_none());
    }
}
