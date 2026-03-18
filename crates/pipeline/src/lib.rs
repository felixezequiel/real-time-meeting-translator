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

/// Force-flush the sentence buffer after this many words without punctuation.
const MAX_WORDS_BEFORE_FORCE_FLUSH: usize = 40;

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
                        let stt_clone = stt.clone();
                        let tx = stt_tx.clone();
                        let expected_lang = source_language;

                        tokio::task::spawn_blocking(move || {
                            let start = Instant::now();
                            let chunk = AudioChunk::new(samples, WHISPER_SAMPLE_RATE, 1);
                            match stt_clone.transcribe(&chunk) {
                                Ok(seg) if !seg.is_empty() => {
                                    let _ = tx.send(SttResult {
                                        text: seg.text,
                                        detected_language: seg.language,
                                        expected_language: expected_lang,
                                        stt_duration: start.elapsed(),
                                    });
                                }
                                Ok(_) => {}
                                Err(e) => tracing::warn!("STT failed: {}", e),
                            }
                        });
                    }
                }

                // ── Receive STT text, detect sentences ────────────────────
                Some(result) = stt_rx.recv() => {
                    tracing::info!("STT ({:?}, {}ms): \"{}\"",
                        result.detected_language,
                        result.stt_duration.as_millis(),
                        result.text);

                    if result.detected_language != result.expected_language {
                        tracing::info!("Language guard: dropping (detected={:?}, expected={:?})",
                            result.detected_language, result.expected_language);
                        continue;
                    }

                    if !sentence_buffer.is_empty() { sentence_buffer.push(' '); }
                    sentence_buffer.push_str(&result.text);

                    // Extract complete sentences and send to the ordered worker
                    if let Some((sentences, remainder)) = extract_complete_sentences(&sentence_buffer) {
                        sentence_buffer = remainder;
                        tracing::info!("→ Sentence: \"{}\"", sentences);
                        let _ = sentence_tx.send(sentences);
                    } else {
                        let word_count = sentence_buffer.split_whitespace().count();
                        if word_count >= MAX_WORDS_BEFORE_FORCE_FLUSH {
                            let text = std::mem::take(&mut sentence_buffer);
                            tracing::info!("→ Force flush ({} words): \"{}\"", word_count, &text[..text.len().min(80)]);
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

                tracing::info!("Translated: \"{}\"", translated.text);

                let tts_start = Instant::now();
                let audio_out = match tts.synthesize(&translated) {
                    Ok(a) => a,
                    Err(e) => { tracing::warn!("TTS failed: {}", e); return None; }
                };
                let tts_ms = tts_start.elapsed().as_millis();

                let audio_out = if audio_out.sample_rate != PLAYBACK_SAMPLE_RATE {
                    resampler::resample_to_target(
                        &audio_out.samples, audio_out.sample_rate, PLAYBACK_SAMPLE_RATE,
                    )
                    .map(|r| AudioChunk::new(r, PLAYBACK_SAMPLE_RATE, 1))
                    .unwrap_or(audio_out)
                } else {
                    audio_out
                };

                let total_ms = start.elapsed().as_millis();
                tracing::info!("Translate+TTS: {}ms (T={}ms, TTS={}ms)", total_ms, translate_ms, tts_ms);

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
