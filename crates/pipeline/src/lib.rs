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

/// Maximum concurrent translate+TTS tasks in flight. Limits GPU/CPU pressure
/// while still allowing pipeline overlap between chunks.
const MAX_CONCURRENT_TRANSLATE: usize = 3;

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
/// - **No sentence buffer** — each STT chunk is forwarded immediately to
///   translation. This eliminates 2-4s of accumulation delay.
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
        let mut next_seq: u64 = 0;          // next sequence number to assign
        let mut expected_seq: u64 = 0;       // next sequence number we expect to receive
        let mut pending: std::collections::BTreeMap<u64, SttResult> = std::collections::BTreeMap::new();
        let mut translate_seq: u64 = 0;      // sequence number for translate worker

        // STT results arrive here (from parallel spawn_blocking tasks)
        let (stt_tx, mut stt_rx) = mpsc::unbounded_channel::<SttResult>();

        // Text chunks go to the concurrent translate+TTS worker (with ordered delivery)
        let (text_tx, text_rx) = mpsc::unbounded_channel::<(u64, String)>();
        start_translate_worker(text_rx, translator.clone(), tts.clone(), audio_output.clone());

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
                            match stt_clone.transcribe(&chunk) {
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

                // ── Receive STT text, reorder, forward immediately ────────
                Some(result) = stt_rx.recv() => {
                    pending.insert(result.seq, result);

                    // Drain consecutive results in order and forward each
                    // non-empty chunk directly to translation — no sentence
                    // accumulation, no waiting for punctuation.
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

                        let seq = translate_seq;
                        translate_seq += 1;
                        let _ = text_tx.send((seq, r.text));
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

                    tokio::task::spawn_blocking(move || {
                        let _permit = permit; // released on drop
                        let start = Instant::now();

                        let segment = TextSegment::new(text, Language::English);

                        let translated = match translator.translate(&segment) {
                            Ok(t) => t,
                            Err(e) => {
                                tracing::warn!("Translation failed: {}", e);
                                let _ = tx.send((seq, None));
                                return;
                            }
                        };
                        let translate_ms = start.elapsed().as_millis();
                        tracing::info!("← \"{}\" ({}ms)", translated.text, translate_ms);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_concurrent_translate_is_positive() {
        assert!(MAX_CONCURRENT_TRANSLATE > 0);
    }

    #[test]
    fn min_rms_threshold_is_positive() {
        assert!(MIN_RMS_FOR_STT > 0.0);
    }
}
