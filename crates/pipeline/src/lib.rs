use audio::resampler;
use audio::vad::EnergyVad;
use shared::{AudioChunk, PipelineCommand, PipelineMetrics};
use stt::WhisperStt;
use tokio::sync::mpsc;
use tracing;
use translation::OpusMtTranslator;
use tts::PiperTts;

use std::sync::Arc;
use std::time::Instant;

const PLAYBACK_SAMPLE_RATE: u32 = 48_000;

pub struct SpeakerPipeline {
    pub stt: WhisperStt,
    pub translator: OpusMtTranslator,
    pub tts: PiperTts,
    pub vad: EnergyVad,
}

impl SpeakerPipeline {
    pub fn new(
        stt: WhisperStt,
        translator: OpusMtTranslator,
        tts: PiperTts,
        vad: EnergyVad,
    ) -> Self {
        Self {
            stt,
            translator,
            tts,
            vad,
        }
    }

    pub async fn run(
        self,
        mut audio_input: mpsc::Receiver<AudioChunk>,
        audio_output: mpsc::Sender<AudioChunk>,
        mut command_rx: mpsc::Receiver<PipelineCommand>,
        metrics_tx: mpsc::Sender<PipelineMetrics>,
    ) {
        let stt = Arc::new(self.stt);
        let translator = Arc::new(self.translator);
        let tts = Arc::new(self.tts);
        let vad = self.vad;

        let mut is_running = true;

        loop {
            tokio::select! {
                Some(command) = command_rx.recv() => {
                    match command {
                        PipelineCommand::Start => {
                            tracing::info!("Pipeline started");
                            is_running = true;
                        }
                        PipelineCommand::Stop => {
                            tracing::info!("Pipeline stopped");
                            is_running = false;
                        }
                    }
                }
                Some(chunk) = audio_input.recv() => {
                    if !is_running {
                        continue;
                    }

                    let rms = calculate_rms(&chunk.samples);
                    let has_speech = vad.contains_speech(&chunk.samples);
                    tracing::info!(
                        "Audio chunk: {} samples, RMS={:.6}, speech={}",
                        chunk.samples.len(),
                        rms,
                        has_speech
                    );

                    if !has_speech {
                        continue;
                    }

                    let pipeline_start = Instant::now();
                    let stt_ref = stt.clone();
                    let translator_ref = translator.clone();
                    let tts_ref = tts.clone();
                    let output_tx = audio_output.clone();
                    let metrics = metrics_tx.clone();

                    tokio::spawn(async move {
                        // STT
                        let stt_start = Instant::now();
                        let text_segment = match stt_ref.transcribe(&chunk) {
                            Ok(segment) => segment,
                            Err(e) => {
                                tracing::warn!("STT failed: {}", e);
                                return;
                            }
                        };
                        let stt_duration = stt_start.elapsed();
                        let _ = metrics.send(PipelineMetrics::new("stt".to_string(), stt_duration)).await;

                        if text_segment.is_empty() {
                            return;
                        }

                        tracing::info!("STT: \"{}\"", text_segment.text);

                        // Translation
                        let translate_start = Instant::now();
                        let translated = match translator_ref.translate(&text_segment) {
                            Ok(t) => t,
                            Err(e) => {
                                tracing::warn!("Translation failed: {}", e);
                                return;
                            }
                        };
                        let translate_duration = translate_start.elapsed();
                        let _ = metrics.send(PipelineMetrics::new("translation".to_string(), translate_duration)).await;

                        tracing::info!("Translated: \"{}\"", translated.text);

                        // TTS
                        let tts_start = Instant::now();
                        let audio_out = match tts_ref.synthesize(&translated) {
                            Ok(a) => a,
                            Err(e) => {
                                tracing::warn!("TTS failed: {}", e);
                                return;
                            }
                        };
                        // Resample TTS output to match playback device sample rate
                        let audio_out = if audio_out.sample_rate != PLAYBACK_SAMPLE_RATE {
                            match resampler::resample_to_target(
                                &audio_out.samples,
                                audio_out.sample_rate,
                                PLAYBACK_SAMPLE_RATE,
                            ) {
                                Ok(resampled) => AudioChunk::new(resampled, PLAYBACK_SAMPLE_RATE, 1),
                                Err(e) => {
                                    tracing::warn!("Resample failed: {}, using original", e);
                                    audio_out
                                }
                            }
                        } else {
                            audio_out
                        };

                        let tts_duration = tts_start.elapsed();
                        let _ = metrics.send(PipelineMetrics::new("tts".to_string(), tts_duration)).await;

                        let total_duration = pipeline_start.elapsed();
                        let _ = metrics.send(PipelineMetrics::new("total".to_string(), total_duration)).await;

                        tracing::info!(
                            "Pipeline complete: {}ms (STT={}ms, Translate={}ms, TTS={}ms)",
                            total_duration.as_millis(),
                            stt_duration.as_millis(),
                            translate_duration.as_millis(),
                            tts_duration.as_millis()
                        );

                        let _ = output_tx.send(audio_out).await;
                    });
                }
                else => break,
            }
        }

        tracing::info!("Pipeline loop ended");
    }
}

fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_of_squares: f32 = samples.iter().map(|s| s * s).sum();
    (sum_of_squares / samples.len() as f32).sqrt()
}
