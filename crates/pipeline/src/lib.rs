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
const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Flush accumulated audio every N seconds during continuous speech
const STREAMING_FLUSH_SECONDS: f32 = 1.0;

/// Minimum accumulated audio worth sending to STT (seconds)
const MIN_FLUSH_SECONDS: f32 = 0.3;

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
        let mut accumulated_samples: Vec<f32> = Vec::new();
        let mut is_speaking = false;

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
                            accumulated_samples.clear();
                            is_speaking = false;
                        }
                    }
                }
                Some(chunk) = audio_input.recv() => {
                    if !is_running {
                        continue;
                    }

                    let has_speech = vad.contains_speech(&chunk.samples);
                    let accumulated_seconds = accumulated_samples.len() as f32 / WHISPER_SAMPLE_RATE as f32;

                    if has_speech {
                        accumulated_samples.extend_from_slice(&chunk.samples);
                        let accumulated_seconds = accumulated_samples.len() as f32 / WHISPER_SAMPLE_RATE as f32;

                        if !is_speaking {
                            is_speaking = true;
                            tracing::info!("Speech started");
                        }

                        // Streaming flush: send chunks every N seconds while speaking
                        if accumulated_seconds >= STREAMING_FLUSH_SECONDS {
                            tracing::info!(
                                "Streaming flush: {:.1}s of audio",
                                accumulated_seconds
                            );
                            let samples_to_process = std::mem::take(&mut accumulated_samples);
                            spawn_pipeline_task(
                                samples_to_process,
                                stt.clone(),
                                translator.clone(),
                                tts.clone(),
                                audio_output.clone(),
                                metrics_tx.clone(),
                            );
                        }
                    } else if is_speaking {
                        // Speech paused — flush immediately
                        is_speaking = false;

                        if accumulated_seconds >= MIN_FLUSH_SECONDS {
                            tracing::info!(
                                "Speech paused — flushing {:.1}s of audio",
                                accumulated_seconds
                            );
                            let samples_to_process = std::mem::take(&mut accumulated_samples);
                            spawn_pipeline_task(
                                samples_to_process,
                                stt.clone(),
                                translator.clone(),
                                tts.clone(),
                                audio_output.clone(),
                                metrics_tx.clone(),
                            );
                        }
                    }
                }
                else => break,
            }
        }

        tracing::info!("Pipeline loop ended");
    }
}

fn spawn_pipeline_task(
    samples: Vec<f32>,
    stt: Arc<WhisperStt>,
    translator: Arc<OpusMtTranslator>,
    tts: Arc<PiperTts>,
    output_tx: mpsc::Sender<AudioChunk>,
    metrics_tx: mpsc::Sender<PipelineMetrics>,
) {
    tokio::spawn(async move {
        let pipeline_start = Instant::now();
        let chunk = AudioChunk::new(samples, WHISPER_SAMPLE_RATE, 1);

        // STT
        let stt_start = Instant::now();
        let text_segment = match stt.transcribe(&chunk) {
            Ok(segment) => segment,
            Err(e) => {
                tracing::warn!("STT failed: {}", e);
                return;
            }
        };
        let stt_duration = stt_start.elapsed();
        let _ = metrics_tx
            .send(PipelineMetrics::new("stt".to_string(), stt_duration))
            .await;

        if text_segment.is_empty() {
            return;
        }

        tracing::info!("STT: \"{}\"", text_segment.text);

        // Translation
        let translate_start = Instant::now();
        let translated = match translator.translate(&text_segment) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!("Translation failed: {}", e);
                return;
            }
        };
        let translate_duration = translate_start.elapsed();
        let _ = metrics_tx
            .send(PipelineMetrics::new(
                "translation".to_string(),
                translate_duration,
            ))
            .await;

        tracing::info!("Translated: \"{}\"", translated.text);

        // TTS
        let tts_start = Instant::now();
        let audio_out = match tts.synthesize(&translated) {
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
        let _ = metrics_tx
            .send(PipelineMetrics::new("tts".to_string(), tts_duration))
            .await;

        let total_duration = pipeline_start.elapsed();
        let _ = metrics_tx
            .send(PipelineMetrics::new("total".to_string(), total_duration))
            .await;

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
