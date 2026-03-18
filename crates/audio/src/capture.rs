use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{Device, SampleFormat, StreamConfig};
use shared::AudioChunk;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing;

use crate::resampler;

const WHISPER_SAMPLE_RATE: u32 = 16_000;
const MONO_CHANNELS: u16 = 1;
const INPUT_GAIN: f32 = 15.0;

#[derive(Debug, Error)]
pub enum CaptureError {
    #[error("Failed to get device config: {0}")]
    ConfigError(String),

    #[error("Failed to build input stream: {0}")]
    StreamBuildError(String),

    #[error("Failed to start stream: {0}")]
    StreamStartError(String),

    #[error("Resampling failed: {0}")]
    ResampleError(#[from] crate::resampler::ResampleError),
}

pub struct AudioCapture {
    device: Device,
    chunk_duration_ms: u64,
}

impl AudioCapture {
    pub fn new(device: Device, chunk_duration_ms: u64) -> Self {
        Self {
            device,
            chunk_duration_ms,
        }
    }

    pub fn start(
        &self,
        sender: mpsc::UnboundedSender<AudioChunk>,
    ) -> Result<cpal::Stream, CaptureError> {
        let config = self
            .device
            .default_input_config()
            .map_err(|e| CaptureError::ConfigError(e.to_string()))?;

        let sample_rate = config.sample_rate().0;
        let channels = config.channels();
        let sample_format = config.sample_format();
        let samples_per_chunk =
            (sample_rate as u64 * channels as u64 * self.chunk_duration_ms / 1000) as usize;

        tracing::info!(
            "Capturing audio: {}Hz, {} channels, {:?}, chunk={}ms",
            sample_rate,
            channels,
            sample_format,
            self.chunk_duration_ms
        );

        let stream_config: StreamConfig = config.into();
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::with_capacity(
            samples_per_chunk,
        )));
        let buffer_clone = buffer.clone();

        let error_callback = |err: cpal::StreamError| {
            tracing::error!("Audio capture stream error: {}", err);
        };

        let stream = match sample_format {
            SampleFormat::F32 => {
                let data_callback = move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut buf = buffer_clone.lock().unwrap();
                    let amplified: Vec<f32> = data.iter().map(|&s| (s * INPUT_GAIN).clamp(-1.0, 1.0)).collect();
                    buf.extend_from_slice(&amplified);

                    while buf.len() >= samples_per_chunk {
                        let chunk_samples: Vec<f32> = buf.drain(..samples_per_chunk).collect();

                        let resampled = if sample_rate != WHISPER_SAMPLE_RATE || channels != MONO_CHANNELS {
                            resampler::resample_to_16khz_mono(&chunk_samples, sample_rate, channels)
                                .unwrap_or_default()
                        } else {
                            chunk_samples
                        };

                        let chunk = AudioChunk::new(resampled, WHISPER_SAMPLE_RATE, MONO_CHANNELS);
                        let _ = sender.send(chunk);
                    }
                };

                self.device
                    .build_input_stream(&stream_config, data_callback, error_callback, None)
                    .map_err(|e| CaptureError::StreamBuildError(e.to_string()))?
            }
            SampleFormat::I16 => {
                let data_callback = move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let float_data: Vec<f32> =
                        data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                    let mut buf = buffer_clone.lock().unwrap();
                    buf.extend_from_slice(&float_data);

                    while buf.len() >= samples_per_chunk {
                        let chunk_samples: Vec<f32> = buf.drain(..samples_per_chunk).collect();

                        let resampled = if sample_rate != WHISPER_SAMPLE_RATE || channels != MONO_CHANNELS {
                            resampler::resample_to_16khz_mono(&chunk_samples, sample_rate, channels)
                                .unwrap_or_default()
                        } else {
                            chunk_samples
                        };

                        let chunk = AudioChunk::new(resampled, WHISPER_SAMPLE_RATE, MONO_CHANNELS);
                        let _ = sender.send(chunk);
                    }
                };

                self.device
                    .build_input_stream(&stream_config, data_callback, error_callback, None)
                    .map_err(|e| CaptureError::StreamBuildError(e.to_string()))?
            }
            _ => {
                return Err(CaptureError::ConfigError(format!(
                    "Unsupported sample format: {:?}",
                    sample_format
                )));
            }
        };

        stream
            .play()
            .map_err(|e| CaptureError::StreamStartError(e.to_string()))?;

        Ok(stream)
    }
}
