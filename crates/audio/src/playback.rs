use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{Device, SampleFormat, StreamConfig};
use shared::AudioChunk;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing;

#[derive(Debug, Error)]
pub enum PlaybackError {
    #[error("Failed to get device config: {0}")]
    ConfigError(String),

    #[error("Failed to build output stream: {0}")]
    StreamBuildError(String),

    #[error("Failed to start stream: {0}")]
    StreamStartError(String),
}

pub struct AudioPlayback {
    device: Device,
}

impl AudioPlayback {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    pub fn start(
        &self,
        mut receiver: mpsc::UnboundedReceiver<AudioChunk>,
    ) -> Result<cpal::Stream, PlaybackError> {
        let config = self
            .device
            .default_output_config()
            .map_err(|e| PlaybackError::ConfigError(e.to_string()))?;

        let sample_rate = config.sample_rate().0;
        let channels = config.channels();
        let sample_format = config.sample_format();

        tracing::info!(
            "Playing audio: {}Hz, {} channels, {:?}",
            sample_rate,
            channels,
            sample_format
        );

        let stream_config: StreamConfig = config.into();
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::<f32>::new()));
        let buffer_writer = buffer.clone();

        std::thread::spawn(move || {
            while let Some(chunk) = receiver.blocking_recv() {
                let mut buf = buffer_writer.lock().unwrap();
                // Duplicate mono to match output channels if needed
                if channels > 1 {
                    for &sample in &chunk.samples {
                        for _ in 0..channels {
                            buf.push_back(sample);
                        }
                    }
                } else {
                    buf.extend(&chunk.samples);
                }
            }
        });

        let error_callback = |err: cpal::StreamError| {
            tracing::error!("Audio playback stream error: {}", err);
        };

        let stream = match sample_format {
            SampleFormat::F32 => {
                let buffer_reader = buffer.clone();
                let data_callback = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let mut buf = buffer_reader.lock().unwrap();
                    for sample in data.iter_mut() {
                        *sample = buf.pop_front().unwrap_or(0.0);
                    }
                };
                self.device
                    .build_output_stream(&stream_config, data_callback, error_callback, None)
                    .map_err(|e| PlaybackError::StreamBuildError(e.to_string()))?
            }
            SampleFormat::I16 => {
                let buffer_reader = buffer.clone();
                let data_callback = move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                    let mut buf = buffer_reader.lock().unwrap();
                    for sample in data.iter_mut() {
                        let float_sample = buf.pop_front().unwrap_or(0.0);
                        *sample = (float_sample * i16::MAX as f32) as i16;
                    }
                };
                self.device
                    .build_output_stream(&stream_config, data_callback, error_callback, None)
                    .map_err(|e| PlaybackError::StreamBuildError(e.to_string()))?
            }
            _ => {
                return Err(PlaybackError::ConfigError(format!(
                    "Unsupported sample format: {:?}",
                    sample_format
                )));
            }
        };

        stream
            .play()
            .map_err(|e| PlaybackError::StreamStartError(e.to_string()))?;

        Ok(stream)
    }
}
