/// WASAPI Loopback capture.
///
/// On Windows, cpal's WASAPI backend automatically enables `AUDCLNT_STREAMFLAGS_LOOPBACK`
/// when an input stream is built from an **output** device. This lets us capture exactly
/// what is being played through a speaker or headphone device — no virtual cable needed.
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{Device, SampleFormat, StreamConfig};
use shared::AudioChunk;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing;

use crate::resampler;

/// Type alias: unbounded sender never drops audio.
pub type AudioSender = mpsc::UnboundedSender<AudioChunk>;

const WHISPER_SAMPLE_RATE: u32 = 16_000;
const MONO_CHANNELS: u16 = 1;

#[derive(Debug, Error)]
pub enum LoopbackError {
    #[error("Failed to get output device config: {0}")]
    ConfigError(String),

    #[error("Failed to build loopback stream: {0}")]
    StreamBuildError(String),

    #[error("Failed to start loopback stream: {0}")]
    StreamStartError(String),
}

/// Captures system audio from an output device via WASAPI loopback.
pub struct LoopbackCapture {
    device: Device,
    chunk_duration_ms: u64,
}

impl LoopbackCapture {
    pub fn new(device: Device, chunk_duration_ms: u64) -> Self {
        Self {
            device,
            chunk_duration_ms,
        }
    }

    pub fn start(&self, sender: mpsc::UnboundedSender<AudioChunk>) -> Result<cpal::Stream, LoopbackError> {
        // Use the OUTPUT config — cpal WASAPI backend uses AUDCLNT_STREAMFLAGS_LOOPBACK
        // when an input stream is built from an output device.
        let config = self
            .device
            .default_output_config()
            .map_err(|e| LoopbackError::ConfigError(e.to_string()))?;

        let sample_rate = config.sample_rate().0;
        let channels = config.channels();
        let sample_format = config.sample_format();
        let samples_per_chunk =
            (sample_rate as u64 * channels as u64 * self.chunk_duration_ms / 1000) as usize;

        tracing::info!(
            "Loopback capture: {}Hz, {} ch, {:?}, chunk={}ms",
            sample_rate,
            channels,
            sample_format,
            self.chunk_duration_ms
        );

        let stream_config: StreamConfig = config.into();
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f32>::with_capacity(
            samples_per_chunk * 2,
        )));
        let buffer_clone = buffer.clone();

        let error_callback = |err: cpal::StreamError| {
            tracing::error!("Loopback stream error: {}", err);
        };

        let stream = match sample_format {
            SampleFormat::F32 => {
                let data_callback = move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut buf = buffer_clone.lock().unwrap();
                    buf.extend_from_slice(data);
                    flush_chunks(&mut buf, samples_per_chunk, sample_rate, channels, &sender);
                };
                self.device
                    .build_input_stream(&stream_config, data_callback, error_callback, None)
                    .map_err(|e| LoopbackError::StreamBuildError(e.to_string()))?
            }
            SampleFormat::I16 => {
                let data_callback = move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let float_samples: Vec<f32> =
                        data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                    let mut buf = buffer_clone.lock().unwrap();
                    buf.extend_from_slice(&float_samples);
                    flush_chunks(&mut buf, samples_per_chunk, sample_rate, channels, &sender);
                };
                self.device
                    .build_input_stream(&stream_config, data_callback, error_callback, None)
                    .map_err(|e| LoopbackError::StreamBuildError(e.to_string()))?
            }
            _ => {
                return Err(LoopbackError::ConfigError(format!(
                    "Unsupported sample format for loopback: {:?}",
                    sample_format
                )));
            }
        };

        stream
            .play()
            .map_err(|e| LoopbackError::StreamStartError(e.to_string()))?;

        Ok(stream)
    }
}

/// Drains complete chunks from the buffer, resamples to 16 kHz mono, and sends them.
fn flush_chunks(
    buf: &mut Vec<f32>,
    samples_per_chunk: usize,
    sample_rate: u32,
    channels: u16,
    sender: &mpsc::UnboundedSender<AudioChunk>,
) {
    while buf.len() >= samples_per_chunk {
        let chunk_samples: Vec<f32> = buf.drain(..samples_per_chunk).collect();

        let resampled = if sample_rate != WHISPER_SAMPLE_RATE || channels != MONO_CHANNELS {
            resampler::resample_to_16khz_mono(&chunk_samples, sample_rate, channels)
                .unwrap_or_default()
        } else {
            chunk_samples
        };

        let chunk = AudioChunk::new(resampled, WHISPER_SAMPLE_RATE, MONO_CHANNELS);
        // Unbounded send — NEVER drops audio
        let _ = sender.send(chunk);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flush_chunks_does_nothing_when_buffer_is_too_small() {
        let (tx, _rx) = mpsc::channel(8);
        let mut buf = vec![0.0f32; 100];
        let samples_per_chunk = 1000;
        flush_chunks(&mut buf, samples_per_chunk, 16_000, 1, &tx);
        assert_eq!(buf.len(), 100); // untouched
    }

    #[test]
    fn flush_chunks_drains_complete_chunk() {
        let (tx, mut rx) = mpsc::channel(8);
        let samples_per_chunk = 400; // 400 samples at 16kHz mono = 25ms
        let mut buf = vec![0.0f32; 400];
        flush_chunks(&mut buf, samples_per_chunk, 16_000, 1, &tx);
        assert!(buf.is_empty());
        assert!(rx.try_recv().is_ok());
    }
}
