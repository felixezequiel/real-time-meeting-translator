use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{Device, SampleFormat, StreamConfig};
use shared::AudioChunk;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use thiserror::Error;
use tokio::sync::mpsc;
use tracing;

use crate::resampler;

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
    /// Signals externally that this playback has audio in its buffer.
    is_playing_flag: Option<Arc<AtomicBool>>,
}

impl AudioPlayback {
    pub fn new(device: Device) -> Self {
        Self { device, is_playing_flag: None }
    }

    /// Create a playback that sets a shared flag while audio is playing.
    pub fn with_playing_flag(device: Device, flag: Arc<AtomicBool>) -> Self {
        Self { device, is_playing_flag: Some(flag) }
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

        let is_playing = self.is_playing_flag.clone();

        let stream = match sample_format {
            SampleFormat::F32 => {
                let buffer_reader = buffer.clone();
                let flag = is_playing.clone();
                let data_callback = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let mut buf = buffer_reader.lock().unwrap();
                    let has_audio = !buf.is_empty();
                    for sample in data.iter_mut() {
                        *sample = buf.pop_front().unwrap_or(0.0);
                    }
                    if let Some(ref f) = flag {
                        f.store(has_audio, Ordering::Release);
                    }
                };
                self.device
                    .build_output_stream(&stream_config, data_callback, error_callback, None)
                    .map_err(|e| PlaybackError::StreamBuildError(e.to_string()))?
            }
            SampleFormat::I16 => {
                let buffer_reader = buffer.clone();
                let flag = is_playing.clone();
                let data_callback = move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                    let mut buf = buffer_reader.lock().unwrap();
                    let has_audio = !buf.is_empty();
                    for sample in data.iter_mut() {
                        let float_sample = buf.pop_front().unwrap_or(0.0);
                        *sample = (float_sample * i16::MAX as f32) as i16;
                    }
                    if let Some(ref f) = flag {
                        f.store(has_audio, Ordering::Release);
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

// ─── Mixer playback: passthrough + TTS with smooth gain ducking ──────────────

/// Passthrough gain while TTS is speaking. 0.35 keeps the original audio
/// clearly audible but unmistakably backgrounded — same mix a TV interpreter
/// produces when talking over the source speaker.
const PASSTHROUGH_DUCK_GAIN: f32 = 0.35;

/// Time for the passthrough gain to settle (~95%) to its new target when
/// TTS starts/stops. Slow enough to avoid pumping artifacts on short TTS
/// gaps, fast enough to not clip the start of a sentence.
const DUCK_SETTLE_SECONDS: f32 = 0.15;

/// Mixes a passthrough stream and a TTS stream into one output device,
/// applying smooth gain modulation to the passthrough while the TTS buffer
/// contains audio. Replaces the older WASAPI ducker — this acts only on our
/// own playback mix, never touches external application volumes.
///
/// Contract: both streams should arrive at the device's sample rate. When
/// they don't, the ingester resamples in 1024-frame chunks (Rubato FFT-based
/// resampler).
pub struct MixerPlayback {
    device: Device,
}

impl MixerPlayback {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    pub fn start(
        &self,
        passthrough_rx: mpsc::UnboundedReceiver<AudioChunk>,
        tts_rx: mpsc::UnboundedReceiver<AudioChunk>,
    ) -> Result<cpal::Stream, PlaybackError> {
        let config = self
            .device
            .default_output_config()
            .map_err(|e| PlaybackError::ConfigError(e.to_string()))?;

        let out_rate = config.sample_rate().0;
        let out_channels = config.channels();
        let sample_format = config.sample_format();

        tracing::info!(
            "Mixer playback: {}Hz, {} ch, {:?}",
            out_rate, out_channels, sample_format,
        );

        let stream_config: StreamConfig = config.into();

        let passthrough_buf: Arc<Mutex<VecDeque<f32>>> =
            Arc::new(Mutex::new(VecDeque::new()));
        let tts_buf: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(VecDeque::new()));

        spawn_ingester("passthrough", passthrough_rx, passthrough_buf.clone(), out_rate, out_channels);
        spawn_ingester("tts", tts_rx, tts_buf.clone(), out_rate, out_channels);

        let error_callback = |err: cpal::StreamError| {
            tracing::error!("Mixer playback stream error: {}", err);
        };

        // Per-frame smoothing coefficient. Solves (1-c)^n = 0.05 at n = rate*settle
        // (3τ ≈ 95 % of the way to target). Clamped in case of tiny sample rates.
        let ramp_coeff = (3.0 / (out_rate as f32 * DUCK_SETTLE_SECONDS)).min(1.0);
        let channels_usize = out_channels as usize;

        let stream = match sample_format {
            SampleFormat::F32 => {
                let pt_reader = passthrough_buf.clone();
                let tts_reader = tts_buf.clone();
                let mut current_gain: f32 = 1.0;
                let data_callback = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let mut pt = pt_reader.lock().unwrap();
                    let mut tts = tts_reader.lock().unwrap();
                    let tts_active = !tts.is_empty();
                    let target_gain = if tts_active { PASSTHROUGH_DUCK_GAIN } else { 1.0 };

                    let frames = data.len() / channels_usize;
                    for frame_idx in 0..frames {
                        current_gain += (target_gain - current_gain) * ramp_coeff;
                        let base = frame_idx * channels_usize;
                        for ch in 0..channels_usize {
                            let tts_sample = tts.pop_front().unwrap_or(0.0);
                            let pt_sample = pt.pop_front().unwrap_or(0.0);
                            let mixed = tts_sample + pt_sample * current_gain;
                            data[base + ch] = mixed.clamp(-1.0, 1.0);
                        }
                    }
                };
                self.device
                    .build_output_stream(&stream_config, data_callback, error_callback, None)
                    .map_err(|e| PlaybackError::StreamBuildError(e.to_string()))?
            }
            SampleFormat::I16 => {
                let pt_reader = passthrough_buf.clone();
                let tts_reader = tts_buf.clone();
                let mut current_gain: f32 = 1.0;
                let data_callback = move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                    let mut pt = pt_reader.lock().unwrap();
                    let mut tts = tts_reader.lock().unwrap();
                    let tts_active = !tts.is_empty();
                    let target_gain = if tts_active { PASSTHROUGH_DUCK_GAIN } else { 1.0 };

                    let frames = data.len() / channels_usize;
                    for frame_idx in 0..frames {
                        current_gain += (target_gain - current_gain) * ramp_coeff;
                        let base = frame_idx * channels_usize;
                        for ch in 0..channels_usize {
                            let tts_sample = tts.pop_front().unwrap_or(0.0);
                            let pt_sample = pt.pop_front().unwrap_or(0.0);
                            let mixed = (tts_sample + pt_sample * current_gain).clamp(-1.0, 1.0);
                            data[base + ch] = (mixed * i16::MAX as f32) as i16;
                        }
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

/// Reads chunks from `rx`, normalizes them to (out_rate, out_channels)
/// interleaved, and appends to the shared ring buffer.
fn spawn_ingester(
    name: &'static str,
    mut rx: mpsc::UnboundedReceiver<AudioChunk>,
    buffer: Arc<Mutex<VecDeque<f32>>>,
    out_rate: u32,
    out_channels: u16,
) {
    std::thread::Builder::new()
        .name(format!("mixer-ingest-{}", name))
        .spawn(move || {
            // Per-channel accumulator for the rate-adapt path (resampler needs ≥1024).
            let mut resample_accum: Vec<f32> = Vec::new();
            let mut last_input_rate: u32 = 0;

            while let Some(chunk) = rx.blocking_recv() {
                // Reset accumulator if the input rate changed (device hot-swap).
                if chunk.sample_rate != last_input_rate {
                    resample_accum.clear();
                    last_input_rate = chunk.sample_rate;
                }

                // Fast path: sample rate matches output — just adapt channels.
                if chunk.sample_rate == out_rate {
                    let adapted = adapt_channels(chunk.samples, chunk.channels, out_channels);
                    let mut buf = buffer.lock().unwrap();
                    buf.extend(adapted);
                    continue;
                }

                // Resample path: downmix to mono, accumulate, resample in ≥1024-sample
                // chunks (Rubato FFT requirement), then re-upmix to out_channels.
                let mono: Vec<f32> = if chunk.channels == 1 {
                    chunk.samples
                } else {
                    chunk
                        .samples
                        .chunks_exact(chunk.channels as usize)
                        .map(|f| f.iter().sum::<f32>() / chunk.channels as f32)
                        .collect()
                };
                resample_accum.extend_from_slice(&mono);

                while resample_accum.len() >= 1024 {
                    let input_chunk: Vec<f32> = resample_accum.drain(..1024).collect();
                    match resampler::resample_mono(&input_chunk, chunk.sample_rate, out_rate) {
                        Ok(resampled) => {
                            let adapted = adapt_channels(resampled, 1, out_channels);
                            let mut buf = buffer.lock().unwrap();
                            buf.extend(adapted);
                        }
                        Err(e) => {
                            tracing::warn!("Mixer '{}' resample failed: {}", name, e);
                        }
                    }
                }
            }
            tracing::debug!("Mixer ingester '{}' ended", name);
        })
        .expect("failed to spawn mixer ingester thread");
}

/// Interleaved channel adaptation. Mono → N duplicates per frame; N → mono
/// averages; M → N (M≠1, N≠1) downmixes to mono then upmixes to N.
fn adapt_channels(samples: Vec<f32>, in_channels: u16, out_channels: u16) -> Vec<f32> {
    if in_channels == out_channels {
        return samples;
    }
    if in_channels == 1 {
        let mut out = Vec::with_capacity(samples.len() * out_channels as usize);
        for s in samples {
            for _ in 0..out_channels {
                out.push(s);
            }
        }
        return out;
    }
    if out_channels == 1 {
        return samples
            .chunks_exact(in_channels as usize)
            .map(|f| f.iter().sum::<f32>() / in_channels as f32)
            .collect();
    }
    let mono = adapt_channels(samples, in_channels, 1);
    adapt_channels(mono, 1, out_channels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapt_channels_noop_when_matching() {
        let input = vec![0.1, 0.2, 0.3, 0.4];
        let out = adapt_channels(input.clone(), 2, 2);
        assert_eq!(out, input);
    }

    #[test]
    fn adapt_channels_mono_to_stereo_duplicates() {
        let input = vec![0.1, 0.2, 0.3];
        let out = adapt_channels(input, 1, 2);
        assert_eq!(out, vec![0.1, 0.1, 0.2, 0.2, 0.3, 0.3]);
    }

    #[test]
    fn adapt_channels_stereo_to_mono_averages() {
        let input = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0];
        let out = adapt_channels(input, 2, 1);
        assert_eq!(out, vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn adapt_channels_stereo_to_quad_duplicates_mono_mix() {
        let input = vec![1.0, 0.0, 0.0, 1.0];
        let out = adapt_channels(input, 2, 4);
        assert_eq!(out, vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn duck_gain_is_audible_but_backgrounded() {
        assert!(PASSTHROUGH_DUCK_GAIN > 0.0);
        assert!(PASSTHROUGH_DUCK_GAIN < 1.0);
    }

    #[test]
    fn ramp_settles_roughly_within_target_time() {
        // Emulate the per-frame smoothing used inside the callback and
        // confirm that 3τ is close to DUCK_SETTLE_SECONDS at 48 kHz.
        let out_rate: u32 = 48_000;
        let coeff = (3.0 / (out_rate as f32 * DUCK_SETTLE_SECONDS)).min(1.0);
        let target = 0.35_f32;
        let mut gain = 1.0_f32;
        let frames_target = (out_rate as f32 * DUCK_SETTLE_SECONDS) as usize;

        for _ in 0..frames_target {
            gain += (target - gain) * coeff;
        }
        // After DUCK_SETTLE_SECONDS frames we should be ≥95 % of the way there.
        let progress = (1.0 - gain) / (1.0 - target); // 1.0 means fully settled
        assert!(progress >= 0.9, "ramp progress = {} (expected ≥0.9)", progress);
    }
}
