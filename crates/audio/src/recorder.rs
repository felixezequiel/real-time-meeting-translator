use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{Device, SampleFormat};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use thiserror::Error;
use tracing;

use crate::resampler;

/// OpenVoice TCC's reference speaker-embedding extractor wants 16 kHz
/// mono WAVs. We resample at capture time so the saved WAV is the
/// canonical format the bridge expects — no transcoding needed later.
const TARGET_SAMPLE_RATE: u32 = 16_000;

const MONO_CHANNELS: u16 = 1;

/// Target peak amplitude after normalisation (≈ -0.5 dBFS). Keeps a
/// touch of headroom so any post-processing rounding doesn't clip.
const NORMALIZE_TARGET_PEAK: f32 = 0.95;

/// Minimum peak required for normalisation. Below this the recording
/// is so quiet that scaling to 0.95 would amplify the noise floor 50×
/// or more — better to skip normalisation and let the user re-record
/// with a louder mic.
const MIN_PEAK_FOR_NORMALIZATION: f32 = 0.005;

/// Cap on the gain factor applied during normalisation. With many
/// laptop mics + headsets the speech peak is around 0.05–0.10, which
/// would call for 9.5–19× gain. We cap at 12× — beyond that the noise
/// floor becomes audible and the SE extractor sees that noise as
/// signal. Better to under-normalise than to inflate hum.
const MAX_NORMALIZATION_GAIN: f32 = 12.0;

/// Pre-amplification applied during capture. Same rationale as the
/// `INPUT_GAIN` in `capture.rs` — USB headsets ship with a fixed,
/// fairly conservative preamp gain. Boosting at the source means the
/// peak meter in the UI matches what the user feels they're saying,
/// and the post-recording normalisation needs much less gain (so it
/// doesn't have to amplify room hum to reach -1 dBFS).
const CAPTURE_GAIN: f32 = 4.0;

/// Atomic-friendly storage for the most-recent RMS reading. We pack
/// f32 bits into a u64 so the UI can poll without locking the buffer
/// mutex on every frame.
fn pack_f32(value: f32) -> u64 {
    value.to_bits() as u64
}
fn unpack_f32(bits: u64) -> f32 {
    f32::from_bits(bits as u32)
}

#[derive(Debug, Error)]
pub enum RecorderError {
    #[error("Failed to get device config: {0}")]
    ConfigError(String),

    #[error("Failed to build input stream: {0}")]
    StreamBuildError(String),

    #[error("Failed to start stream: {0}")]
    StreamStartError(String),

    #[error("Failed to write WAV: {0}")]
    WavWriteError(String),

    #[error("Recorder already finalized")]
    AlreadyFinalized,
}

/// Captures audio from a specific input device and accumulates the
/// samples in memory, resampled to 16 kHz mono. Calling `stop_and_save`
/// flushes the buffer to a WAV file at the requested path.
///
/// Designed for short, user-driven recordings (~30 s of voice profile
/// enrolment). Not appropriate for streaming pipelines — the buffer
/// grows unbounded until `stop_and_save` is called.
pub struct VoiceRecorder {
    _stream: cpal::Stream,
    samples: Arc<Mutex<Vec<f32>>>,
    rms_bits: Arc<AtomicU64>,
    /// Capture-side sample rate (kept for the single resample pass at
    /// stop time). The cpal callback delivers buffers in chunks far
    /// smaller than rubato's `FftFixedIn` 1024-sample minimum, so per
    /// callback resampling silently dropped every chunk on the floor.
    /// Storing at the source rate and resampling once at finalization
    /// time avoids the issue entirely.
    source_rate: u32,
    finalized: bool,
}

impl VoiceRecorder {
    /// Begin capturing from `device`. The recorder runs until
    /// `stop_and_save` is called or the recorder is dropped.
    pub fn start(device: &Device) -> Result<Self, RecorderError> {
        let device_name = device
            .name()
            .unwrap_or_else(|_| "(unnamed)".to_string());
        let config = device
            .default_input_config()
            .map_err(|e| RecorderError::ConfigError(format!(
                "Device \"{}\" has no default input config: {}",
                device_name, e,
            )))?;
        let source_rate = config.sample_rate().0;
        let source_channels = config.channels();
        let sample_format = config.sample_format();
        let stream_config: cpal::StreamConfig = config.clone().into();
        tracing::info!(
            "VoiceRecorder opening \"{}\" ({} Hz, {} ch, {:?})",
            device_name, source_rate, source_channels, sample_format,
        );

        let samples = Arc::new(Mutex::new(Vec::with_capacity(
            (source_rate as usize) * 30,
        )));
        let rms_bits = Arc::new(AtomicU64::new(pack_f32(0.0)));

        let samples_cb = Arc::clone(&samples);
        let rms_cb = Arc::clone(&rms_bits);

        let err_fn = |e| tracing::warn!("Voice recorder stream error: {}", e);

        let stream = match sample_format {
            SampleFormat::F32 => device.build_input_stream(
                &stream_config,
                move |data: &[f32], _| {
                    push_chunk(
                        data,
                        source_channels,
                        &samples_cb,
                        &rms_cb,
                    );
                },
                err_fn,
                None,
            ),
            SampleFormat::I16 => device.build_input_stream(
                &stream_config,
                move |data: &[i16], _| {
                    let f32_data: Vec<f32> = data
                        .iter()
                        .map(|s| *s as f32 / i16::MAX as f32)
                        .collect();
                    push_chunk(
                        &f32_data,
                        source_channels,
                        &samples_cb,
                        &rms_cb,
                    );
                },
                err_fn,
                None,
            ),
            other => return Err(RecorderError::ConfigError(format!(
                "Unsupported sample format: {:?}",
                other
            ))),
        }
        .map_err(|e| RecorderError::StreamBuildError(e.to_string()))?;

        stream
            .play()
            .map_err(|e| RecorderError::StreamStartError(e.to_string()))?;

        tracing::info!(
            "VoiceRecorder started ({} Hz, {} ch, {:?})",
            source_rate, source_channels, sample_format,
        );

        Ok(Self {
            _stream: stream,
            samples,
            rms_bits,
            source_rate,
            finalized: false,
        })
    }

    /// Current short-window RMS of the input. UI polls this each frame
    /// to draw a level meter.
    pub fn current_rms(&self) -> f32 {
        unpack_f32(self.rms_bits.load(Ordering::Relaxed))
    }

    /// Number of mono samples captured so far at the *source* rate.
    pub fn sample_count(&self) -> usize {
        self.samples.lock().map(|s| s.len()).unwrap_or(0)
    }

    /// Duration of audio captured so far, in seconds.
    pub fn duration_seconds(&self) -> f32 {
        self.sample_count() as f32 / self.source_rate.max(1) as f32
    }

    /// Stop the stream and write the accumulated samples as a 16-bit
    /// PCM mono WAV at `output_path`. The recorder is consumed.
    pub fn stop_and_save(mut self, output_path: &Path) -> Result<PathBuf, RecorderError> {
        if self.finalized {
            return Err(RecorderError::AlreadyFinalized);
        }
        self.finalized = true;

        let samples = match self.samples.lock() {
            Ok(g) => g.clone(),
            Err(e) => {
                return Err(RecorderError::WavWriteError(format!(
                    "Lock poisoned: {}",
                    e
                )))
            }
        };

        // Empty buffer means the cpal callback never fired (wrong
        // device, muted hardware, or driver issue). Refuse to write
        // a 0-byte WAV — that would persist a useless reference and
        // poison the OpenVoice SE extractor on first use.
        if samples.is_empty() {
            return Err(RecorderError::WavWriteError(
                "Nenhum áudio capturado — verifique o microfone selecionado."
                    .to_string(),
            ));
        }

        // Single resampling pass at finalization time. rubato's FFT
        // resampler needs ≥1024 samples per call, which the per-
        // callback chunks never provided — doing it here once over
        // the entire buffer side-steps that constraint.
        let mut samples = if self.source_rate != TARGET_SAMPLE_RATE {
            match resampler::resample_to_target(
                &samples,
                self.source_rate,
                TARGET_SAMPLE_RATE,
            ) {
                Ok(out) if !out.is_empty() => out,
                Ok(_) => {
                    return Err(RecorderError::WavWriteError(format!(
                        "Resample yielded zero samples (source_rate={}, captured={} samples)",
                        self.source_rate,
                        samples.len(),
                    )));
                }
                Err(e) => {
                    return Err(RecorderError::WavWriteError(format!(
                        "Resample failed: {}",
                        e
                    )));
                }
            }
        } else {
            samples
        };

        // Peak-normalise so quiet headsets / low-gain mics don't
        // produce a reference WAV at -25 dBFS (which the OpenVoice
        // SE extractor handles, but only barely — quieter inputs
        // give noisier embeddings). Cap the gain to MAX_NORMALIZATION_GAIN
        // so we never inflate the noise floor more than the signal.
        let peak = samples
            .iter()
            .fold(0.0_f32, |acc, &v| acc.max(v.abs()));
        if peak >= MIN_PEAK_FOR_NORMALIZATION {
            let raw_gain = NORMALIZE_TARGET_PEAK / peak;
            let gain = raw_gain.min(MAX_NORMALIZATION_GAIN);
            if gain > 1.001 {
                tracing::info!(
                    "VoiceRecorder normalising: peak={:.3} → gain×{:.2} (capped at {:.0}×)",
                    peak,
                    gain,
                    MAX_NORMALIZATION_GAIN,
                );
                for s in samples.iter_mut() {
                    *s = (*s * gain).clamp(-1.0, 1.0);
                }
            }
        } else {
            tracing::warn!(
                "VoiceRecorder skipping normalisation: peak={:.5} below threshold. \
                 The mic gain in Windows may be too low — open Sound settings → \
                 Input properties → Levels.",
                peak,
            );
        }

        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                RecorderError::WavWriteError(format!(
                    "Cannot create parent directory: {}",
                    e
                ))
            })?;
        }

        write_mono_pcm_wav(output_path, &samples, TARGET_SAMPLE_RATE)
            .map_err(|e| RecorderError::WavWriteError(e.to_string()))?;

        tracing::info!(
            "VoiceRecorder saved {} samples ({:.1} s) to {}",
            samples.len(),
            samples.len() as f32 / TARGET_SAMPLE_RATE as f32,
            output_path.display(),
        );
        Ok(output_path.to_path_buf())
    }
}

fn push_chunk(
    data: &[f32],
    source_channels: u16,
    samples: &Arc<Mutex<Vec<f32>>>,
    rms_bits: &Arc<AtomicU64>,
) {
    if data.is_empty() {
        return;
    }

    // Mono downmix + capture-time gain. Applying CAPTURE_GAIN here
    // means the level meter the user sees on screen reflects the
    // *boosted* signal (they get visual feedback that the headset is
    // delivering enough), and the saved buffer is closer to a usable
    // level before the final peak-normalize pass.
    let mono: Vec<f32> = if source_channels <= 1 {
        data.iter()
            .map(|s| (s * CAPTURE_GAIN).clamp(-1.0, 1.0))
            .collect()
    } else {
        data.chunks(source_channels as usize)
            .map(|frame| {
                let avg = frame.iter().sum::<f32>() / frame.len() as f32;
                (avg * CAPTURE_GAIN).clamp(-1.0, 1.0)
            })
            .collect()
    };

    // RMS for the level meter — computed on the source-rate mono signal
    // so the UI sees pre-resample energy.
    if !mono.is_empty() {
        let rms = (mono.iter().map(|s| s * s).sum::<f32>() / mono.len() as f32).sqrt();
        rms_bits.store(pack_f32(rms), Ordering::Relaxed);
    }

    // Store at the source rate. We resample once at `stop_and_save`
    // time — per-chunk resampling silently dropped every callback
    // because rubato's FftFixedIn requires ≥1024 input samples and
    // cpal callbacks deliver ~128–480 at a time.
    if let Ok(mut buf) = samples.lock() {
        buf.extend_from_slice(&mono);
    }
}

fn write_mono_pcm_wav(path: &Path, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
    let data_bytes = (samples.len() * 2) as u32;
    let chunk_size = 36 + data_bytes;
    let byte_rate = sample_rate * 2;

    let mut file = File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&chunk_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&MONO_CHANNELS.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_bytes.to_le_bytes())?;
    for &sample in samples {
        let clipped = sample.clamp(-1.0, 1.0);
        let int16 = (clipped * 32767.0) as i16;
        file.write_all(&int16.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        for v in [0.0_f32, 0.5, -0.5, 1.0, -1.0, 1e-6] {
            let bits = pack_f32(v);
            let back = unpack_f32(bits);
            assert!((v - back).abs() < 1e-6, "{} != {}", v, back);
        }
    }
}
