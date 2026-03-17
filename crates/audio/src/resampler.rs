use rubato::{FftFixedIn, Resampler};
use thiserror::Error;

const WHISPER_SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Error)]
pub enum ResampleError {
    #[error("Resampler creation failed: {0}")]
    CreationFailed(String),

    #[error("Resampling failed: {0}")]
    ProcessingFailed(String),
}

pub fn resample_to_16khz_mono(
    samples: &[f32],
    source_sample_rate: u32,
    source_channels: u16,
) -> Result<Vec<f32>, ResampleError> {
    let mono_samples = if source_channels > 1 {
        downmix_to_mono(samples, source_channels)
    } else {
        samples.to_vec()
    };

    if source_sample_rate == WHISPER_SAMPLE_RATE {
        return Ok(mono_samples);
    }

    let chunk_size = 1024;
    let mut resampler = FftFixedIn::<f32>::new(
        source_sample_rate as usize,
        WHISPER_SAMPLE_RATE as usize,
        chunk_size,
        2,
        1,
    )
    .map_err(|e| ResampleError::CreationFailed(e.to_string()))?;

    let mut output = Vec::new();
    let mut position = 0;
    let frames_needed = resampler.input_frames_next();

    while position + frames_needed <= mono_samples.len() {
        let input_chunk = &mono_samples[position..position + frames_needed];
        let result = resampler
            .process(&[input_chunk.to_vec()], None)
            .map_err(|e| ResampleError::ProcessingFailed(e.to_string()))?;
        if let Some(channel) = result.first() {
            output.extend_from_slice(channel);
        }
        position += frames_needed;
    }

    Ok(output)
}

pub fn resample_to_target(
    samples: &[f32],
    source_rate: u32,
    target_rate: u32,
) -> Result<Vec<f32>, ResampleError> {
    if source_rate == target_rate {
        return Ok(samples.to_vec());
    }

    let chunk_size = 1024;
    let mut resampler = FftFixedIn::<f32>::new(
        source_rate as usize,
        target_rate as usize,
        chunk_size,
        2,
        1,
    )
    .map_err(|e| ResampleError::CreationFailed(e.to_string()))?;

    let mut output = Vec::new();
    let mut position = 0;
    let frames_needed = resampler.input_frames_next();

    while position + frames_needed <= samples.len() {
        let input_chunk = &samples[position..position + frames_needed];
        let result = resampler
            .process(&[input_chunk.to_vec()], None)
            .map_err(|e| ResampleError::ProcessingFailed(e.to_string()))?;
        if let Some(channel) = result.first() {
            output.extend_from_slice(channel);
        }
        position += frames_needed;
    }

    Ok(output)
}

fn downmix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    let channel_count = channels as usize;
    samples
        .chunks(channel_count)
        .map(|frame| frame.iter().sum::<f32>() / channel_count as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_stereo_to_mono_averages_channels() {
        let stereo = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0];
        let mono = downmix_to_mono(&stereo, 2);
        assert_eq!(mono.len(), 3);
        assert_eq!(mono[0], 0.5);
        assert_eq!(mono[1], 0.5);
        assert_eq!(mono[2], 0.5);
    }

    #[test]
    fn resample_16khz_mono_returns_same_data() {
        let samples = vec![0.1, 0.2, 0.3, 0.4];
        let result = resample_to_16khz_mono(&samples, 16_000, 1).unwrap();
        assert_eq!(result, samples);
    }

    #[test]
    fn resample_48khz_to_16khz_reduces_sample_count() {
        let sample_count = 48_000;
        let samples: Vec<f32> = (0..sample_count).map(|i| (i as f32 / 100.0).sin()).collect();
        let result = resample_to_16khz_mono(&samples, 48_000, 1).unwrap();
        let expected_approximate_count = sample_count / 3;
        let tolerance = expected_approximate_count / 10;
        assert!(
            (result.len() as i64 - expected_approximate_count as i64).unsigned_abs() < tolerance as u64,
            "Expected ~{} samples, got {}",
            expected_approximate_count,
            result.len()
        );
    }
}
