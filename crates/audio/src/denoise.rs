/// Real-time audio denoising using RNNoise (nnnoiseless crate).
///
/// Removes background noise, music, and non-speech sounds from audio
/// before it reaches the STT engine. This significantly improves Whisper's
/// accuracy on audio with background music (e.g. YouTube documentaries).
///
/// RNNoise operates at 48kHz mono with fixed 480-sample frames (~10ms).
/// Processing cost is <1ms per frame — negligible latency.
use nnnoiseless::DenoiseState;

const RNNOISE_FRAME_SIZE: usize = DenoiseState::FRAME_SIZE; // 480 samples

/// Denoise a buffer of 48kHz mono f32 audio in-place.
/// Samples that don't fill a complete frame are left untouched.
pub fn denoise_48khz_mono(samples: &mut [f32]) {
    let mut state = DenoiseState::new();
    let mut input_frame = [0.0f32; RNNOISE_FRAME_SIZE];
    let mut output_frame = [0.0f32; RNNOISE_FRAME_SIZE];

    let full_frames = samples.len() / RNNOISE_FRAME_SIZE;
    for i in 0..full_frames {
        let start = i * RNNOISE_FRAME_SIZE;
        let end = start + RNNOISE_FRAME_SIZE;

        // RNNoise expects f32 samples scaled to [-32768, 32767]
        for (j, s) in samples[start..end].iter().enumerate() {
            input_frame[j] = s * 32767.0;
        }

        state.process_frame(&mut output_frame, &input_frame);

        // Scale back to [-1.0, 1.0]
        for (j, s) in output_frame.iter().enumerate() {
            samples[start + j] = s / 32767.0;
        }
    }
}

/// Downmix interleaved stereo to mono.
pub fn stereo_to_mono(interleaved: &[f32]) -> Vec<f32> {
    interleaved
        .chunks_exact(2)
        .map(|pair| (pair[0] + pair[1]) * 0.5)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denoise_silence_stays_silent() {
        let mut samples = vec![0.0f32; 480 * 10];
        denoise_48khz_mono(&mut samples);
        let rms: f32 = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        assert!(rms < 0.01, "RMS after denoising silence should be near zero, got {}", rms);
    }

    #[test]
    fn stereo_to_mono_averages_channels() {
        let stereo = vec![1.0f32, 0.0, 0.0, 1.0, 0.5, 0.5];
        let mono = stereo_to_mono(&stereo);
        assert_eq!(mono, vec![0.5, 0.5, 0.5]);
    }
}
