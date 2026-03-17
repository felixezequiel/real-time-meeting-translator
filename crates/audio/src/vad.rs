/// Energy-based Voice Activity Detection.
/// Uses RMS (Root Mean Square) threshold to detect speech vs silence.
/// Simple and zero-dependency — upgrade path to Silero VAD via ONNX later.

const DEFAULT_RMS_THRESHOLD: f32 = 0.00005;
const DEFAULT_MIN_SPEECH_DURATION_MS: u64 = 250;
const SAMPLE_RATE_16KHZ: f32 = 16_000.0;

pub struct EnergyVad {
    rms_threshold: f32,
    min_speech_samples: usize,
}

impl EnergyVad {
    pub fn new(rms_threshold: f32, min_speech_duration_ms: u64) -> Self {
        let min_speech_samples =
            (SAMPLE_RATE_16KHZ * min_speech_duration_ms as f32 / 1000.0) as usize;
        Self {
            rms_threshold,
            min_speech_samples,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_RMS_THRESHOLD, DEFAULT_MIN_SPEECH_DURATION_MS)
    }

    pub fn contains_speech(&self, samples: &[f32]) -> bool {
        if samples.len() < self.min_speech_samples {
            return false;
        }

        let rms = calculate_rms(samples);
        rms > self.rms_threshold
    }

    pub fn speech_ratio(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let window_size = self.min_speech_samples.max(1);
        let mut speech_windows = 0;
        let mut total_windows = 0;

        for window in samples.chunks(window_size) {
            total_windows += 1;
            let rms = calculate_rms(window);
            if rms > self.rms_threshold {
                speech_windows += 1;
            }
        }

        speech_windows as f32 / total_windows as f32
    }
}

fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_of_squares: f32 = samples.iter().map(|s| s * s).sum();
    (sum_of_squares / samples.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_of_silence_is_zero() {
        let silence = vec![0.0f32; 16_000];
        assert_eq!(calculate_rms(&silence), 0.0);
    }

    #[test]
    fn rms_of_constant_signal_equals_absolute_value() {
        let signal = vec![0.5f32; 1000];
        let rms = calculate_rms(&signal);
        assert!((rms - 0.5).abs() < 0.001);
    }

    #[test]
    fn rms_of_empty_is_zero() {
        assert_eq!(calculate_rms(&[]), 0.0);
    }

    #[test]
    fn vad_detects_silence_correctly() {
        let vad = EnergyVad::with_defaults();
        let silence = vec![0.0f32; 16_000];
        assert!(!vad.contains_speech(&silence));
    }

    #[test]
    fn vad_detects_loud_signal_as_speech() {
        let vad = EnergyVad::with_defaults();
        let loud_signal = vec![0.5f32; 16_000];
        assert!(vad.contains_speech(&loud_signal));
    }

    #[test]
    fn vad_rejects_short_signal_below_min_duration() {
        let vad = EnergyVad::new(0.01, 1000);
        let short_loud = vec![0.5f32; 100];
        assert!(!vad.contains_speech(&short_loud));
    }

    #[test]
    fn speech_ratio_is_zero_for_silence() {
        let vad = EnergyVad::with_defaults();
        let silence = vec![0.0f32; 16_000];
        assert_eq!(vad.speech_ratio(&silence), 0.0);
    }

    #[test]
    fn speech_ratio_is_one_for_constant_speech() {
        let vad = EnergyVad::with_defaults();
        let speech = vec![0.5f32; 16_000];
        assert_eq!(vad.speech_ratio(&speech), 1.0);
    }

    #[test]
    fn speech_ratio_is_zero_for_empty() {
        let vad = EnergyVad::with_defaults();
        assert_eq!(vad.speech_ratio(&[]), 0.0);
    }
}
