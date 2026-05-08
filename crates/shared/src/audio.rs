use serde::{Deserialize, Serialize};

const WHISPER_EXPECTED_SAMPLE_RATE: u32 = 16_000;
const MONO_CHANNEL_COUNT: u16 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    /// True when this chunk is one of several mid-phrase fragments of a
    /// streaming TTS utterance (XTTS-v2 `inference_stream`). The mixer
    /// uses it to suppress the per-chunk fade-in/fade-out envelope: a
    /// 12 ms fade applied at the boundary of a 250 ms mid-sentence
    /// fragment chops the waveform between adjacent chunks of the SAME
    /// word and is audible as clicks/gaps. Atomic chunks (Kokoro,
    /// passthrough, mic capture) keep the default `false` and continue
    /// to receive the smoothing envelope.
    #[serde(default, skip_serializing_if = "is_false")]
    pub is_streaming_chunk: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl AudioChunk {
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
            is_streaming_chunk: false,
        }
    }

    /// Constructor for mid-phrase fragments emitted by streaming TTS.
    /// See `is_streaming_chunk` for why these need different mixer
    /// treatment than atomic chunks.
    pub fn streaming(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
            is_streaming_chunk: true,
        }
    }

    pub fn duration_seconds(&self) -> f32 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        self.samples.len() as f32 / (self.sample_rate as f32 * self.channels as f32)
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    pub fn is_whisper_compatible(&self) -> bool {
        self.sample_rate == WHISPER_EXPECTED_SAMPLE_RATE && self.channels == MONO_CHANNEL_COUNT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_calculated_correctly_for_mono_16khz() {
        let one_second_of_samples = vec![0.0f32; 16_000];
        let chunk = AudioChunk::new(one_second_of_samples, 16_000, 1);

        assert_eq!(chunk.duration_seconds(), 1.0);
    }

    #[test]
    fn duration_calculated_correctly_for_stereo_48khz() {
        let two_seconds_stereo = vec![0.0f32; 48_000 * 2 * 2];
        let chunk = AudioChunk::new(two_seconds_stereo, 48_000, 2);

        assert_eq!(chunk.duration_seconds(), 2.0);
    }

    #[test]
    fn duration_returns_zero_for_empty_chunk() {
        let chunk = AudioChunk::new(vec![], 16_000, 1);
        assert_eq!(chunk.duration_seconds(), 0.0);
    }

    #[test]
    fn duration_returns_zero_when_sample_rate_is_zero() {
        let chunk = AudioChunk::new(vec![0.0; 100], 0, 1);
        assert_eq!(chunk.duration_seconds(), 0.0);
    }

    #[test]
    fn is_empty_returns_true_when_no_samples() {
        let chunk = AudioChunk::new(vec![], 16_000, 1);
        assert!(chunk.is_empty());
    }

    #[test]
    fn is_empty_returns_false_when_has_samples() {
        let chunk = AudioChunk::new(vec![0.1], 16_000, 1);
        assert!(!chunk.is_empty());
    }

    #[test]
    fn is_whisper_compatible_true_for_16khz_mono() {
        let chunk = AudioChunk::new(vec![0.0; 16_000], 16_000, 1);
        assert!(chunk.is_whisper_compatible());
    }

    #[test]
    fn is_whisper_compatible_false_for_48khz() {
        let chunk = AudioChunk::new(vec![0.0; 48_000], 48_000, 1);
        assert!(!chunk.is_whisper_compatible());
    }

    #[test]
    fn is_whisper_compatible_false_for_stereo() {
        let chunk = AudioChunk::new(vec![0.0; 32_000], 16_000, 2);
        assert!(!chunk.is_whisper_compatible());
    }

    #[test]
    fn new_chunk_defaults_to_atomic_not_streaming() {
        let chunk = AudioChunk::new(vec![0.1; 16], 16_000, 1);
        assert!(!chunk.is_streaming_chunk);
    }

    #[test]
    fn streaming_chunk_marks_is_streaming_chunk_true() {
        let chunk = AudioChunk::streaming(vec![0.1; 16], 24_000, 1);
        assert!(chunk.is_streaming_chunk);
    }
}
