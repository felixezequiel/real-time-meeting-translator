//! Silero VAD — neural voice activity detection.
//!
//! Replaces the energy-based RMS gate. Backed by the `voice_activity_detector`
//! crate, which embeds the Silero v5 ONNX model + ORT bindings. We tried
//! a hand-rolled ORT integration first; it produced degenerate
//! probabilities (~0.0006 even on clear speech) and the root cause was
//! buried somewhere between input shapes, scalar tensor handling, and
//! model state plumbing. The maintained crate is validated against real
//! audio in its own tests, so we let it own that complexity.
//!
//! See `docs/adr/0008-silero-vad.md`.

use std::path::Path;
use std::sync::Mutex;
use thiserror::Error;
use voice_activity_detector::VoiceActivityDetector;

/// Number of audio samples Silero v5 expects per inference call at 16 kHz.
/// The library accepts other sizes but we keep the contract explicit.
pub const FRAME_SIZE: usize = 512;

/// Sample rate the model was trained at. We resample upstream to this
/// before feeding the VAD.
pub const SAMPLE_RATE_HZ: i64 = 16_000;

/// Default speech-probability threshold. 0.5 is the value snakers4
/// recommends in their reference implementation; lower trades
/// false-positives for fewer missed soft-spoken segments.
pub const DEFAULT_THRESHOLD: f32 = 0.5;

#[derive(Debug, Error)]
pub enum SileroVadError {
    #[error("Failed to load Silero VAD model: {0}")]
    LoadFailed(String),

    #[error("Failed to run Silero VAD inference: {0}")]
    InferenceFailed(String),
}

/// Stateful Silero VAD. One instance per audio stream — the underlying
/// LSTM state must not be shared between mic and loopback contexts.
pub struct SileroVad {
    inner: Mutex<VoiceActivityDetector>,
    threshold: f32,
}

impl SileroVad {
    /// Load the model with the default threshold. The `model_path`
    /// argument is accepted for API stability with the previous ORT
    /// implementation but is ignored — the crate ships the v5 model
    /// embedded.
    pub fn from_path(_model_path: &Path) -> Result<Self, SileroVadError> {
        Self::with_threshold(DEFAULT_THRESHOLD)
    }

    /// Load the model with a custom threshold ∈ [0, 1].
    pub fn from_path_with_threshold(
        _model_path: &Path,
        threshold: f32,
    ) -> Result<Self, SileroVadError> {
        Self::with_threshold(threshold)
    }

    /// Build a `SileroVad` with the embedded model.
    pub fn with_threshold(threshold: f32) -> Result<Self, SileroVadError> {
        let detector = VoiceActivityDetector::builder()
            .sample_rate(SAMPLE_RATE_HZ as i64)
            .chunk_size(FRAME_SIZE)
            .build()
            .map_err(|e| SileroVadError::LoadFailed(e.to_string()))?;

        Ok(Self {
            inner: Mutex::new(detector),
            threshold,
        })
    }

    /// True when any frame in `samples` has speech probability above the
    /// threshold. Empty input returns false without running the model.
    /// Trailing samples shorter than `FRAME_SIZE` are zero-padded — the
    /// final partial frame at chunk boundaries still gets a fair shot.
    pub fn has_speech(&self, samples: &[f32]) -> bool {
        if samples.is_empty() {
            return false;
        }
        let threshold = self.threshold;
        let mut detector = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };

        for frame in samples.chunks(FRAME_SIZE) {
            if frame_speech_probability(&mut detector, frame) >= threshold {
                return true;
            }
        }
        false
    }

    /// Maximum speech probability across all frames in `samples`. Useful
    /// for diagnostics and the metrics panel; the gate uses `has_speech`.
    pub fn speech_probability(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let mut detector = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return 0.0,
        };

        let mut max_prob = 0.0_f32;
        for frame in samples.chunks(FRAME_SIZE) {
            let prob = frame_speech_probability(&mut detector, frame);
            if prob > max_prob {
                max_prob = prob;
            }
        }
        max_prob
    }

    /// Reset the LSTM state. Call when starting a new utterance / pipeline
    /// so leftover context from the previous one doesn't bias detection.
    pub fn reset_state(&self) {
        if let Ok(mut detector) = self.inner.lock() {
            detector.reset();
        }
    }

    /// Configured decision threshold. Exposed for diagnostics.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}

/// Run a single frame through the detector. Frames shorter than
/// `FRAME_SIZE` are zero-padded so the underlying crate doesn't reject
/// them — its `predict` requires exactly `chunk_size` samples.
fn frame_speech_probability(detector: &mut VoiceActivityDetector, frame: &[f32]) -> f32 {
    if frame.len() == FRAME_SIZE {
        return detector.predict(frame.iter().copied());
    }
    let mut padded = [0.0_f32; FRAME_SIZE];
    let copy_len = frame.len().min(FRAME_SIZE);
    padded[..copy_len].copy_from_slice(&frame[..copy_len]);
    detector.predict(padded.iter().copied())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// API stability: kept for the integration tests that pass a path.
    /// The `voice_activity_detector` crate ships the model embedded, so
    /// this is unused in practice.
    fn model_path() -> PathBuf {
        PathBuf::from("models/silero_vad.onnx")
    }

    fn synthetic_silence(num_samples: usize) -> Vec<f32> {
        vec![0.0; num_samples]
    }

    /// Modulated tone in the speech-pitch band. Silero is robust enough
    /// that even synthetic carriers register above the threshold when
    /// modulated like a human voice envelope.
    fn synthetic_speech_like(num_samples: usize) -> Vec<f32> {
        let carrier_hz = 200.0_f32;
        let modulation_hz = 4.0_f32;
        let sample_rate = SAMPLE_RATE_HZ as f32;
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                let carrier = (t * carrier_hz * 2.0 * std::f32::consts::PI).sin();
                let envelope = 0.5
                    + 0.5 * (t * modulation_hz * 2.0 * std::f32::consts::PI).sin();
                0.4 * carrier * envelope
            })
            .collect()
    }

    #[test]
    fn frame_size_matches_silero_v5_spec() {
        // Silero v5 only accepts 512-sample frames at 16 kHz. Anything
        // else makes the model reject the input. This test guards us
        // from accidental reflows.
        assert_eq!(FRAME_SIZE, 512);
    }

    #[test]
    fn default_threshold_is_silero_recommended() {
        assert!((DEFAULT_THRESHOLD - 0.5).abs() < 1e-6);
    }

    #[test]
    fn empty_input_does_not_panic() {
        let vad = SileroVad::with_threshold(DEFAULT_THRESHOLD)
            .expect("embedded Silero model must load");
        assert!(!vad.has_speech(&[]));
        assert_eq!(vad.speech_probability(&[]), 0.0);
    }

    #[test]
    fn detects_silence_as_no_speech() {
        let vad = SileroVad::with_threshold(DEFAULT_THRESHOLD)
            .expect("embedded Silero model must load");
        let silence = synthetic_silence(FRAME_SIZE * 4);
        assert!(!vad.has_speech(&silence));
    }

    #[test]
    fn detects_modulated_tone_as_speech() {
        let vad = SileroVad::with_threshold(DEFAULT_THRESHOLD)
            .expect("embedded Silero model must load");
        let speech_like = synthetic_speech_like(FRAME_SIZE * 8);
        let p_silence = vad.speech_probability(&synthetic_silence(FRAME_SIZE * 8));
        vad.reset_state();
        let p_speech = vad.speech_probability(&speech_like);
        // Demand a non-trivial gap between speech-like input and silence
        // so the test catches regressions where both probabilities are
        // pinned near zero (the symptom we hit while integrating ORT
        // directly).
        let speech_to_silence_ratio = (p_speech + 1e-6) / (p_silence + 1e-6);
        assert!(
            speech_to_silence_ratio >= 5.0,
            "speech probability ({}) must be at least 5x silence probability ({}); \
             ratio {} suggests the model is returning degenerate values",
            p_speech, p_silence, speech_to_silence_ratio,
        );
    }

    #[test]
    fn from_path_ignores_path_argument() {
        // The `_model_path` parameter exists for API stability with
        // earlier callers; the crate ships the model embedded, so any
        // path (even nonexistent) must succeed.
        let _vad = SileroVad::from_path(&model_path())
            .expect("embedded Silero model must load regardless of path");
    }
}
