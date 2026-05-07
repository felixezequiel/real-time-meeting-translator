//! Adaptive VAD-driven phrase segmentation (ADR 0013).
//!
//! Replaces the fixed 500 ms streaming chunk model. A `PhraseSegmenter`
//! accumulates audio while the caller reports speech, and closes a
//! window when one of three conditions is met:
//!
//! 1. The caller reports silence for at least `silence_tail_ms` after
//!    speech has started — the natural end of an utterance.
//! 2. The accumulated audio reaches `max_window_ms` — protects against
//!    a speaker who never pauses.
//! 3. The pipeline is stopped or restarted (`flush`).
//!
//! The segmenter is intentionally decoupled from any specific VAD: the
//! caller passes `is_speech` per ingested slice. This keeps the state
//! machine pure and unit-testable without spinning up Silero. In
//! production, `crates/pipeline` calls `SileroVad::has_speech` and
//! forwards the result.

use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub struct PhraseSegmenterConfig {
    pub max_window: Duration,
    pub silence_tail: Duration,
    pub min_window: Duration,
}

impl Default for PhraseSegmenterConfig {
    fn default() -> Self {
        Self {
            max_window: Duration::from_millis(5000),
            silence_tail: Duration::from_millis(400),
            min_window: Duration::from_millis(600),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloseReason {
    SilenceTail,
    MaxWindow,
    ManualFlush,
}

#[derive(Debug, Clone)]
pub struct PhraseSegment {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub closed_by: CloseReason,
    /// Total samples that were classified as speech across the window.
    /// Useful for diagnostics — short/noisy windows can be filtered out
    /// downstream by threshold.
    pub speech_samples: usize,
}

pub struct PhraseSegmenter {
    sample_rate: u32,
    max_window_samples: usize,
    silence_tail_samples: usize,
    /// Held only so callers can recover the configured threshold via
    /// `min_window_samples()`. The segmenter itself does not filter —
    /// downstream uses `PhraseSegment::speech_samples` to decide.
    min_window_samples: usize,

    buffer: Vec<f32>,
    in_speech: bool,
    silence_run_samples: usize,
    speech_run_samples: usize,
}

impl PhraseSegmenter {
    pub fn new(sample_rate: u32, config: PhraseSegmenterConfig) -> Self {
        let to_samples = |d: Duration| {
            (sample_rate as u128 * d.as_millis() / 1000) as usize
        };
        Self {
            sample_rate,
            max_window_samples: to_samples(config.max_window),
            silence_tail_samples: to_samples(config.silence_tail),
            min_window_samples: to_samples(config.min_window),
            buffer: Vec::new(),
            in_speech: false,
            silence_run_samples: 0,
            speech_run_samples: 0,
        }
    }

    /// Configured `min_window` expressed in samples. Downstream decides
    /// whether to drop a segment whose `speech_samples` falls below this.
    pub fn min_window_samples(&self) -> usize {
        self.min_window_samples
    }

    /// Feed one slice of audio plus its speech classification.
    /// Returns `Some(segment)` when this ingest closed a window.
    pub fn ingest(&mut self, samples: &[f32], is_speech: bool) -> Option<PhraseSegment> {
        if samples.is_empty() {
            return None;
        }

        if is_speech {
            self.in_speech = true;
            self.silence_run_samples = 0;
            self.speech_run_samples += samples.len();
            self.buffer.extend_from_slice(samples);
        } else if self.in_speech {
            self.silence_run_samples += samples.len();
            self.buffer.extend_from_slice(samples);
            if self.silence_run_samples >= self.silence_tail_samples {
                return Some(self.close(CloseReason::SilenceTail));
            }
        }
        // else: pre-speech silence — drop frame, don't even buffer it

        if self.buffer.len() >= self.max_window_samples {
            return Some(self.close(CloseReason::MaxWindow));
        }

        None
    }

    /// Force-close the current window. Returns `None` if no speech has
    /// been buffered yet.
    pub fn flush(&mut self) -> Option<PhraseSegment> {
        if !self.in_speech || self.buffer.is_empty() {
            self.reset_state();
            return None;
        }
        Some(self.close(CloseReason::ManualFlush))
    }

    fn close(&mut self, reason: CloseReason) -> PhraseSegment {
        let samples = std::mem::take(&mut self.buffer);
        let speech_samples = self.speech_run_samples;
        self.reset_state();
        PhraseSegment {
            samples,
            sample_rate: self.sample_rate,
            closed_by: reason,
            speech_samples,
        }
    }

    fn reset_state(&mut self) {
        self.in_speech = false;
        self.silence_run_samples = 0;
        self.speech_run_samples = 0;
        self.buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 16_000;

    fn quick_config() -> PhraseSegmenterConfig {
        PhraseSegmenterConfig {
            max_window: Duration::from_millis(2000),
            silence_tail: Duration::from_millis(200),
            min_window: Duration::from_millis(0),
        }
    }

    fn signal(num_samples: usize, value: f32) -> Vec<f32> {
        vec![value; num_samples]
    }

    fn samples_for_ms(ms: u64) -> usize {
        (SR as u64 * ms / 1000) as usize
    }

    #[test]
    fn pre_speech_silence_is_discarded() {
        let mut seg = PhraseSegmenter::new(SR, quick_config());
        let result = seg.ingest(&signal(samples_for_ms(500), 0.0), false);
        assert!(result.is_none());
    }

    #[test]
    fn segment_closes_after_silence_tail() {
        let mut seg = PhraseSegmenter::new(SR, quick_config());
        // 500 ms of speech, then 200 ms of silence (= silence_tail).
        let s = seg.ingest(&signal(samples_for_ms(500), 0.1), true);
        assert!(s.is_none());
        let s = seg.ingest(&signal(samples_for_ms(200), 0.0), false);
        let segment = s.expect("silence tail should close the window");
        assert_eq!(segment.closed_by, CloseReason::SilenceTail);
        assert_eq!(segment.sample_rate, SR);
        // Window contains both the speech and the trailing silence.
        assert_eq!(segment.samples.len(), samples_for_ms(700));
        assert_eq!(segment.speech_samples, samples_for_ms(500));
    }

    #[test]
    fn segment_closes_at_max_window_cap() {
        let mut seg = PhraseSegmenter::new(SR, quick_config());
        // Push 2000 ms of continuous speech (= max_window_ms).
        let s = seg.ingest(&signal(samples_for_ms(2000), 0.1), true);
        let segment = s.expect("hitting the max cap should close");
        assert_eq!(segment.closed_by, CloseReason::MaxWindow);
        assert_eq!(segment.samples.len(), samples_for_ms(2000));
    }

    #[test]
    fn min_window_filters_out_short_blips() {
        let cfg = PhraseSegmenterConfig {
            max_window: Duration::from_millis(2000),
            silence_tail: Duration::from_millis(200),
            min_window: Duration::from_millis(500),
        };
        let mut seg = PhraseSegmenter::new(SR, cfg);
        // 100 ms of speech then 200 ms of silence — speech run too short.
        seg.ingest(&signal(samples_for_ms(100), 0.1), true);
        let s = seg.ingest(&signal(samples_for_ms(200), 0.0), false);
        // Segment is still emitted (the segmenter doesn't drop it itself —
        // downstream filters by `speech_samples`); but speech_samples is
        // recorded so the caller can apply min_window.
        let segment = s.expect("close still happens");
        assert!(
            segment.speech_samples < samples_for_ms(500),
            "downstream can filter via speech_samples vs configured min"
        );
    }

    #[test]
    fn handles_multiple_segments_in_sequence() {
        let mut seg = PhraseSegmenter::new(SR, quick_config());
        // First phrase
        seg.ingest(&signal(samples_for_ms(400), 0.1), true);
        let first = seg
            .ingest(&signal(samples_for_ms(200), 0.0), false)
            .expect("first close");
        assert_eq!(first.samples.len(), samples_for_ms(600));

        // Second phrase after a gap of silence (which is dropped pre-speech)
        seg.ingest(&signal(samples_for_ms(800), 0.0), false); // no speech yet
        seg.ingest(&signal(samples_for_ms(300), 0.1), true);
        let second = seg
            .ingest(&signal(samples_for_ms(200), 0.0), false)
            .expect("second close");
        assert_eq!(second.samples.len(), samples_for_ms(500));
    }

    #[test]
    fn flush_returns_pending_buffer_when_in_speech() {
        let mut seg = PhraseSegmenter::new(SR, quick_config());
        seg.ingest(&signal(samples_for_ms(300), 0.1), true);
        let segment = seg.flush().expect("flush returns active buffer");
        assert_eq!(segment.closed_by, CloseReason::ManualFlush);
        assert_eq!(segment.samples.len(), samples_for_ms(300));
    }

    #[test]
    fn flush_returns_none_when_no_speech_buffered() {
        let mut seg = PhraseSegmenter::new(SR, quick_config());
        // Only pre-speech silence — nothing buffered.
        seg.ingest(&signal(samples_for_ms(500), 0.0), false);
        assert!(seg.flush().is_none());
    }

    #[test]
    fn segmenter_resets_after_emitting() {
        let mut seg = PhraseSegmenter::new(SR, quick_config());
        seg.ingest(&signal(samples_for_ms(400), 0.1), true);
        let _ = seg
            .ingest(&signal(samples_for_ms(200), 0.0), false)
            .unwrap();
        // After emit, internal state is clean — next speech starts fresh.
        seg.ingest(&signal(samples_for_ms(500), 0.1), true);
        let next = seg
            .ingest(&signal(samples_for_ms(200), 0.0), false)
            .expect("post-reset close");
        assert_eq!(next.samples.len(), samples_for_ms(700));
        assert_eq!(next.speech_samples, samples_for_ms(500));
    }

    #[test]
    fn empty_ingest_is_noop() {
        let mut seg = PhraseSegmenter::new(SR, quick_config());
        assert!(seg.ingest(&[], true).is_none());
        assert!(seg.ingest(&[], false).is_none());
    }
}
