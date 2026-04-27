//! Streaming STT with rolling window + boundary-triggered commit.
//!
//! Earlier this module attempted a textbook Local Agreement-2 (Macháček et
//! al., 2023) implementation. That algorithm is correct in theory but had
//! two practical failure modes in our pipeline:
//!
//!   1. After a window slide, the new (shorter) transcript no longer
//!      aligns by index with the previous one. The longest-common-prefix
//!      check then either committed nothing (false negative) or, worse,
//!      silently re-committed words from the kept tail (duplicates).
//!   2. Whisper's transcript at temperature-0 *usually* stabilises but a
//!      single unstable token in the middle of the prefix invalidates the
//!      whole LCP — so a misread word that flips between runs blocks
//!      every subsequent commit until pause.
//!
//! The replacement is much simpler and correct by construction: keep a
//! rolling window of audio, run Whisper on it on every chunk, and emit the
//! full transcript exactly once when an *utterance boundary* is detected
//! — defined as one of:
//!
//!   - The transcript ends with sentence-final punctuation (`.`, `!`, `?`).
//!   - The trailing audio went silent (the speaker paused).
//!   - The window has been growing for `MAX_WINDOW_SECONDS` without one
//!     of the above (long monologue safety valve).
//!
//! On commit, the window is cleared and the next chunk starts a fresh
//! window. This means **each transcript is committed exactly once** —
//! no duplicates, no LCP gymnastics. The cost is per-utterance latency
//! (we pay one Whisper call worth of context, not 250 ms) but that
//! latency is bounded by the punctuation / pause heuristic and capped by
//! `MAX_WINDOW_SECONDS`.

use crate::WhisperStt;
use shared::{AudioChunk, Language};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Sample rate Whisper itself runs at. Audio fed into the session must
/// already be at this rate (the existing pipeline resamples on capture).
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Maximum rolling-window length in seconds. When this is exceeded we
/// force-commit even without a punctuation or pause boundary, so a
/// motormouth speaker doesn't accumulate unbounded state.
const MAX_WINDOW_SECONDS: f32 = 12.0;

/// Minimum window length before we bother running Whisper. Below this
/// the model produces unstable, often-hallucinated transcripts that hurt
/// downstream more than they help.
const MIN_INFERENCE_SECONDS: f32 = 1.0;

/// Minimum interval between successive Whisper calls. Audio chunks
/// arrive every ~500 ms, but Whisper itself takes ~150–500 ms per
/// inference; running on every chunk would queue up. 250 ms keeps the
/// pipeline responsive without saturating the GPU.
const MIN_INFERENCE_INTERVAL_MS: u64 = 250;

/// Trailing-silence window (seconds) used to detect a speaker pause.
/// Mirrors the heuristic the previous accumulator used.
const TAIL_SILENCE_SECONDS: f32 = 0.35;

/// Minimum word count for a clause-boundary commit (transcript ends with
/// a comma or semicolon). Lower than that we'd be shipping fragments like
/// "However, " which the translator turns into nonsense.
const MIN_WORDS_FOR_CLAUSE_COMMIT: usize = 4;

/// RMS threshold below which the trailing window counts as silent. Same
/// value as the energy gate in `crates/pipeline/src/lib.rs`.
const TAIL_SILENCE_RMS: f32 = 0.005;

/// A complete utterance committed by the streaming session. The pipeline
/// turns each one into a translation request.
#[derive(Debug, Clone)]
pub struct CommittedWords {
    pub words: Vec<String>,
    pub language: Language,
    /// True when the trailing audio of this utterance was silent — the
    /// pipeline can use this as an extra hint that the speaker has
    /// finished (vs. punctuation, which only suggests they did).
    pub tail_silent: bool,
}

pub struct StreamingSession {
    stt: Arc<WhisperStt>,
    language: Language,
    /// Rolling audio window for the current utterance. Cleared on commit.
    window: Vec<f32>,
    /// When the current window started accumulating, used to enforce
    /// `MAX_WINDOW_SECONDS`.
    window_start: Option<Instant>,
    last_inference: Option<Instant>,
}

impl StreamingSession {
    pub fn new(stt: Arc<WhisperStt>, language: Language) -> Self {
        Self {
            stt,
            language,
            window: Vec::with_capacity(
                (WHISPER_SAMPLE_RATE as f32 * MAX_WINDOW_SECONDS) as usize,
            ),
            window_start: None,
            last_inference: None,
        }
    }

    /// Append fresh audio. Returns `Some(CommittedWords)` when the
    /// session detects an utterance boundary; `None` otherwise (just
    /// buffering, or Whisper-throttle blocked the inference call).
    pub fn push_audio(&mut self, samples: &[f32]) -> Option<CommittedWords> {
        if self.window_start.is_none() {
            self.window_start = Some(Instant::now());
        }
        self.window.extend_from_slice(samples);

        if self.window.len()
            < (MIN_INFERENCE_SECONDS * WHISPER_SAMPLE_RATE as f32) as usize
        {
            return None;
        }

        if let Some(t) = self.last_inference {
            if t.elapsed() < Duration::from_millis(MIN_INFERENCE_INTERVAL_MS) {
                return None;
            }
        }
        self.last_inference = Some(Instant::now());

        let chunk = AudioChunk::new(self.window.clone(), WHISPER_SAMPLE_RATE, 1);
        let segment = match self.stt.transcribe(&chunk, self.language) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("Streaming STT inference failed: {}", e);
                return None;
            }
        };

        let detected_lang = segment.language;
        let trimmed = segment.text.trim().to_string();
        if trimmed.is_empty() {
            return None;
        }

        let tail_silent = self.tail_is_silent();
        let ends_with_punct = ends_with_sentence_punct(&trimmed);
        let word_count = trimmed.split_whitespace().count();
        // Clause boundary: comma / semicolon + enough words. This is the
        // coherence-based commit trigger — when the speaker has finished
        // a self-contained clause, we release it without waiting for the
        // full sentence. The downstream translator handles fragments
        // gracefully when there's at least a 4-word context.
        let ends_with_clause = (trimmed.ends_with(',') || trimmed.ends_with(';'))
            && word_count >= MIN_WORDS_FOR_CLAUSE_COMMIT;
        let window_too_long = self
            .window_start
            .map(|t| t.elapsed().as_secs_f32() >= MAX_WINDOW_SECONDS)
            .unwrap_or(false);

        if !(tail_silent || ends_with_punct || ends_with_clause || window_too_long) {
            return None;
        }

        let words = split_words(&trimmed);
        // The boundary closes the current utterance — clear the window
        // so the next chunk starts fresh. This is what guarantees a
        // single commit per utterance.
        self.window.clear();
        self.window_start = None;

        if words.is_empty() {
            None
        } else {
            Some(CommittedWords {
                words,
                language: detected_lang,
                tail_silent,
            })
        }
    }

    /// Force-emit whatever is in the window as one final utterance.
    /// Called when the upstream audio source goes idle (long silence),
    /// so a trailing fragment doesn't sit forever waiting for the next
    /// boundary that never comes.
    pub fn flush_tentative(&mut self) -> Option<CommittedWords> {
        if self.window.is_empty() {
            return None;
        }
        // Run Whisper one last time on the residual buffer so we don't
        // lose whatever the speaker said in the final fragment.
        let chunk = AudioChunk::new(self.window.clone(), WHISPER_SAMPLE_RATE, 1);
        let segment = match self.stt.transcribe(&chunk, self.language) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("Streaming STT flush_tentative failed: {}", e);
                self.reset();
                return None;
            }
        };
        let trimmed = segment.text.trim().to_string();
        let words = split_words(&trimmed);
        let lang = segment.language;
        self.reset();
        if words.is_empty() {
            None
        } else {
            Some(CommittedWords {
                words,
                language: lang,
                tail_silent: true,
            })
        }
    }

    /// Reset all state. Used on PipelineCommand::Stop and after any
    /// commit so the next utterance starts clean.
    pub fn reset(&mut self) {
        self.window.clear();
        self.window_start = None;
        self.last_inference = None;
    }

    fn tail_is_silent(&self) -> bool {
        let tail_len = (WHISPER_SAMPLE_RATE as f32 * TAIL_SILENCE_SECONDS) as usize;
        if self.window.len() < tail_len {
            return false;
        }
        let tail = &self.window[self.window.len() - tail_len..];
        let sum_sq: f32 = tail.iter().map(|s| s * s).sum();
        let rms = (sum_sq / tail.len() as f32).sqrt();
        rms < TAIL_SILENCE_RMS
    }
}

/// Split a transcript into whitespace-separated tokens, preserving any
/// trailing punctuation as part of the surrounding word. The downstream
/// translator joins them back with spaces, so the round-trip is
/// information-preserving for any unicode-clean transcript.
fn split_words(text: &str) -> Vec<String> {
    text.split_whitespace().map(|s| s.to_string()).collect()
}

fn ends_with_sentence_punct(text: &str) -> bool {
    let trimmed = text.trim_end();
    trimmed.ends_with('.')
        || trimmed.ends_with('!')
        || trimmed.ends_with('?')
        || trimmed.ends_with('…')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_words_handles_whitespace() {
        let words = split_words("  hello   world ");
        assert_eq!(words, vec!["hello", "world"]);
    }

    #[test]
    fn ends_with_punct_period() {
        assert!(ends_with_sentence_punct("Hello world."));
    }

    #[test]
    fn ends_with_punct_question() {
        assert!(ends_with_sentence_punct("How are you?"));
    }

    #[test]
    fn ends_with_punct_no_punctuation() {
        assert!(!ends_with_sentence_punct("Still talking"));
    }

    #[test]
    fn ends_with_punct_ignores_trailing_whitespace() {
        assert!(ends_with_sentence_punct("Hello world.  "));
    }

    #[test]
    fn ends_with_punct_does_not_match_comma() {
        // Commas are NOT utterance boundaries — the speaker is still
        // mid-thought. Only sentence-final marks qualify.
        assert!(!ends_with_sentence_punct("Hello,"));
    }

    #[test]
    fn ends_with_punct_handles_unicode_ellipsis() {
        assert!(ends_with_sentence_punct("Maybe…"));
    }
}
