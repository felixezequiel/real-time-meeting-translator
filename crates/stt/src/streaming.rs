//! Streaming STT — partial Whisper passes inside an open phrase window
//! with **Local Agreement-N** word-level commit (ADR 0015).
//!
//! Why this is shape is different from the V1 streaming module that
//! lived here before (deleted in ADR 0013 Phase 3.4):
//!
//!   - V1 owned its own rolling audio window AND slid it after each
//!     commit. The slide caused word-position drift between consecutive
//!     partials, which broke the textbook LA-2 alignment by index.
//!   - V1 also tried to commit on per-utterance boundaries
//!     (punctuation / pause / cap), which conflated the *window* and
//!     the *commit unit* and made the algorithm complicated.
//!
//! This module does the opposite — it is *passive on the audio side*:
//!
//!   - The PhraseSegmenter (`crates/audio`) owns the open buffer. It
//!     never slides mid-utterance; the buffer monotonically grows
//!     until the phrase closes.
//!   - `run_partial(open_buffer)` is called periodically by the V2
//!     pipeline; it runs Whisper on the *current* buffer and applies
//!     LA-N at word-position level. Because the buffer never shrinks
//!     mid-phrase, word positions stay aligned across partials and
//!     index-based agreement is correct by construction.
//!   - Commit is monotonic — `committed.len()` only grows. Returned
//!     words are always strictly past the previous commit point.
//!   - When the phrase closes, V2 calls `finalize(final_transcript)`.
//!     That returns whatever portion of the final transcript hadn't
//!     already been emitted. We never "unsay" a committed word, even
//!     if the final transcribe disagrees — the audio is already
//!     playing or about to play, and audio rewinds are worse UX than
//!     a single occasional wrong word.
//!
//! Word-level Local Agreement-N (Macháček, Polák & Hladová, 2023):
//!
//! ```text
//! Commit position i  ⟺  word_at_i is identical in the last N
//!                       consecutive partial transcripts.
//! ```
//!
//! N = `LOCAL_AGREEMENT_N` (currently 2). Higher N is more cautious
//! (fewer wrong commits) at the cost of one more partial-cycle of
//! latency per stable word. The pure LA function is exposed for unit
//! testing without spinning up Whisper.

use crate::WhisperStt;
use shared::{AudioChunk, Language};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Whisper sample rate. Audio fed into the session must already be at
/// this rate (the V2 pipeline guarantees this — capture / loopback are
/// both resampled to 16 kHz upstream).
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Number of consecutive partials that must agree on a word at a given
/// position before the word is committed. Two is the textbook LA-2
/// value — high enough to reject single-pass instabilities, low enough
/// that commits don't fall further than one cycle behind speech.
pub const LOCAL_AGREEMENT_N: usize = 2;

/// Minimum interval between successive Whisper partial calls. The V2
/// PhraseSegmenter ingests every ~280 ms; we don't want to re-decode
/// on every chunk because per-call Whisper time is in the same ballpark
/// (~150-300 ms). 400 ms gives Whisper time to finish one partial
/// before we ask for the next, and keeps the cumulative compute under
/// roughly 2× real-time on small-q5_1 / RTX 3050.
pub const PARTIAL_INTERVAL_MS: u64 = 400;

/// Minimum buffer length before the first partial pass. Below this
/// Whisper hallucinates aggressively (no temporal context). Tied to the
/// V2 `phrase_min_window_ms` floor — if the segmenter would have
/// rejected the buffer for being too short anyway, no point asking
/// Whisper to score it.
pub const MIN_PARTIAL_SECONDS: f32 = 0.6;

/// Word-level Local Agreement-N. Commits the longest prefix where every
/// word at position i is identical across the LAST `n` partials.
///
/// Invariant: returns a prefix at most as long as the *shortest* of the
/// last `n` partials (positions beyond that don't exist in every
/// partial, so no agreement is possible).
///
/// Public so the V2 pipeline (and tests) can drive it without owning
/// a `StreamingSession`.
pub fn longest_stable_prefix(partials: &[Vec<String>], n: usize) -> Vec<String> {
    if n == 0 || partials.len() < n {
        return Vec::new();
    }
    let recent = &partials[partials.len() - n..];
    let min_len = recent.iter().map(|p| p.len()).min().unwrap_or(0);
    let mut stable = Vec::with_capacity(min_len);
    for i in 0..min_len {
        let candidate = &recent[0][i];
        if recent.iter().all(|p| &p[i] == candidate) {
            stable.push(candidate.clone());
        } else {
            break;
        }
    }
    stable
}

/// Tokenise a transcript on whitespace, preserving any attached
/// punctuation as part of the surrounding word. Round-trip is
/// information-preserving for any unicode-clean input.
fn split_words(text: &str) -> Vec<String> {
    text.split_whitespace().map(|s| s.to_string()).collect()
}

/// Output of `StreamingSession::finalize`. Carries both the suffix that
/// hadn't yet been committed AND a hint about whether the final pass
/// disagreed with what was already emitted — the V2 pipeline can use
/// this to log a metric or surface the divergence in a debug panel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalisedPhrase {
    /// Words from the final transcribe that the streaming partials had
    /// not yet committed. These should be appended to the accumulator
    /// after the streaming-committed prefix.
    pub uncommitted_suffix: Vec<String>,
    /// True when the final transcribe's prefix disagreed with the
    /// already-committed words. The session keeps the streaming commit
    /// regardless — this flag is purely diagnostic.
    pub committed_diverged: bool,
}

/// Streaming STT session. One per pipeline (Speaker / Mic). Keeps the
/// last-N partial transcripts and the running commit prefix; emits
/// newly-committed words on every `run_partial` call.
pub struct StreamingSession {
    stt: Arc<WhisperStt>,
    language: Language,
    /// Words already committed within the current open phrase. Reset on
    /// `finalize` and `reset`.
    committed: Vec<String>,
    /// Sliding history of the last N partial transcripts. Cap is N
    /// itself — we only need the last N to compute LA-N.
    history: Vec<Vec<String>>,
    last_inference: Option<Instant>,
}

impl StreamingSession {
    pub fn new(stt: Arc<WhisperStt>, language: Language) -> Self {
        Self {
            stt,
            language,
            committed: Vec::new(),
            history: Vec::with_capacity(LOCAL_AGREEMENT_N),
            last_inference: None,
        }
    }

    /// Run a partial Whisper pass on the current open buffer, applying
    /// LA-N to commit any newly-stable words. Returns the words emitted
    /// by *this* call (i.e. words that crossed the agreement threshold
    /// since the previous call); empty when the buffer is too short,
    /// the throttle kicked in, or no new agreement was reached.
    ///
    /// `open_buffer` is borrowed — the session never copies the entire
    /// buffer permanently; it only keeps what Whisper returns (the
    /// transcript). The PhraseSegmenter retains ownership of the audio.
    pub fn run_partial(&mut self, open_buffer: &[f32]) -> Vec<String> {
        let buffer_seconds = open_buffer.len() as f32 / WHISPER_SAMPLE_RATE as f32;
        if buffer_seconds < MIN_PARTIAL_SECONDS {
            return Vec::new();
        }
        if let Some(t) = self.last_inference {
            if t.elapsed() < Duration::from_millis(PARTIAL_INTERVAL_MS) {
                return Vec::new();
            }
        }
        self.last_inference = Some(Instant::now());

        let chunk = AudioChunk::new(open_buffer.to_vec(), WHISPER_SAMPLE_RATE, 1);
        let transcribed = match self.stt.transcribe(&chunk, self.language) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!("Streaming partial Whisper pass failed: {}", e);
                return Vec::new();
            }
        };

        let words = split_words(transcribed.text.trim());
        if words.is_empty() {
            return Vec::new();
        }
        self.push_partial(words);

        let stable = longest_stable_prefix(&self.history, LOCAL_AGREEMENT_N);
        if stable.len() <= self.committed.len() {
            return Vec::new();
        }
        let new_words = stable[self.committed.len()..].to_vec();
        self.committed = stable;
        new_words
    }

    /// Called by the V2 pipeline when the PhraseSegmenter closes the
    /// window. `final_transcript` is the result of a fresh
    /// `WhisperStt::transcribe` on the closed segment — the
    /// authoritative read. We keep what's already committed (audio is
    /// playing) and return the suffix that's new.
    ///
    /// Resets the session for the next phrase.
    pub fn finalize(&mut self, final_transcript: &str) -> FinalisedPhrase {
        let final_words = split_words(final_transcript.trim());
        let committed_len = self.committed.len();

        // Did the final transcribe disagree with what we already played?
        // We detect this by comparing position-by-position up to
        // `min(committed_len, final_words.len())`. If a position differs,
        // the streaming pass over-committed and the final pass would
        // have said something else. We log it but still keep the
        // streaming commit — undoing audio is worse UX.
        let compare_len = committed_len.min(final_words.len());
        let mut diverged = final_words.len() < committed_len;
        for i in 0..compare_len {
            if final_words[i] != self.committed[i] {
                diverged = true;
                break;
            }
        }

        let suffix = if final_words.len() > committed_len {
            final_words[committed_len..].to_vec()
        } else {
            Vec::new()
        };

        self.reset();
        FinalisedPhrase {
            uncommitted_suffix: suffix,
            committed_diverged: diverged,
        }
    }

    /// Drop all per-phrase state. Called on `PipelineCommand::Stop` and
    /// internally after `finalize`.
    pub fn reset(&mut self) {
        self.committed.clear();
        self.history.clear();
        self.last_inference = None;
    }

    /// Words committed so far within the current phrase. Useful for
    /// the pipeline to know how many words it has already emitted to
    /// the accumulator (avoids double-counting on edge cases).
    #[allow(dead_code)]
    pub fn committed_count(&self) -> usize {
        self.committed.len()
    }

    fn push_partial(&mut self, words: Vec<String>) {
        if self.history.len() >= LOCAL_AGREEMENT_N {
            self.history.remove(0);
        }
        self.history.push(words);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &str) -> Vec<String> {
        s.split_whitespace().map(|w| w.to_string()).collect()
    }

    // ─── longest_stable_prefix: the algorithmic core ───────────────────────

    #[test]
    fn la_returns_empty_when_history_below_n() {
        let history = vec![words("hello world")];
        // Only one partial — LA-2 needs two. Nothing stable yet.
        assert_eq!(longest_stable_prefix(&history, 2), Vec::<String>::new());
    }

    #[test]
    fn la_returns_full_prefix_when_two_partials_match() {
        let history = vec![words("hello world"), words("hello world")];
        assert_eq!(longest_stable_prefix(&history, 2), words("hello world"));
    }

    #[test]
    fn la_returns_common_prefix_when_partials_diverge() {
        // partials agree on "hello" then differ at position 1.
        let history = vec![words("hello world"), words("hello there")];
        assert_eq!(longest_stable_prefix(&history, 2), words("hello"));
    }

    #[test]
    fn la_handles_growing_partials() {
        // Realistic streaming case: partial 1 had 3 words, partial 2
        // grew to 5. The shared 3-word prefix is stable.
        let history = vec![
            words("I want to"),
            words("I want to refactor this"),
        ];
        assert_eq!(longest_stable_prefix(&history, 2), words("I want to"));
    }

    #[test]
    fn la_blocks_commit_on_mid_prefix_flip() {
        // Whisper changed "want" to "really" at position 1. Nothing
        // past position 0 is stable.
        let history = vec![
            words("I want to refactor"),
            words("I really want to refactor"),
        ];
        assert_eq!(longest_stable_prefix(&history, 2), words("I"));
    }

    #[test]
    fn la_uses_only_last_n_partials() {
        // Older partial said "I want this" but the recent pair both say
        // "I want that" — LA-2 looks at the last 2, so the older
        // disagreement doesn't block commit.
        let history = vec![
            words("I want this"),
            words("I want that"),
            words("I want that"),
        ];
        assert_eq!(longest_stable_prefix(&history, 2), words("I want that"));
    }

    #[test]
    fn la_with_n3_requires_three_consecutive_agreements() {
        // Only 2 consecutive agreements — LA-3 says "not yet stable".
        let history = vec![
            words("I want this"),
            words("I want that"),
            words("I want that"),
        ];
        assert_eq!(longest_stable_prefix(&history, 3), words("I want"));
    }

    #[test]
    fn la_n_zero_returns_empty() {
        let history = vec![words("hello"), words("hello")];
        // Defensive — N=0 means "agree with nothing", caller bug.
        assert!(longest_stable_prefix(&history, 0).is_empty());
    }

    #[test]
    fn la_handles_empty_partial_in_history() {
        // First partial was empty (model said nothing); second has
        // content. min_len = 0, so nothing is stable.
        let history = vec![Vec::<String>::new(), words("hello world")];
        assert!(longest_stable_prefix(&history, 2).is_empty());
    }

    #[test]
    fn la_punctuation_attaches_to_word() {
        // Word tokens carry trailing punctuation — "world." is one
        // token, not two. So a partial that adds the period flips the
        // last word from "world" to "world." and stops being stable
        // until the next partial confirms the new form.
        let history = vec![
            words("hello world"),
            words("hello world."),
        ];
        // Only "hello" agrees position-for-position; "world" vs
        // "world." differs.
        assert_eq!(longest_stable_prefix(&history, 2), words("hello"));
    }

    // ─── split_words helpers ────────────────────────────────────────────────

    #[test]
    fn split_words_collapses_whitespace() {
        assert_eq!(split_words("  hello   world "), words("hello world"));
    }

    #[test]
    fn split_words_handles_empty() {
        let result: Vec<String> = split_words("");
        assert!(result.is_empty());
    }
}
