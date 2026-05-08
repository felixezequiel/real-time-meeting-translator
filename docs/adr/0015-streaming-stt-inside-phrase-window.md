# ADR 0015 — Streaming STT inside the V2 phrase window (LocalAgreement-2)

- **Status:** Proposed (drafted 2026-05-08; implementation pending)
- **Date:** 2026-05-08
- **Deciders:** felix
- **Related:** ADR 0004 (superseded streaming STT), ADR 0013 (V2 hybrid pipeline), ADR 0009 (Qwen streaming translation), ADR 0008 (Silero VAD)

## Context

ADR 0013 Phase 3.4 cleaned up V1's streaming local-agreement code and
made V2's phrase-aligned pipeline the sole path. The V2 architecture
solved real problems — Sepformer permutation chaos, GPU contention
from N parallel Whisper instances, fragmented translations — but it
also baked in a structural latency floor: the listener hears nothing
until the *entire phrase window has closed* and then runs through
STT → translate → TTS in series.

The 2026-05-08 amendments to ADR 0013 (latency-first accumulator) and
ADR 0009 (compression target in the prompt) brought the typical TTFA
from ~3-4 s into the 1.8-2.2 s range. That hits the user's *acceptable*
target ("no máximo 3 s") but not the *preferred* target ("2 s se
possível"). The dominant remaining contributor is the
phrase-window-then-STT shape itself — even with `silence_tail = 280 ms`
and the soft-flush rule, a typical phrase pays:

```
silence_tail 280  ─┐
STT 250            │  serial, all paid before the listener hears anything
accum_soft 800     ┘
translate 200      ── streaming starts overlapping here
TTS 250            ── overlaps with translate
                   ───────────────
                    ≈ 1780 ms common case TTFA
```

The first three rows are the structural floor. To get TTFA under 1 s
we need to **start STT-translate-TTS *while* the phrase is still being
spoken**, not after it closes. This is what professional simultaneous
interpreters do, and it's what V1's local-agreement architecture
attempted before the Sepformer / GPU contention combo broke it.

The conditions that broke V1's streaming in 2026-05-07 are now
resolved: only one Whisper instance per pipeline (ADR 0013
Phase 1+2), Sepformer is conditional and rarely armed (ADR 0013
Phase 3.2), the accumulator pattern is proven for translation
coherence (ADR 0013 amendments). The streaming idea was never
wrong — its environment was hostile. The environment is now
friendly.

## Decision

Re-introduce streaming STT, but **inside the open phrase window
rather than as a replacement for V2's phrase-aligned shape**. The
phrase window still defines the unit of meaning the translator sees;
the new streaming layer commits *partial words* into the accumulator
*before the window closes*, so translation and TTS can begin earlier.

### Architecture

```
[Audio chunk] ──► [VAD] ──► [PhraseSegmenter (open window)]
                                     │
                                     │  partial transcripts
                                     ▼
                             [StreamingSession]
                                (LocalAgreement-2)
                                     │
                                     │  newly-committed words
                                     ▼
                                [Accumulator]
                            (existing flush rules:
                             punctuation / soft-flush /
                             max-hold / max-words)
                                     │
                                     ▼
                              [translate_stream]
                              [tts streaming]
                              [audio_output]

                           ──────────────────────

[PhraseSegmenter (closed window)] ──► [WhisperStt::transcribe (final pass)]
                                              │
                                              ▼
                                  [Reconcile committed vs. final]
                                              │
                                              ▼
                                   [Flush accumulator tail]
```

### How it composes with V2

1. **Phrase window stays.** `PhraseSegmenter` continues to define a
   unit of meaning; `silence_tail` and `max_window` still close
   windows. Diarisation still runs per closed window. The translator
   still receives phrase-shaped chunks via the accumulator's flush
   logic. None of that changes.

2. **A `StreamingSession` runs *inside* the open window.** While the
   `PhraseSegmenter` is accumulating samples for a window, every
   `WHISPER_PARTIAL_INTERVAL_MS` (proposed: 400 ms) the open buffer
   is fed through Whisper for a *partial* transcription. The
   `StreamingSession` keeps the previous partial and applies
   **LocalAgreement-2**: words that appear in two consecutive
   partials at the same prefix position are *committed* and pushed
   into the accumulator immediately.

3. **Accumulator behaviour is unchanged.** Committed words enter the
   accumulator the same way fully-transcribed phrases do today. The
   existing flush rules (punctuation, soft-flush at 6 words / 800 ms,
   speaker change, hard caps) still decide when to release. The
   accumulator can now flush mid-window because it sees text earlier.

4. **Final reconciliation on window close.** When the window finally
   closes, a final `WhisperStt::transcribe` call runs on the full
   closed segment. The output is compared with the words already
   committed by the streaming session; any committed tail that
   diverges is **kept as committed** (we already played it — undoing
   audio is worse UX than living with one wrong word) but the
   reconciliation result REPLACES the uncommitted tail still sitting
   in the streaming buffer, and the accumulator is fed the
   reconciled remainder. Rare wrong-commit cases are masked by the
   interpreter prompt's natural compression — a half-clause that
   the model "didn't hear right" tends to be dropped or paraphrased
   into the next clause anyway.

### Why LocalAgreement-2 (not chunked Whisper, not RNN-T)

ADR 0004's prior implementation worked. The bug was its environment,
not the algorithm. Resurrecting `StreamingSession` is the cheapest
path to streaming STT we have: the rolling buffer + per-cycle
re-transcribe + word-prefix agreement is ~200 lines of Rust,
isolated in one module, with no Python bridge.

Other options considered:

1. **Sliding-chunk Whisper without LA-2.** Run Whisper on consecutive
   500 ms chunks and emit each result. Drops accuracy hard — short
   clips trigger Whisper hallucinations and language-detection
   wobble.
2. **Switch STT engine to NVIDIA Parakeet / Riva** (RNN-T, native
   streaming). Best theoretical latency. Massive migration cost: new
   CUDA stack, new quantisation pipeline, lose the proven PT-BR
   quality of whisper-small. Out of scope for a latency tweak.
3. **whisper-streaming Python lib as a bridge.** Already a working
   LA-2 implementation but adds another Python subprocess and another
   model load. Reimplementing in Rust avoids the bridge.

## Consequences

### Positive

- **TTFA budget drops to ~600-1200 ms** in the common case — words
  appear in the accumulator while the speaker is still talking, the
  soft-flush rule fires sooner, translation and TTS overlap with the
  rest of the speaker's utterance.
- **Subtitle responsiveness goes way up.** Words land in the overlay
  ~400-800 ms after being spoken (one StreamingSession cycle), close
  to the "live caption" UX of YouTube auto-captions.
- **Interpreter behaviour matches reality.** TV simultaneous
  interpreters already speak ~2-3 words behind the source; we're
  finally architected for that pacing.
- **Phrase-level coherence preserved.** The accumulator + the final
  reconciliation pass mean the translator still sees a phrase-shaped
  prompt; we don't hand the LLM 3-word fragments.

### Negative

- **Whisper compute increases ~3-4×** during active speech (one
  partial per `WHISPER_PARTIAL_INTERVAL_MS`). At small-q5_1 quant on
  CUDA this is fine on an RTX 3050; on weaker GPUs the partial
  interval can be widened (configurable). Field-monitor.
- **Wrong-commit risk on noisy audio.** LocalAgreement-2 commits when
  two consecutive partials agree on a prefix. Background noise that
  fools Whisper twice in a row produces a wrong-but-played word. Rare
  with temperature-0 sampling and the existing VAD gate, but the
  reconciliation pass can't undo audio that was already synthesised.
  Mitigation: require *three* agreements before committing on the
  first ~500 ms of a window (cold-start is the most error-prone
  region); after that two suffices.
- **Reconciliation logic adds a moving part.** The "what to do when
  the final pass disagrees with what we already played" decision is
  not free of cases: kept-committed-but-skipped, kept-committed-and-
  appended, etc. We commit to one rule (kept-committed always wins
  for words already played; uncommitted tail is replaced) and live
  with the edge cases.

### Neutral

- The hexagonal layering is preserved: `StreamingSession` is a pure-
  domain primitive in `crates/stt`; the V2 pipeline orchestrates it
  alongside `PhraseSegmenter` without leaking either's state.
- Pipeline-V1 nostalgia: this looks like V1 but is *not* V1. V1
  bypassed phrase-level units entirely; V2-with-streaming keeps the
  phrase window as the unit of meaning and just opens a faster
  channel for partial commits inside it.

## Rollout

PRs ordered by dependency. Each is independently reviewable.

1. **ADR 0015** — this document.
2. **Resurrect `StreamingSession`** in `crates/stt/src/streaming.rs`
   from the V1 cleanup commit (git history). Adapt to current types
   (`AudioChunk`, `WhisperStt`). Full unit-test coverage of the
   LA-2 logic: prefix-agreement detection, commit-only-on-two-runs,
   reset behaviour, tail-trim on long windows. **Production code is
   gated behind a NEW config flag `streaming_stt = false` (default
   off) so the resurrection is a no-op for current users.**
3. **`PhraseSegmenter` exposes the open buffer.** Add a
   non-destructive `peek_open_buffer(&self) → &[f32]` accessor so
   the pipeline can run a partial Whisper pass without consuming
   samples.
4. **`SpeakerPipelineV2` integrates `StreamingSession`** behind the
   `streaming_stt` flag. While the flag is off, V2 behaves exactly
   as it does today. While on:
   - Schedule a partial-pass Tokio task every
     `WHISPER_PARTIAL_INTERVAL_MS`.
   - Feed committed words into the accumulator via a new
     `Accumulator::ingest_words(words)` path that bypasses the
     "wait for full segment" assumption.
   - On window close, run the final `transcribe()` and call
     `Accumulator::reconcile_uncommitted(final_text)`.
5. **Field-validate** with `streaming_stt = true`. Iterate on
   `WHISPER_PARTIAL_INTERVAL_MS`, the LA-2 agreement count, and the
   reconciliation rule based on real recordings.
6. **Default flip** — once field-validated, `streaming_stt = true`
   becomes the default. Keep the flag as an escape hatch for one
   more release cycle, then remove.

## Rollback

Set `streaming_stt = false` in `config.toml`. The V2 phrase-aligned
path remains available throughout the rollout. After the default
flip, rollback is reverting the default-flip PR; the implementation
stays alive behind the flag.

## Open questions (resolve during step 2-4)

- **Partial interval.** 400 ms is the proposed default. Lower means
  more compute and faster commits; higher means less compute and
  fewer wrong-commit opportunities. Tune by measuring P50/P95 of
  per-partial Whisper time on the target GPU.
- **Agreement count near window start.** "Three agreements for the
  first 500 ms, two thereafter" is a heuristic. May want to gate
  on VAD confidence instead, or on Whisper's per-token confidence
  if exposed.
- **Diarisation cadence.** Currently runs per closed window.
  Streaming partials don't have a `speaker_id` until close. Decision:
  keep diarisation per-closed-window, attribute committed words to
  the *last known* speaker; reconcile on close. Cheap and correct
  in 95% of cases (single speaker per window).
- **TCC reference path.** The voice profile is also resolved at
  flush time today. Streaming committed words may flush before the
  final speaker_id is known. Same answer as above — use last-known
  reference, accept rare mismatches.
- **Subtitle phrase_id.** The overlay's update-in-place semantics
  expect one `phrase_id` per `flush_phrase` call. With streaming
  committed words triggering earlier flushes, a single utterance may
  produce multiple phrase_ids. Decision: assign one phrase_id per
  *speaker turn* (start at speech onset, end on speaker change or
  long silence), not per accumulator flush. Fold into step 4.
