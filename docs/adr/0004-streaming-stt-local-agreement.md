# ADR 0004 — Streaming STT via Local Agreement-2

- **Status:** Accepted
- **Date:** 2026-04-26
- **Deciders:** felix
- **Related:** ADR 0003 (CosyVoice TTS migration)

## Context

The previous pipeline buffered ~2.5 s of audio before the first call to
Whisper, then ran a second waiting layer (the text accumulator) for a
sentence boundary before the translator saw anything. Combined, these
two waits added 1.5–3 s of latency before the network of fast stages
(translation 30–80 ms, TTS ~150 ms) even started.

Whisper itself is throughput-friendly — q5_1 small at GPU runs roughly
7–10× real-time on 5–8 s clips — but the project was using it as a
batch processor instead of as a streaming primitive.

## Decision

Implement streaming STT using **Local Agreement-2** (Macháček, Polák &
Hladová, 2023):

- A rolling 8 s audio window is kept per pipeline.
- Whisper is invoked at most every 250 ms on the entire window.
- Each new transcript is compared word-by-word with the previous one:
  the **longest common prefix** is the part both runs agree on. New
  words inside that prefix (i.e. words that weren't in the previous
  prefix) are released downstream immediately.
- The trailing tail of the transcript stays uncommitted until a later
  run confirms it.
- Once committed audio passes 6 s, the window is trimmed to its
  trailing 3 s so Whisper still has lookback context but compute stays
  bounded for long monologues.

This trades a bit of redundant Whisper compute (each window is run
multiple times) for per-word latency proportional to the chunk
interval, not the window length.

The implementation lives in `crates/stt/src/streaming.rs` as
`StreamingSession`, used by `start_stt_worker` in
`crates/pipeline/src/lib.rs`.

## Alternatives considered

1. **Shrink chunk to 500 ms, no Local Agreement.** Cuts the wait but
   destroys Whisper accuracy — short clips have no context, increase
   hallucinations and language-detection errors.
2. **NVIDIA Parakeet / Nemotron-Streaming-0.6B.** Native streaming
   RNN-T, 80–320 ms chunks. Would eliminate the redundant compute at
   the cost of swapping the entire STT engine and re-quantising for
   GPU. Decided to keep whisper.cpp because (a) it's already
   integrated and tuned in `crates/stt/src/lib.rs`, and (b) PT-BR
   quality is well-known.
3. **whisper-streaming as a subprocess.** Exists upstream, written in
   Python. Considered, but pulls another Python bridge with its own
   model load. Local Agreement is small enough to inline in Rust.

## Consequences

### Positive
- First committed words appear ~1 s after the speaker starts talking,
  not 2.5–3 s.
- Sentence boundary detection (punctuation/pause) now sees text as it
  arrives, so the accumulator can flush within the same speaker
  utterance instead of waiting for the next chunk.
- Speaker-change confidence smoothing (ADR 0005) lives naturally in
  the same STT worker — diarisation runs per-chunk while STT commits
  per-word, both attributed to the same stable speaker_id.
- Reorder buffers in the STT path are gone; commits are inherently
  ordered by Local Agreement.

### Negative
- Whisper inference runs ~3–4× more often than before. Net GPU cost
  goes up (offset by removing KNN-VC, which freed ~500 ms per
  utterance — see ADR 0003).
- Punctuation in committed words is sensitive to Whisper's output —
  a comma that flips between runs delays commit by one cycle. In
  practice the temperature-0 sampling in `crates/stt/src/lib.rs` makes
  this rare.
- Force-flush is needed when the speaker stops mid-sentence, otherwise
  the trailing uncommitted words sit forever. `flush_tentative()`
  fires when an empty (silent) chunk arrives.

### Neutral
- The `WhisperStt` primitive is unchanged — `StreamingSession` is a
  thin wrapper that calls `transcribe()` on slices of its rolling
  buffer.

## Rollout

1. New module `crates/stt/src/streaming.rs` with `StreamingSession`,
   `CommittedWords`, `flush_tentative`, `reset`.
2. `crates/stt/src/lib.rs` re-exports the public types.
3. `crates/pipeline/src/lib.rs` replaces the old chunk accumulator
   with a long-lived spawn_blocking STT worker that owns the session.
4. `chunk_duration_ms` default in `crates/shared/src/config.rs`
   moves from 2000 to 500 — the streaming session throttles
   internally at 250 ms, so 500 ms chunks match its cadence well.

## Rollback

Revert to the previous `crates/pipeline/src/lib.rs` accumulator
design. `StreamingSession` is fully self-contained and doesn't change
the underlying `WhisperStt` API.
