# ADR 0012 — Permutation tracking for Sepformer outputs

- **Status:** Accepted
- **Date:** 2026-05-07
- **Deciders:** felix
- **Related:** ADR 0007 (Sepformer source separation), ADR 0004 (streaming STT)

## Context

ADR 0007 introduced Sepformer-libri2mix as an opt-in stage that
splits the mono loopback into two channel streams (`channel_a`,
`channel_b`), each feeding an independent `SpeakerPipeline`
(`Speaker-A` / `Speaker-B`). The downstream domain code is identical
to the mono path — both branches share the same STT, translator and
TTS via `Arc`, only the audio adapter differs.

In production we observed that with `enable_separation = true` no
translation is produced during a run. Only a single fragment
(`"Oh, looks like."`) is committed at shutdown, when the pipeline
flushes pending state. The streaming STT (ADR 0004) never reaches
its commit threshold while the run is active.

Root cause: **Sepformer is permutation-invariant per chunk**. The
model decides arbitrarily, on every 500 ms call, which output index
holds which speaker. With one active speaker the voice bounces
between `channel_a` and `channel_b` from chunk to chunk. Each
`SpeakerPipeline` therefore sees a fragmented stream — half the
speech, interspersed with the residual "silent" channel — and the
local-agreement streaming STT never accumulates enough stable
context to commit.

The current adapter (`start_separation_worker` in `src/main.rs`)
forwards `channel_a` to `Speaker-A` and `channel_b` to `Speaker-B`
without any cross-chunk correlation. This breaks the implicit
contract `SpeakerPipeline` relies on: each `audio_rx` is a coherent
stream of one speaker over time.

## Decision

Add a stateful **`PermutationTracker`** to the separation crate and
wire it into `start_separation_worker`. The tracker keeps a short
tail (~50 ms) of each previously published channel and, for every
new separated pair, decides whether to swap them so the loud channel
in chunk N+1 lands on the same downstream pipeline as the loud
channel in chunk N.

Continuity metric: **log-RMS of the head window vs the stored tail**.

```
same_cost = |log RMS(tail_a) − log RMS(head_a)|
          + |log RMS(tail_b) − log RMS(head_b)|
swap_cost = |log RMS(tail_a) − log RMS(head_b)|
          + |log RMS(tail_b) − log RMS(head_a)|

if swap_cost < same_cost → swap (channel_a, channel_b)
```

Tails are updated *after* the swap decision so they always reflect
what was actually published downstream. On the first chunk (no
history) the input pair is forwarded unchanged.

Architecturally this stays in the **audio adapter** layer — the
domain (`SpeakerPipeline`, STT, translator, TTS) is untouched. The
mono path is also unchanged.

## Alternatives considered

1. **Do nothing, document limitation.** Rejected — separation is
   currently unusable, not just degraded.
2. **Spectral / MFCC similarity.** More robust than log-RMS for the
   "two simultaneous speakers" case, but adds an FFT dependency for
   a problem that, in 95% of observed traffic (single speaker most
   of the time), is solved by energy alone. Reserved as a follow-up
   if dual-speaker overlap regresses.
3. **Permutation-Invariant Training-aware joint decoder.** Would
   require retraining or replacing Sepformer-libri2mix. Out of scope.
4. **Force separation off when the diariser sees only one ID.**
   Solves the symptom for single-speaker, but the diariser runs *after*
   STT; coupling adapters to its state would invert the layering.

## Consequences

### Positive
- Streaming STT receives a coherent per-speaker stream, so commits
  fire at the normal cadence.
- Zero changes to domain code — pure adapter-layer fix.
- Cheap: log-RMS over a 50 ms window per chunk is < 0.05 ms.

### Negative / Limits
- Energy-only metric can mis-track when two speakers have similar
  loudness throughout a chunk pair. Acceptable: in that regime both
  pipelines still receive speech, the worst case is a single
  per-speaker swap event, recovered on the next quiet/loud
  transition.
- Adds 50 ms × 2 channels of `f32` state per worker (~6.4 KB) —
  negligible.

### Neutral
- New public API `PermutationTracker` in `crates/separation`.
  Mirrors the bridge-client pattern of the existing crate.

## Rollout

1. Add `PermutationTracker` to `crates/separation/src/lib.rs` with
   unit tests (RED → GREEN).
2. Replace the inline `ch_a_tx.send(channel_a) / ch_b_tx.send(channel_b)`
   block in `start_separation_worker` (`src/main.rs`) with a call to
   `tracker.align(...)` and forward the aligned pair.
3. RMS gate (`SILENT_CHANNEL_RMS = 0.01`) is reapplied *after*
   alignment using the tracker's reported RMS, so it gates the right
   channel.

## Follow-up — Sepformer device offload

After the tracker rollout, in-field testing exposed a second issue:
on a 6–8 GB GPU, Sepformer-libri2mix coexists with three Whisper
instances (Speaker-A / Speaker-B / Mic) plus OpenVoice TCC plus the
ECAPA diariser. GPU memory contention occasionally stalls one
Sepformer call for **10–60 seconds**, which destroys the streaming
STT real-time budget — once the audio queue backs up, local-agreement
never re-stabilises and commits stop.

`scripts/separation_bridge.py` now defaults `device="cpu"` and reads
`SEPFORMER_DEVICE` from the environment for users with a dedicated
GPU. Libri2mix on CPU runs ~150–250 ms per 500 ms chunk — well under
cadence — and isolates Sepformer from GPU pressure entirely.

This is a configuration default, not a hardware requirement: anyone
can flip back to CUDA via `SEPFORMER_DEVICE=cuda:0` once the GPU
budget is no longer shared.

## Rollback

Revert the call to `tracker.align(...)` and restore the direct
`ch_a_tx.send` / `ch_b_tx.send` block. Tracker code stays in the
crate — unused but harmless.
