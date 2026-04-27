# ADR 0008 — Silero VAD substitui o RMS energy gate

- **Status:** Accepted
- **Date:** 2026-04-27
- **Deciders:** felix
- **Related:** ADR 0004 (streaming STT), ADR 0007 (source separation)

## Context

Two upstream gates currently decide whether a chunk reaches Whisper:

1. `MIN_RMS_FOR_STT = 0.003` in `crates/pipeline/src/lib.rs` — rejects
   chunks whose RMS energy is below the threshold.
2. `EnergyVad` in `crates/audio/src/vad.rs` — same idea, also energy-based,
   but unused by the live pipeline (only present as a library helper).

Energy gates have two well-known failure modes that we hit in practice:

- **False positives.** Steady noise (air conditioner, paper rustling,
  keyboard, breath into the mic) trips the gate and is fed to Whisper.
  Whisper at temperature-0 is allowed to hallucinate plausible text on
  ambiguous input — those hallucinations then contaminate the
  accumulator and produce nonsense translations. The `is_repetitive()`
  guard in `crates/stt/src/lib.rs:147` exists specifically as a
  downstream patch for this.
- **False negatives.** Soft-spoken speech, or any voice slightly below
  the threshold, gets dropped silently. The user has to repeat. This
  also feeds back into the streaming session: when the gate sends an
  empty chunk, `StreamingSession::flush_tentative()` runs, which can
  prematurely commit a fragment that was still mid-thought.

The energy gate cannot tell speech from non-speech of similar volume.
A neural VAD can.

## Decision

Replace both energy gates with **Silero VAD** (snakers4/silero-vad v5),
loaded via the `voice_activity_detector` crate which embeds the ONNX
model and the ORT bindings together.

- Model: Silero v5, embedded in the Rust binary at compile time.
- Input: 16 kHz mono PCM, processed in 512-sample frames (32 ms).
- Output: per-frame speech probability ∈ [0, 1]. The model is
  stateful (LSTM-based); the crate keeps the recurrent state across
  calls so context flows between adjacent chunks.
- Decision rule: a chunk has speech when **any frame in the chunk
  exceeds the threshold** (default 0.5). This matches Silero's
  reference implementation and is more permissive than averaging,
  which is the right behaviour for a binary "feed to Whisper or not"
  decision — half a syllable at the chunk boundary should still pass.

The new VAD lives in `crates/audio/src/silero_vad.rs` and is wired into
`SpeakerPipeline` as `Arc<SileroVad>`. The energy `MIN_RMS_FOR_STT`
constant is removed; the inline RMS check in the audio_input branch of
the `tokio::select!` loop is replaced by a call to `vad.has_speech()`.

`EnergyVad` stays in `crates/audio/src/vad.rs` as a fallback path used
when the embedded Silero detector fails to initialise (rare —
typically only an ORT init failure on an unusual driver setup). The
fallback emits a `tracing::warn!` so the regression is visible.

### Why the dedicated crate instead of calling ORT directly

The first attempt invoked ORT directly via the `ort` crate. Despite
input/output names matching the spec (`input/state/sr` →
`output/stateN`) and shapes being correct (`(1, 512)` /
`(2, 1, 128)`), the model returned probabilities pinned around
0.0006–0.11 even on clear human speech. We tried `sr` as a 1-D tensor
of shape `(1,)`, then as a rank-0 scalar; tried `TensorRef::from_array_view`
on owned and viewed `ndarray` instances; verified the state input/output
plumbing. None of these moved the probability into a usable range.
Rather than continue debugging a low-level ORT integration, we adopted
the maintained `voice_activity_detector` crate — its own integration
tests validate against real speech, so probabilities now behave as
documented (>0.5 on clear speech, <0.05 on silence).

## Alternatives considered

1. **Whisper's own no-speech probability (`no_speech_thold`).**
   Already used inside `WhisperStt` (set to 0.5). Not a substitute —
   it runs *after* the expensive Whisper inference, so it filters
   the output but doesn't save the GPU pass. We need a gate that
   runs before Whisper.
2. **Energy + zero-crossing-rate heuristic.** Better than pure RMS,
   still fails on tonal noise (fans, music humming). Marginal gain
   for a fixed-quality ceiling; not worth the new code over a
   strictly better neural model.
3. **WebRTC VAD (libfvad).** Older, energy+spectral hybrid. Faster
   than Silero (<1 ms) but less accurate on far-field / soft speech.
   We can spare the ~2 ms.
4. **Pyannote VAD (segmentation-3.0).** Higher quality than Silero
   but Hugging Face gated, larger (~17 MB), and would need a Python
   bridge. We've worked hard to push bridges out of the audio path
   (ADR 0001 → CT2, native STT). Reintroducing one for VAD goes the
   wrong direction.
5. **Keep energy gate, raise threshold.** Treats the symptom; the
   false-negative half (soft speech dropped) gets worse.

## Consequences

### Positive
- Cleaner Whisper input → fewer hallucinations → shorter `is_repetitive`
  filter chain, more stable streaming session prefixes.
- Robust to ambient noise that the energy gate currently accepts.
- Chunk size can drop (500 ms → ~280 ms, see follow-up change) without
  proportionally raising GPU cost: most chunks become VAD no-ops.
- All-Rust audio path between capture and Whisper. No new bridge.

### Negative
- ~1.8 MB of model weights embedded in the binary (raises the exe
  size accordingly; offsets the install-time download cost).
- New crate dependency: `voice_activity_detector = "0.2"` (which
  pulls in `ort 2.0.0-rc.10` transitively and ships the ONNX
  Runtime native library via its `download-binaries` feature).
- Stateful model means concurrent calls need a Mutex. Both pipelines
  (speaker + mic) get their own `SileroVad` instance to keep state
  per-stream — sharing one would mix contexts.

### Neutral
- The constant `MIN_RMS_FOR_STT` and the inline RMS computation in
  `crates/pipeline/src/lib.rs` go away.
- Tests for the new module mirror the pattern in `vad.rs`: synthetic
  silence (must reject), synthetic 200 Hz tone in speech band (must
  accept), zero-length input (must reject without panicking).

## Rollout

1. Add `voice_activity_detector = "0.2"` to `crates/audio/Cargo.toml`.
2. New file `crates/audio/src/silero_vad.rs` exporting `SileroVad`.
3. Re-export from `crates/audio/src/lib.rs`.
4. `SpeakerPipeline::with_vad` takes `Arc<SileroVad>`; `main.rs`
   constructs one per pipeline branch via `try_create_silero_vad`.
5. Replace inline RMS gate with `vad.has_speech(&chunk.samples)`.
6. Remove `MIN_RMS_FOR_STT` constant.

## Rollback

Remove the `with_vad(...)` call in `start_pipelines` (or have
`try_create_silero_vad` always return `None`). The pipeline falls
back to its built-in RMS energy gate.

## How we'll know it worked

The follow-up "stage metrics" change adds P50/P95 for each stage
including a new `vad` stage. After rollout we expect:

- `vad` stage: P50 ~1–2 ms, P95 ~3 ms.
- `stt` stage: P50 unchanged or slightly lower (fewer Whisper calls
  on garbage chunks).
- `total` end-to-end: lower P95 because the `is_repetitive` /
  hallucination retry path fires less often.
- Subjective: soft-spoken speech no longer dropped; ambient-noise
  hallucinations no longer reach the translator.
