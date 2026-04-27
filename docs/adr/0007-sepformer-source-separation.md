# ADR 0007 — Source separation via Sepformer (opt-in)

- **Status:** Accepted (opt-in)
- **Date:** 2026-04-27
- **Deciders:** felix
- **Related:** ADR 0006 (Piper TTS), ADR 0005 (ECAPA diarisation)

## Context

The loopback capture is mono — when two participants in a meeting
speak at the same time, Whisper sees the mixed audio and produces a
"salad" transcript that's neither speaker's words. Real-world example:

  Speaker A: "Meu objetivo nessa conversa é..."
  Speaker B (interrupts): "Pera, quê?"
  Speaker A: "...entender o roadmap"

Without separation, Whisper might transcribe roughly "meu objetivo
nessa conversa é pera quê entender o roadmap" — useless. The user
explicitly flagged this as important: interruptions are common in
their meetings and must not be silently dropped or fused into the
primary speaker's translation.

## Decision

Add **SpeechBrain Sepformer (libri2mix)** as an opt-in stage between
the loopback capture and the speaker pipeline. When enabled by
`enable_separation: true` in `config.toml`, each ~500 ms loopback
chunk is fed through Sepformer and split into two channel streams
(channel A and channel B). The speaker pipeline is then duplicated:
two `SpeakerPipeline` instances run in parallel, each consuming one
channel, both feeding their TTS output into the same headphones
mixer.

Default is **off** because the cost-benefit changes by use case: for
1-on-1 conversations with no overlap, the ~50–80 ms / chunk Sepformer
runtime and ~120 MB GPU/RAM is wasted. For multi-person meetings
where interruptions matter, it's the only way to keep both voices.

The duplicated pipeline branches share:
- The `OnlineDiarizer` (ECAPA + F0) — speaker_id and F0 readings on
  separated audio are far cleaner than on the original mix, so
  diarisation quality actually improves with separation on.
- The translator + TTS instances (Piper bridges; concurrency is
  handled by the existing in-flight-task semaphore).
- The echo buffer (cross-pipeline echo detection still works).
- The headphones mixer / playback.

When only one speaker is active, Sepformer still returns two
channels, but one is near-silent. The pipeline gates on per-channel
RMS (≥0.01) and skips empty channels — single-speaker chunks pay the
~50 ms separation cost but produce only one downstream STT call.

## Alternatives considered

1. **Always-on separation, always two pipelines.** Same cost as
   opt-in, but penalises 1-on-1 use. Opt-in is strictly better.
2. **Detect overlap first, separate only when overlapping.** Needs
   a reliable overlap-detection model — pyannote/segmentation-3.0 is
   gated, ECAPA-distance-based detection has too many false positives.
   Opt-in single-flag separation is simpler and gives equivalent
   end-user behaviour.
3. **Beamforming on a stereo capture.** Not applicable — loopback
   is single-channel.
4. **Manual VAD-based time slicing.** Doesn't help when speakers
   genuinely overlap (the case we care about).

## Consequences

### Positive
- Interruption-tolerant: each speaker's words are heard intact even
  when their audio overlaps in the meeting.
- Diarisation quality improves on separated audio (one speaker per
  channel → cleaner ECAPA embeddings → fewer single-chunk
  misclassifications).
- Modular: when off, the pipeline is exactly as before — no risk
  added by the existence of this code path.

### Negative
- ~120 MB of weights to download (libri2mix Sepformer).
- ~50–80 ms per chunk on GPU, ~150–250 ms on CPU. The pipeline's
  500 ms chunk cadence absorbs that on GPU; on CPU the bridge can
  fall behind on a fast talker.
- Sepformer is trained for exactly two simultaneous speakers. Three
  or more concurrent speakers degrade quality — channel attribution
  may flip mid-chunk.
- Output ordering across the two parallel pipelines isn't strictly
  preserved by timestamp — the mixer plays in arrival order, so a
  fast TTS for channel B may arrive before a slow TTS for channel A
  even when A spoke first. Acceptable for the "interruption" use
  case but not perfect.

### Neutral
- A new Rust crate `crates/separation/` mirrors the bridge-client
  pattern of `crates/diarization/` and `crates/tts/`.

## Rollout

1. Add `scripts/separation_bridge.py` running Sepformer-libri2mix.
2. New `crates/separation/` crate (Rust client). Add to workspace.
3. Add `enable_separation: bool` to `PipelineConfig`, default false.
4. `LoadedModels.separator: Option<Arc<Sepformer>>` populated only
   when the flag is on.
5. `start_pipelines` builds either one or two speaker branches based
   on `models.separator`. Both branches share `spk_out_tx`.
6. `ActivePipelines.speaker_cmd_tx` becomes `speaker_cmd_txs:
   Vec<Sender>` so Start/Stop reaches every branch.
7. New `start_separation_worker` in `main.rs` spawns the bridge
   forwarder.
8. `Install-SeparationModel` in `install.ps1` pre-downloads the
   weights so the first runtime use isn't a multi-second stall.

## Rollback

Set `enable_separation = false` in `config.toml` and restart. The
single-pipeline path is the default; the dual-pipeline path is dead
code in that mode (the `Sepformer` instance isn't even created).
