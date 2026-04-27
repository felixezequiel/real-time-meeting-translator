# ADR 0005 — Diarisation embeddings via SpeechBrain ECAPA-TDNN

- **Status:** Accepted
- **Date:** 2026-04-26
- **Deciders:** felix
- **Related:** ADR 0003 (CosyVoice TTS), ADR 0004 (streaming STT)

## Context

Online speaker diarisation has two jobs in this pipeline:

1. **Attribute audio to speakers** so per-speaker reference WAVs can
   be auto-enrolled and CosyVoice clones the right voice.
2. **Drive flush decisions**: when a different person starts talking
   the accumulator should flush so the previous sentence is rendered
   in the previous voice.

The previous in-progress design used **Resemblyzer** (`VoiceEncoder`,
256-d LSTM, ~50 MB). Its `MATCH_THRESHOLD` had to sit at 0.75 because
intra/inter-speaker similarity distributions overlapped; even there,
short windows (sub-second) routinely produced false speaker changes
that flushed the accumulator mid-sentence — one of the root causes of
the choppy/garbled output the user reported.

## Decision

Replace Resemblyzer with **`speechbrain/spkrec-ecapa-voxceleb`**, the
ECAPA-TDNN model (192-d embedding) from SpeechBrain. Open weights, no
HuggingFace authentication required, ~22 MB on disk, ~30–40 ms per
embedding on CPU.

Pair it with **speaker-change confidence smoothing** in the pipeline:
a candidate new speaker_id must be observed on
`MIN_CHUNKS_FOR_SPEAKER_CHANGE` (=2) consecutive chunks before the
pipeline accepts the change and triggers a flush. Single-chunk wobble
(brief noise, similar voices on a low-confidence frame) is suppressed.

- Bridge: `scripts/diarization_bridge.py` (rewritten — same Rust-side
  protocol as before, only the embedding model and clustering
  thresholds change).
- Cosine threshold: `MATCH_THRESHOLD = 0.55` (down from 0.75 with
  Resemblyzer). Empirically, ECAPA's intra-speaker similarities sit at
  0.65–0.85 and inter-speaker at 0.10–0.45, so 0.55 separates them
  cleanly.
- `MIN_SAMPLES_FOR_EMBED` shrunk from 0.75 s to 0.6 s — ECAPA tolerates
  shorter windows than Resemblyzer.
- `MIN_RMS_FOR_EMBED = 0.005` rejects silence/background noise that
  used to spawn spurious "new speaker" hits in Resemblyzer.

Speaker-change smoothing is implemented in `start_stt_worker` in
`crates/pipeline/src/lib.rs` so it sits next to the streaming STT
state and shares the same chunk-by-chunk loop.

## Alternatives considered

1. **`pyannote/embedding`.** Higher quality than ECAPA on some
   benchmarks but gated on HuggingFace (requires user-token). Rejected
   because the project goal is "100% local, no API keys".
2. **DiArt (pyannote-streaming).** Full pipeline including VAD +
   clustering. Rejected because we already have an RMS gate, energy
   detection in the pipeline, and our own running-mean clustering;
   adopting DiArt's full pipeline would conflict with those.
3. **3D-Speaker (Alibaba).** Open weights, similar quality to ECAPA.
   Rejected because SpeechBrain's loader is more mature on Windows
   and the embedding quality difference is within noise.
4. **Stick with Resemblyzer, just add confidence smoothing.** The
   smoothing alone hides single-chunk wobble but doesn't fix cases
   where two speakers genuinely sound similar — Resemblyzer
   conflates them under cosine 0.75 regardless of how many chunks
   you average.

## Consequences

### Positive
- Cleaner separation between speakers — the threshold has visible
  margin instead of sitting on the boundary.
- Confidence smoothing turns the remaining single-chunk noise into a
  quiet "wait one chunk" instead of a destructive mid-sentence flush.
- Open weights keep the project's offline guarantee intact.

### Negative
- ~22 MB cache on disk + ~50 MB extra Python deps (speechbrain has a
  larger import surface than resemblyzer).
- Cold start: SpeechBrain downloads the model on first use unless
  `Install-DiarizationModel` in `scripts/install.ps1` ran first.

### Neutral
- The Rust-side `OnlineDiarizer` and JSON protocol are unchanged. Only
  the bridge implementation and threshold constants moved.

## Rollout

1. Rewrite `scripts/diarization_bridge.py` to load ECAPA-TDNN.
2. Update `scripts/requirements.txt` — drop `resemblyzer`, add
   `speechbrain`.
3. Add `Install-DiarizationModel` to `scripts/install.ps1` so the
   model is pre-downloaded; otherwise the first session pays a
   ~5–10 s wait while the bridge stalls before `ready`.
4. Add speaker-change confidence smoothing to
   `crates/pipeline/src/lib.rs` (`MIN_CHUNKS_FOR_SPEAKER_CHANGE`,
   `pending_speaker_id`, `pending_count`).

## Rollback

Restore the Resemblyzer-based bridge from git and revert the
threshold/smoothing changes in `crates/pipeline/src/lib.rs`.
`crates/diarization/` Rust code is untouched.
