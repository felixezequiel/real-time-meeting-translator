# ADR 0011 — OpenVoice v2 Tone Color Conversion

**Status:** Accepted
**Date:** 2026-04-27

## Context

After ADRs 0009 (Qwen streaming MT) and 0010 (Kokoro TTS), the remaining
gap between the synthetic pipeline output and the real meeting was
**voice identity**: documentaries with multiple participants of the
same gender ended up sounding alike because Kokoro draws from a fixed
catalog of ~9 neural voices. The voice-routing layer (TTS bridge) plus
WORLD pitch shifting could distinguish men from women and broadly high
voices from low ones, but two adult men in the same panel still
sounded like the same Kokoro voice — only the F0 changed, not the
**timbre** (vocal-tract resonance, formants, glottal source colour).

Earlier attempts at differentiation:

- **WORLD formant warping** (ADR 0006). Disabled because even ±15 %
  spectral-envelope shifts produced audible analysis-synthesis
  artefacts ("robotic colour"). `formant_shift_for_f0` now always
  returns 1.0 — see `crates/pipeline/src/lib.rs`.
- **Catalog expansion + sticky routing**. Helped when the meeting had
  ≤ 3 same-gender participants; broke down for documentaries.
- **CosyVoice 2 zero-shot cloning**. Worked but synth latency was
  2–4 s per fragment on the RTX 3050 — outside the simultaneous
  interpreter budget.

We needed a path that preserves Kokoro's prosody and latency (the
~80–150 ms / fragment we already pay) but rewrites the timbre to
match the actual speaker.

## Decision

Add an optional **OpenVoice v2 Tone Color Converter (TCC)** stage
between Kokoro's output and the playback mixer. TCC is a learned
flow model trained on multilingual speech that takes a synthetic
audio sample plus a 5–10 s reference WAV from the target speaker
and rewrites the spectral envelope dynamics onto the synthetic audio
without changing words or rhythm.

### Architecture

```
Kokoro fragment ──┐
                  ├─► [OpenVoice TCC] ──► resampler ──► mixer
ref_<speaker>.wav ┘    (Python bridge)
        ▲
        │
[Auto-enrolment]
        │
Live audio chunks (≥ 6 s clean speech, RMS > 0.015 per speaker)
```

Components added:

1. `scripts/voice_convert_bridge.py` — persistent Python subprocess
   loading OpenVoice from `third_party/OpenVoice/` and the TCC
   checkpoint from `models/openvoice/converter/`. Binary-framed
   protocol mirroring the diarization / TTS bridges. Caches the
   per-reference target SE and a single shared source SE for the
   Kokoro domain.
2. `crates/voice_convert` — Rust client (`ToneColorConverter`) plus
   `PipelineStage` lifecycle. One `AtomicBool` `dead` flag flips on
   any conversion failure; subsequent calls return `Ok(None)` so the
   pipeline degrades gracefully to raw Kokoro instead of stalling.
3. `VoiceProfileRegistry` enrolment (in `crates/pipeline/src/lib.rs`).
   Per pipeline branch, accumulates `REFERENCE_ENROLL_SECONDS` (= 6 s)
   of clean speech per `speaker_id`, gated by
   `REFERENCE_INGEST_MIN_RMS` to reject silence and music. On reaching
   the buffer target we flush a mono 16-bit PCM WAV to the temp dir
   and the OpenVoice bridge starts using it.
4. `enable_voice_conversion: bool` config flag (default `true`) in
   `crates/shared/src/config.rs`. Off-switch for users on machines
   without the OpenVoice repo / checkpoint vendored.
5. `Install-OpenVoice` step in `scripts/install.ps1` and a block of
   runtime deps in `scripts/requirements.txt` (librosa, soundfile,
   pydub, inflect, unidecode, eng-to-ipa, pypinyin, cn2an, jieba,
   wavmark).

### Trade-offs

| Concern              | Result                                         |
|----------------------|------------------------------------------------|
| Quality              | Substantially better — captures detailed envelope dynamics that WORLD cannot. |
| Latency              | +150–250 ms per fragment on GPU. Inside budget. |
| GPU memory           | ~150 MB for the TCC + ~150 MB for the SE extractor. Co-exists with Whisper + Qwen + Kokoro on the RTX 3050. |
| Disk                 | ~50 MB checkpoint + repo (~30 MB).             |
| Failure mode         | Bridge dead → `vc.convert` returns `Ok(None)` → pipeline plays raw Kokoro output unchanged. Never crashes. |
| Cold start           | First conversion pays SE extraction (~150 ms on GPU). Cached for the session. |

## Consequences

### Positive

- Two same-gender speakers in a documentary now sound like distinct
  individuals — the original goal.
- Per-speaker enrolment is automatic: no UI prompt, no pre-recorded
  references. The first ~6 s of clean speech the diariser attributes
  to a speaker becomes their reference.
- Stage is fully opt-out: setting `enable_voice_conversion = false`
  returns the pipeline to its pre-ADR-0011 behaviour.

### Negative

- One more Python subprocess (now four: STT, translation, TTS,
  diarization, separation when on, voice-convert when on).
- OpenVoice's repo is unmaintained (last commit 2024). The pinned
  runtime deps may drift; the pip-install fallback in
  `Install-PythonDeps` is the safety net.
- Reference-WAV enrolment requires diarisation to attribute chunks
  correctly. A speaker the diariser never identifies (silence-only
  channel, very short interjections) never enrols and falls through
  to raw Kokoro — acceptable but worth knowing.

### Neutral

- Pipeline still runs at zero recurring cost, fully local, on the
  RTX 3050 6 GB target.
- The WORLD pitch-shift path (ADR 0006) stays in place beneath TCC:
  Kokoro's output already has the right F0 from voice routing, and
  TCC inherits it.

## Alternatives considered

- **Coqui XTTS v2**. Higher quality but ~30 % slower and uses ~2 GB
  VRAM — would push the 3050 over its budget when Whisper + Qwen are
  resident.
- **Real-time RVC (Retrieval-based Voice Conversion)**. Faster but
  requires per-speaker model training (10+ minutes per speaker) — not
  compatible with auto-enrolment from live audio.
- **Tortoise / Bark voice cloning**. Quality is excellent; latency is
  multiple seconds per fragment — incompatible with simultaneous
  interpretation.

## References

- OpenVoice v2 paper: https://arxiv.org/abs/2312.01479
- OpenVoice repo: https://github.com/myshell-ai/OpenVoice
- HF checkpoint: https://huggingface.co/myshell-ai/OpenVoiceV2
- Bridge: `scripts/voice_convert_bridge.py`
- Rust client: `crates/voice_convert/src/lib.rs`
- Pipeline integration: `crates/pipeline/src/lib.rs`
  (`VoiceProfileRegistry::ingest_audio`, TTS thread VC call site)
