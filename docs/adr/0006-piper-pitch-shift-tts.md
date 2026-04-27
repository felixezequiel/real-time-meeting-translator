# ADR 0006 — TTS via Piper + pyworld pitch/formant shifting

- **Status:** Accepted
- **Date:** 2026-04-27
- **Deciders:** felix
- **Supersedes:** ADR 0003 (CosyVoice 2 zero-shot cloning)
- **Related:** ADR 0005 (ECAPA diarisation), ADR 0007 (Sepformer separation)

## Context

ADR 0003 chose CosyVoice 2-0.5B to give every translated utterance the
voice of the original speaker, via the model's zero-shot cloning path.
That decision didn't survive contact with the hardware budget:

- CosyVoice 2-0.5B inference on a RTX 3050 6 GB laptop GPU runs at
  RTF 0.5–1.0 (1–4 seconds of compute per utterance) even with fp16,
  SDPA attention, and the CUDA-accelerated ONNX paths enabled.
- That alone violates the project's 2-second end-to-end target, before
  Whisper, NLLB, audio capture, or playback are factored in.
- The per-speaker reference WAV path required either an awkward
  bootstrap clip (we tried `cross_lingual_prompt.wav`, a Chinese-
  accented sample shipped with CosyVoice) or a multi-second wait to
  enrol the user's own voice. Both caused obvious artefacts the user
  observed in testing ("começou inglês, virou português").

The original analysis (Plan A from the conversation that produced this
project) had pre-empted this risk: voice differentiation can be
achieved with **DSP**, not a cloning model. This ADR finally adopts
that path.

## Decision

Replace the CosyVoice 2 zero-shot cloning pipeline with **Piper TTS +
pyworld analysis-synthesis pitch/formant shifting**:

1. **Piper** synthesises the translated text in a fixed voice per
   language (Faber-medium for pt-BR, Ryan-medium for en-US). ~150 ms
   per utterance, ONNX-only, runs entirely on CPU. ~25 MB per voice.

2. **pyworld** WORLD analysis-synthesis transforms the Piper output:
   - Extract F0 (fundamental frequency / pitch), spectral envelope,
     and aperiodicity from the synthesised audio.
   - Replace F0 with the **target speaker's running F0 mean** (from
     the diarisation bridge — see ADR 0005).
   - Optionally warp the spectral envelope along the frequency axis
     for a "vocal weight" shift (formant shift).
   - Resynthesise. ~80–150 ms per call.

3. **Per-speaker F0 tracking** lives in `crates/pipeline`'s
   `VoiceProfileRegistry`: a running mean per speaker_id updated each
   chunk by the diarisation bridge, with values clamped to [70, 400]
   Hz to reject pyworld noise on near-silent audio.

The diarisation bridge (`scripts/diarization_bridge.py`) was extended
to return F0 alongside speaker_id — both come from per-chunk audio,
both run on CPU, ~30–50 ms total per chunk.

## Alternatives considered

1. **Continue optimising CosyVoice 2.** Even with cuDNN, SDPA,
   and a tight model.load wrapper, the dominant cost (~2–3 s) is the
   Qwen LLM autoregressive decode. The hardware ceiling on a 3050 is
   ~RTF 0.3 best case — still too slow for <1 s utterance turnaround.
2. **Smaller cloning model (Voxtral-style 100 M-class).** None
   currently available with PT-BR support that we control offline.
3. **Plain Piper, no shifting (Plan A from the original analysis).**
   Same speed but loses the per-speaker voice-differentiation cue.
   The shift is cheap enough (<150 ms) that there's no reason to
   skip it.
4. **RVC-style retrieval voice conversion.** Needs per-speaker
   training data — incompatible with the "zero setup" goal.

## Consequences

### Positive
- TTS step drops from 2–4 s to ~250 ms — comfortably inside the live
  translation budget.
- No GPU needed for TTS at all (CPU is plenty for Piper + pyworld at
  500 ms-chunk cadence). Frees ~1.5 GB VRAM that CosyVoice consumed,
  leaves headroom for Sepformer and other future additions.
- Voice differentiation is automatic per speaker via the F0 running
  mean — no enrolment, no reference WAV, no bootstrap clip.
- Bridge protocol is simple: `(text, language, target_f0,
  formant_shift)` → PCM. Works the same with or without F0 (zero F0
  skips the analysis-synthesis pass).

### Negative
- The output voice does not actually sound like the original speaker.
  It sounds like Piper, pitched up or down to match the speaker's
  pitch profile. For "tell who's speaking" UX this is enough; for
  "this sounds exactly like Alice" UX it isn't.
- Two speakers with very similar F0 (e.g. two adult men at 110 Hz)
  will sound the same. The mitigation (formant shift) helps a little
  but is conservative (clamped to 0.85–1.15) to avoid artefacts.
- pyworld DIO occasionally produces noisy F0 readings on consonant-
  heavy or breath-heavy audio. Smoothed by the running mean, but a
  speaker who just whispered for 3 s might get a brief pitch wobble.

### Neutral
- Same Piper voice files (~50 MB total) we used pre-CosyVoice. The
  install path was reverted to download Faber + Ryan; CosyVoice
  weights and the cloned `third_party/CosyVoice` repo are gone.

## Rollout

1. Delete `models/CosyVoice2-0.5B/` and `third_party/CosyVoice/`.
2. Rewrite `scripts/tts_bridge.py` for Piper + pyworld.
3. Rename `CosyVoiceTts` → `PiperTts` in `crates/tts/src/lib.rs`,
   replace the `ref_wav_path` parameter with `VoiceProfile {
   target_f0_hz, formant_shift }`.
4. Extend `scripts/diarization_bridge.py` to return per-chunk F0.
5. Add `f0_hz` field to `SpeakerIdentification` in
   `crates/diarization/src/lib.rs`.
6. Replace `SpeakerRegistry` (file-based) with `VoiceProfileRegistry`
   (running F0 mean) in `crates/pipeline/src/lib.rs`.
7. Map running F0 → `formant_shift` heuristically (higher F0 →
   slightly narrower vocal tract → formant_shift < 1).
8. Update `scripts/install.ps1`: drop `Install-CosyVoice` /
   `Install-TorchCuda`, add `Install-PiperVoices`.

## Rollback

Restore CosyVoice from git history (`scripts/tts_bridge.py`,
`crates/tts/src/lib.rs`, `models/CosyVoice2-0.5B/` re-download). The
new `VoiceProfile` parameter is additive on the bridge protocol;
ignoring it in tts_bridge.py would degrade gracefully to
unaccompanied Piper output.
