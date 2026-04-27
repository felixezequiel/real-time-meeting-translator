# ADR 0003 — TTS migration to CosyVoice 2 (zero-shot voice cloning)

- **Status:** **Superseded by ADR 0006 (2026-04-27).** CosyVoice 2-0.5B
  was unable to hit the simultaneous-translation latency budget on a
  RTX 3050 6 GB laptop GPU. End-to-end TTS time was 2-4 seconds per
  utterance (RTF ~0.5–1.0 even fully optimised), against a target of
  <1 s. The replacement (Piper + pyworld pitch/formant shift, ADR
  0006) hits ~250 ms per utterance with per-speaker voice
  differentiation via DSP rather than cloning.
- **Date:** 2026-04-26
- **Deciders:** felix
- **Supersedes:** an unversioned in-progress design that paired Piper TTS
  with a separate KNN-VC voice-conversion stage.
- **Related:** ADR 0001 (translation), ADR 0002 (audio ducking)

## Context

The previous in-progress design synthesised speech with Piper (a single
generic neural voice per language) and then post-processed it through
KNN-VC, a nearest-neighbour search over WavLM features that bent the
phonemes towards a per-speaker reference WAV. Two systemic problems
emerged in testing:

1. **Latency.** KNN-VC added ~500 ms warm per utterance, plus ~150 ms
   for the first `get_matching_set` call per reference. Combined with
   the existing chunk-based STT and accumulator, end-to-end latency
   landed at 2.5–3 s — past the 2-second target.
2. **Quality ("voz embaralhada").** KNN-VC quality is bounded by how
   well the reference WAV covers the phoneme space. With ~10 s of
   auto-enrolled audio, uncovered phonemes (rare consonants, stressed
   diphthongs) snap to the nearest available neighbour — frequently a
   different phoneme — producing the garbled "fanha" artefact the user
   reported.

## Decision

Replace the Piper + KNN-VC two-stage pipeline with **CosyVoice 2-0.5B**,
a streaming TTS model with built-in zero-shot voice cloning. The model
generates phonemes directly conditioned on the prompt speech, so there
is no "uncovered phoneme" failure mode by construction.

- Bridge: `scripts/tts_bridge.py` (rewritten).
- Rust client: `crates/tts/src/lib.rs` exports `CosyVoiceTts`,
  drop-in replacement for the old `PiperTts` with one extra parameter
  on `synthesize` — the per-speaker reference WAV path resolved by
  `SpeakerRegistry`.
- Reference enrolment: `SpeakerRegistry` now writes 6 s WAVs (down from
  10 s); CosyVoice's prompt encoder is trained on 3–10 s clips and 6 s
  is the empirical sweet spot.
- Built-in fallback: when no reference is available yet (very first
  utterance of a session), CosyVoice uses its bundled `inference_sft`
  voice catalogue — the user hears a generic voice rather than silence.

## Alternatives considered

1. **F5-TTS / E2-TTS.** Simpler install, but autoregressive — latency
   scales with output length and PT support is community-maintained.
   Rejected for unknown PT-BR quality at the cadence we need.
2. **XTTS-v2 (Coqui).** Mature, 200 ms TTFT, PT supported. Rejected
   because Coqui's package is unmaintained and CosyVoice 2 is roughly
   30% faster on the same hardware.
3. **Voxtral TTS (Mistral, 2026).** State of the art on latency
   (70 ms) but the 4B model is too heavy for a 6 GB laptop GPU shared
   with Whisper + CosyVoice + ECAPA.
4. **Piper multi-speaker (no cloning).** Cheapest option — assigns a
   distinct stock voice per diarised speaker. Considered, but the user
   explicitly chose route C (real cloning) over route A (multi-voice
   tagging).
5. **Keep KNN-VC, replace with RT-VC / SynthVC.** Cuts ~400 ms but
   doesn't fix the "phoneme not in reference" failure mode — same
   class of artefact, smaller window of pain.

## Consequences

### Positive
- Single model owns synthesis + cloning — no pipeline-level routing
  between two Python bridges, fewer round-trips.
- Streaming output: ~150 ms TTFA from CosyVoice's internal token
  streaming, vs ~300–500 ms for Piper's whole-utterance synthesis.
- No "uncovered phoneme" artefact — the model generates phonemes
  directly in the cloned voice.
- Multilingual: the same model handles PT and EN; we no longer need
  per-language voice files (Piper had separate `faber-medium` and
  `ryan-medium`).

### Negative
- ~2 GB of model weights on disk (vs ~50 MB for Piper + ~1.2 GB for
  KNN-VC). Net storage actually goes down because we drop KNN-VC.
- ~1.5 GB VRAM at fp16. Tight on a 6 GB laptop GPU shared with
  Whisper (~600 MB) and NLLB-CT2 (~600 MB) — leaves ~3 GB headroom.
- CosyVoice's PyPI package is unstable; install pulls the upstream
  repo into `third_party/CosyVoice/` and adds it to `sys.path`. Done
  by `Install-CosyVoice` in `scripts/install.ps1`.

### Neutral
- The bridge protocol is unchanged from the Rust side: same JSON
  header + binary PCM framing as the previous TTS bridge. Only one
  extra optional field (`ref_wav_path`).

## Rollout

1. Delete `crates/voice_convert/` (Rust client of KNN-VC).
2. Delete `scripts/voice_convert_bridge.py`.
3. Rewrite `scripts/tts_bridge.py` to load CosyVoice 2 from
   `models/CosyVoice2-0.5B/`.
4. Rewrite `crates/tts/src/lib.rs` — `CosyVoiceTts` with
   `synthesize(segment, ref_wav_path)`.
5. Update `crates/pipeline/src/lib.rs` translate worker — drop the
   VC step, pass `ref_wav` to `tts.synthesize` directly.
6. Update `scripts/install.ps1` — add `Install-CosyVoice` (clones repo,
   downloads weights via `huggingface_hub`).
7. Update `scripts/requirements.txt` — drop `piper-tts`, `resemblyzer`,
   `einops` already kept for cosyvoice; add `hyperpyyaml`,
   `omegaconf`, `modelscope`, etc.

## Rollback

The two stages this ADR replaces are fully removed from the workspace.
Rolling back means restoring `crates/voice_convert/` and the previous
`tts_bridge.py` from git. Reference WAVs written by `SpeakerRegistry`
work for both pipelines unchanged — same WAV format, same paths.
