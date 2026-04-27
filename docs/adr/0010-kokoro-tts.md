# ADR 0010 — Kokoro v1.0 TTS (replaces Piper)

- **Status:** Accepted
- **Date:** 2026-04-27
- **Deciders:** felix
- **Supersedes:** ADR 0006 (Piper + pyworld pitch shift) — only the TTS
  backend changes; pyworld pitch/formant shifting on top stays.
- **Related:** ADR 0006 (pitch shift mechanism), ADR 0009 (streaming MT)

## Context

ADR 0006 chose Piper as the local TTS backend. It worked: ~150 ms per
utterance on CPU, enough for sub-2-second end-to-end translation. But
the output sounded clearly synthetic — robotic colour on every
utterance, especially noticeable when pyworld shifted the pitch up
for high-F0 speakers.

The friend's voiceMaster spike runs **Kokoro v1.0** locally on GPU.
Listening to its samples side-by-side with Piper made the gap
obvious: Kokoro has substantially more natural prosody, better
breathing pauses, and the voice library covers PT-BR with `pm_alex`
and (in some snapshots) `pf_dora`. It's the same compute footprint
class — both are local ONNX models — just newer and bigger.

## Decision

Replace Piper with **Kokoro v1.0** (~82M params, ONNX, 24 kHz output)
as the TTS backend, keeping the pyworld pitch/formant shifter on top
unchanged.

The Rust client (`crates/tts/src/lib.rs`) keeps its `PiperTts` struct
name and protocol (`{text, language, target_f0, formant_shift}` →
binary PCM). Only the Python bridge implementation swaps. The
struct's name is misleading now but renaming would churn every
downstream import for no functional gain — a doc note covers it.

The bridge:
1. Loads `kokoro-v1.0.onnx` + `voices-v1.0.bin` (~330 MB total) from
   `models/kokoro/`.
2. Picks an ONNX provider via `kokoro-onnx` (`onnxruntime-gpu` is
   preferred when installed; CPU fallback works).
3. At synthesis time, picks the voice closest to `target_f0` from the
   per-language list (e.g. EN: `am_michael` ~120 Hz, `af_bella`
   ~210 Hz). This minimises the residual pitch shift pyworld has to
   apply.
4. Runs Kokoro `.create(text, voice, lang, speed=1.0)`.
5. Optionally pipes the output through pyworld pitch+formant shift
   (same as before).

## Why Kokoro

| Candidate | Params | Latency | VRAM | PT support | Streaming |
|---|---|---|---|---|---|
| **Kokoro v1.0** | 82M | 100–200 ms (GPU) | ~300 MB | yes (pm_alex, pf_dora) | non-native, batched |
| Piper (previous) | ~25M | ~150 ms (CPU) | 0 (CPU) | yes (faber, cori) | native chunked |
| StyleTTS2 | 100M+ | 200–400 ms | ~500 MB | limited | partial |
| F5-TTS | 300M+ | 500–800 ms | ~1.5 GB | yes | non-native |
| OpenVoice v2 | — | 300–600 ms | — | yes | native |

Kokoro hits the right balance for our use case:
- **3× more natural** sounding than Piper in informal A/B listening.
- **Low enough latency** (100–200 ms GPU, 250–400 ms CPU) that the
  per-fragment streaming pipeline (ADR 0009) stays under-budget. With
  small fragments (~3–8 words), a single synthesis call is even
  faster than full-sentence Kokoro.
- **Voice catalogue includes PT-BR** out of the box, with both male
  and female options on the most recent voice bundles. Same
  multi-voice routing logic we built for Piper applies (pick closest
  base F0, shift the residual).
- **Same on-disk footprint class** as Piper + voice files (~330 MB
  vs ~50 MB). No headache for the install step.

## Why keep pyworld on top

Kokoro voices are fixed identities — without pyworld every PT speaker
in a meeting would sound like `pm_alex`, every EN speaker like
`am_michael`. pyworld pitch+formant shift gives us per-speaker
differentiation at zero model-cost. The combination of Kokoro's
naturalness + pyworld's per-speaker cue is strictly better than
either alone.

The shift ratios stay small thanks to multi-voice routing: an EN
female speaker (~210 Hz target) maps to `af_bella` (~210 Hz base),
shift ratio ~1.0, no audible pyworld artefact. A PT male around
~110 Hz maps to `pm_alex` (~135 Hz), shift ~0.81, close to the
clean-range floor.

## Why not just use Kokoro voices directly without pyworld

You could; it's simpler. But:
1. Speaker A (male) and Speaker B (also male, slightly different
   pitch) would both render as `pm_alex`. They'd sound identical, and
   the listener loses the "who's talking now" cue we added in ADR 0006.
2. The whole rationale for the diarisation+F0 stack collapses if
   nothing downstream uses the F0.

So pyworld stays. Kokoro raises the floor of base voice quality;
pyworld adds the per-speaker delta.

## Consequences

### Positive
- **Substantially more natural-sounding TTS**, the single biggest
  perceptual quality improvement in the pipeline since switching
  off CosyVoice.
- **Larger voice catalogue** — Kokoro's voice bundle ships 30+
  voices across English, Portuguese, Spanish, etc. We use 2–4 of
  them today; adding more is a one-line change to
  `KOKORO_VOICES_BY_LANG` in the bridge.
- **GPU offload available**: with `onnxruntime-gpu` installed,
  Kokoro runs on the same GPU that hosts Whisper. Frees the CPU
  slot Piper was using; useful when Sepformer is also on (ADR 0007).

### Negative
- **+330 MB on disk and ~300 MB VRAM** vs Piper's ~50 MB on disk + 0
  VRAM. Still inside the 6 GB GPU budget shared with whisper-small,
  Qwen 1.5B Q4, and Sepformer.
- **GPU thermal pressure**: laptop GPUs running three ONNX/llama.cpp
  models simultaneously (Whisper, Kokoro, Qwen) get warmer than
  before. No measured throttling on the i5/3050 reference machine,
  but a fanless laptop might hit thermals during long sessions.
- **kokoro-onnx PyPI package versioning is loose**; the model file
  layout has changed between snapshots. We pin the v1.0 release
  files explicitly in `Install-KokoroModel`.

### Neutral
- The Rust-side TTS API is unchanged: same `synthesize(segment,
  voice_profile)` call, same binary PCM response framing. Only the
  output sample rate moved from 22 050 Hz (Piper) to 24 000 Hz
  (Kokoro); the playback resampler handles either.

## Rollout

1. Rewrite `scripts/tts_bridge.py` to load Kokoro instead of Piper.
2. Update `scripts/install.ps1`: replace `Install-PiperVoices` with
   `Install-KokoroModel` (downloads `kokoro-v1.0.onnx` + voice bank
   from the kokoro-onnx GitHub release).
3. Update `scripts/requirements.txt`: drop `piper-tts`, add
   `kokoro-onnx`.
4. Doc note on `PiperTts` struct explaining the historical name vs
   current backend.

## Rollback

Restore the Piper-era `tts_bridge.py` and `Install-PiperVoices` from
git history. The `PiperTts` Rust struct works with both — only the
Python bridge implementation changes.
