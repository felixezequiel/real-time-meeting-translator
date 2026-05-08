# ADR 0014 — XTTS-v2 voice-cloning TTS replaces Kokoro + OpenVoice TCC

- **Status:** Proposed (awaiting user approval)
- **Date:** 2026-05-08
- **Deciders:** felix
- **Supersedes:** ADR 0010 (Kokoro TTS), ADR 0011 (OpenVoice TCC)
- **Related:** ADR 0013 (V2 pipeline integration point)

## Context

The current TTS path is two stages:

1. **Kokoro TTS** (ADR 0010) synthesises the translated text using one
   of ~10 pre-trained voice tensors, picked sticky per `speaker_id`
   with F0 hysteresis. Voice is *consistent per speaker* but always
   one of Kokoro's stock voices — never matches the actual person
   in the source audio.
2. **OpenVoice v2 TCC** (ADR 0011) runs as a post-step that takes
   Kokoro's output plus a 6-second reference WAV of the original
   speaker and converts the timbre. The reference is either the
   user-recorded `voice_profile/user.wav` (mic side) or auto-enrolled
   per `speaker_id` from live audio (loopback side).

In field testing TCC fails on **every call** with
`ValueError: could not broadcast input array from shape (0,) into
shape (0,8)` inside OpenVoice's preprocessing. ADR 0011's amendment
shipped a `converted: bool` protocol flag and an auto-disable after
five consecutive fallbacks — which means **TCC is effectively off
in practice**. The user only ever hears Kokoro's stock voices.

The user explicitly asked for higher fidelity: *"conseguimos deixar
a voz mais fiel ao timbre do falante?"*. Two paths exist:

- **(A) Fix OpenVoice**: investigate the empty-tensor failure,
  probably a silence-trimming or F0-extraction issue triggered by
  Kokoro's output characteristics. Low scope, unknown landing time.
- **(B) Replace the whole stack with a TTS that does voice cloning
  natively in one model call.** This ADR.

Modern (2024+) zero-shot voice-cloning TTS architectures fold the
"clone speaker timbre from N seconds of reference audio" capability
directly into the synthesis pipeline. One model, one call, no
two-stage post-processing.

## Decision

Adopt **Coqui XTTS-v2** as the unified TTS + voice-cloning engine.
Remove Kokoro and OpenVoice TCC from the runtime. The translation
pipeline still calls the same API surface — `tts.synthesize(text,
reference_wav_path)` — but the bridge underneath is XTTS-v2 and the
reference path goes directly into the model.

### Why XTTS-v2 specifically

- **Multi-lingual native support** for English and Portuguese
  (Brazilian listed as supported) with a single model. No language
  switching at the model level.
- **Streaming inference** via `model.inference_stream(...)` returns
  a generator that yields ~80 ms audio chunks. Critical for
  preserving the streaming-translate → streaming-TTS architecture
  we just shipped (ADR 0013 amendment); without it we'd give back
  the TTFA win.
- **6-second reference** is enough for stable cloning. Same audio
  budget as the current `VoiceProfileRegistry` enrolment loop
  produces, so no upstream change.
- **Apache-2.0** licence, mirrored on HuggingFace, ~1.8 GB model
  download. Fits comfortably alongside Whisper, the diariser, and
  Qwen on a 6 GB GPU.
- **Mature** — used in production by multiple voice cloning apps;
  the failure modes are well-understood and documented.

### Why not the alternatives

- **F5-TTS** — newer, claims faster inference, but less battle-
  tested for live use. Multi-lingual support is more recent. Worth
  re-evaluating once XTTS-v2 lands and has telemetry.
- **GPT-SoVITS v2** — strong on Chinese, less so on PT-BR. Larger
  reference window (3-10 s) than we currently capture.
- **Fix OpenVoice** (path A above) — the failure root cause is
  unknown; even if fixed, we keep a two-stage pipeline that costs
  150 ms per fragment and adds a second failure surface. The
  one-stage XTTS architecture is structurally simpler.
- **MeloTTS** — fast but no zero-shot voice cloning; same fidelity
  ceiling as Kokoro alone.

## Consequences

### Positive

- **Voice fidelity matches the speaker** by design — that's the
  user's stated goal.
- **One stage instead of two** — fewer failure surfaces, fewer
  latency hops, no post-processing artefact class.
- **Removes ~150 ms of TCC latency per fragment** (when it was
  working), partially offsetting XTTS's higher synthesis cost.
- **Simpler model**: no F0 measurement, no Kokoro voice picker, no
  sticky voice hysteresis. The reference WAV uniquely defines the
  voice. `VoiceProfileRegistry` keeps its role (collect ~6 s of
  clean audio per `speaker_id`) but the running-mean-F0 logic and
  the TTS bridge's voice-routing table can both go.
- **Streaming preserved** — XTTS-v2 streaming yields chunks at
  similar TTFA to Kokoro (~200-300 ms first audio).

### Negative

- **Per-fragment latency rises**: Kokoro synthesises ~3-5 word
  fragments in 200-400 ms; XTTS-v2 streaming-mode for the same
  budget is 300-500 ms. Field-validate before deciding whether the
  TTFA gain from removing TCC offsets this.
- **GPU memory pressure**: XTTS-v2 occupies ~1.8 GB vs Kokoro's
  ~330 MB. Remaining headroom on 6 GB GPU after Whisper + Qwen +
  diariser + XTTS = ~2 GB. Tight if more components are added.
- **Migration scope**: TTS bridge rewrite, install script update,
  model download (~1.8 GB), removal of Kokoro and OpenVoice
  binaries from `models/`. ~1-2 days of focused work.

### Neutral

- **VoiceProfileRegistry stays.** Same auto-enrolment cadence, same
  output shape (one WAV per speaker_id). Only consumer changes.
- **Mic-side `voice_profile/user.wav`** keeps working as-is — the
  reference path the user already records is the same shape XTTS
  expects.
- **The `voice_convert` crate becomes empty** but I'll keep the
  module file in place as a stub during migration so downstream
  imports don't have to change in a single sweep.

## Migration plan (PR sequence)

Following the safe-refactoring discipline (parallel implementation
with feature toggle, validate, cleanup):

1. **This ADR lands** as the anchor for the decision.
2. **`scripts/xtts_bridge.py`** new file — XTTS-v2 streaming bridge
   with the same wire protocol as `tts_bridge.py` (text + reference
   in, audio chunks out). Stream-mode wrapper around
   `model.inference_stream(...)`.
3. **`crates/tts/src/lib.rs`** gains `tts_engine` enum and dispatches
   to either `kokoro` (legacy) or `xtts` (new) based on
   `config.tts_engine`. Default stays `"kokoro"` until XTTS validates.
4. **`scripts/install.ps1`** gets `Install-XttsModel` step that
   downloads `tts_models/multilingual/multi-dataset/xtts_v2` from
   HuggingFace into `models/xtts_v2/`.
5. **`crates/pipeline/src/v2.rs::flush_phrase`** is updated to pass
   the reference WAV path through the existing
   `apply_tcc_if_eligible` helper (which gets renamed to
   `apply_voice_clone_or_passthrough` and routes to the active
   engine).
6. **Field validate** with `tts_engine = "xtts"` — confirm voice
   fidelity, measure latency, watch GPU usage.
7. **Cleanup** when validated: default flag to xtts, remove
   `tts_bridge.py` (Kokoro), remove `voice_convert_bridge.py`
   (OpenVoice), remove their model files from `models/` and the
   install script, mark ADRs 0010 and 0011 Superseded by 0014.

## Rollback

`tts_engine = "kokoro"` returns the runtime to the current state
(Kokoro + dead-on-arrival TCC). Until step 7 above, the legacy
bridges and crates are still present and toggleable. After cleanup,
rollback is `git revert` of the cleanup PR plus the flag.

## Open questions for review

1. **Streaming chunk size**: XTTS-v2 streaming yields `stream_chunk_size`
   in samples. Default is around 80 ms of audio at 24 kHz. Need to
   confirm the mixer plays these gracefully with the existing
   per-chunk equal-power envelope (no clicks at chunk boundaries
   inside the same fragment).
2. **GPU vs CPU**: XTTS-v2 supports both. CPU inference is ~5-10×
   slower. Default should be GPU; expose `XTTS_DEVICE` env var
   mirroring the Sepformer pattern from ADR 0011 for users without
   CUDA.
3. **Per-speaker enrolment timing**: V2 currently enrols on the
   FIRST 6 s of clean audio per speaker_id. With XTTS we might want
   to refresh the reference periodically (longer reference → more
   stable timbre). Open question for after Phase 1 ships.

## Amendment 2026-05-08 — Atomic inference + fallback reference

First field run with the streaming-mode bridge (as originally
specified in this ADR) produced unusable wall-clock latency:
**RTF ~3.2 on RTX 3050** — 13 s of synthesis for 4 s of output. The
pipeline's bridge Mutex serialised flush_phrase calls, the queue
accumulated, and after ~40 s the listener heard nothing while
backlog grew. Two corrections shipped after that test:

### Atomic `inference()` instead of `inference_stream()`

`inference_stream(stream_chunk_size=20)` forces a decode + GPU→CPU
sync per ~80 ms audio chunk. On a 6 GB consumer GPU shared with
Whisper + Qwen + diariser, the per-chunk overhead (kernel launch,
memory allocation, sync) dominates the autoregressive cost itself
and pushes total wall-clock 3-5× past the atomic path.

`xtts_bridge.py::synthesize` now calls `model.inference(...)` —
single GPT pass, single decode at the end. We give up the
fragment-internal streaming (which the V2 pipeline never exposed
end-to-end anyway, since each call already represents one
clause-aligned fragment from `translate_stream`) in exchange for
inference time inside the budget the original ADR estimated
(~300-500 ms / fragment) instead of multi-second.

The streaming API would still be useful at a different boundary
(streaming TTS per character, with custom mixer logic), but adopting
it requires a deeper refactor than this ADR's "drop-in TTS engine
swap" envelope. Reserved as future work.

### Fallback reference WAV when none provided

The original ADR said: *"`reference_wav_path` is REQUIRED. Without
it the bridge emits silence."* Field UX feedback contradicted this:
new speakers that hadn't completed auto-enrolment heard nothing for
the first 1-2 phrases. The user reasonably asked to hear *something*
even if not yet calibrated to that person — *"mesmo que robótica"*.

`find_fallback_reference()` resolves a fallback once at bridge
startup, in this priority:
1. `models/xtts_v2/fallback_voice.wav` — explicit shippable
   fallback (not yet generated; hook for future "neutral voice"
   asset).
2. `voice_profile/user.wav` — the user's own enrolled mic profile.
   Translations come out in the user's voice while waiting for the
   real speaker enrolment. Unusual but unmistakably "real human"
   audio; the listener can tell it's a placeholder.

When auto-enrolment writes the real `ref_speaker_N.wav`, the
pipeline starts passing that as `reference_wav_path` and the bridge
swaps to the actual speaker's timbre. No code change in
VoiceProfileRegistry — only the bridge's response to a missing
reference changed from "silence" to "fallback".

If neither candidate exists, behaviour reverts to the original
silence fallback — no regression for users who delete their voice
profile.

### Cold-start warmup at bridge init

Second field run with the atomic-inference bridge surfaced a
different failure mode that masquerades as a TTS bug but is
actually a cuDNN autotune + model-lazy-init cost:

- **Conditioning latents on the first call: 51 s.** Subsequent
  calls: ~0.9 s.
- **First inference: 53 s** (RTF ~9). Second inference: 6.8 s.
  Subsequent: 1-2 s (RTF ~0.4-0.6, in budget).

In a session lasting only ~50 s, the bridge was still warming up
when the user pressed Stop, and the queue of buffered phrases
flushed *after* the pipeline had been torn down — the listener
heard nothing live, then a flood of late audio post-shutdown.

Root cause: when `torch.backends.cudnn.benchmark = True` is set
(which speeds up steady-state inference), PyTorch tests several
cuDNN convolution algorithms on the first few forward passes to
pick the fastest. With XTTS-v2's autoregressive GPT decoder
running ~50 forward passes per second of audio, that autotune
runs MANY times before stabilising, and each early pass picks a
slow algorithm. Combined with first-call lazy module
initialisation (XTTS doesn't move every sub-module to CUDA in
`.cuda()`; some are deferred until the first real forward), the
first 1-2 inferences pay 50× the steady-state cost.

Fix: `XttsBridge._warmup()` runs 2 dummy inferences (one PT, one
EN, both on the fallback reference) inside `__init__` *before*
the bridge writes `{"status": "ready"}` to stdout. Total init
goes from ~15 s → ~12 s wall-clock, but the first user-visible
call now runs at steady-state speed (~1 s) instead of cold-start
speed (~50 s).

Smoke-test result:
```
weights loaded (8313 ms)
conditioning latents for user.wav in 859 ms     # was 51 000 ms cold
warmup 1/2 in 969 ms                            # was 53 000 ms cold
warmup 2/2 in 1343 ms
ready (total init: 11 484 ms)
# first user phrase:
52224 samples for 31 chars / pt in 1297 ms      # RTF ~0.6 ✓
```

`torch.backends.cudnn.benchmark = True` is set explicitly before
weights load so the autotune happens during warmup, not later.

This warmup pattern applies more broadly than XTTS — see
`memory/project_ml_bridge_warmup_gotcha.md` for the generic
recurring lesson for any autoregressive ML bridge added to the
pipeline in the future.

### Collapse to a single TTS instance

Even with the warmup landed, a second field run still produced
silent output. Logs showed `V2 →` translations followed by
`[xtts] N samples (24000 Hz) ... in 37937 ms` (RTF ~8 — far worse
than the smoke-test's RTF ~0.6) and the user heard nothing on
either the headphones or the virtual mic for WhatsApp.

The proximate cause was that `load_models()` inherited a Kokoro-era
convention of one `PiperTts` instance per language (one for PT
output, one for EN output). With Kokoro that cost ~330 MB × 2 =
~660 MB of GPU — fine. With XTTS-v2 it costs ~1.8 GB × 2 = ~3.6 GB,
and combined with Whisper-small (400 MB) + Qwen 1.5B (1 GB) +
diariser (50 MB) it pushed total VRAM use past ~5.1 GB on a 6 GB
card. CUDA started swapping pages, cuDNN autotune picked
numerically unstable algorithms, and the model output came back
with NaN-laced tensors. `NaN × 32767 → 0` on the int16 cast
produced a non-zero sample count of *silent* PCM — which is exactly
what the log showed: large `num_samples` values, zero audio.

The fix in three parts:

1. **One bridge instance.** Both Kokoro and XTTS bridges already
   accepted the language per request in their JSON header. The
   Rust client (`crates/tts/src/lib.rs::PiperTts`) was hard-wiring
   `self.language` into every request. Changed `synthesize(...)`
   to derive the language from `TextSegment.language` instead, and
   dropped the `language` field from the struct. Constructor keeps
   the param as `_language: Language` for API stability — callers
   still pass it but it's ignored.

2. **`LoadedModels.tts: Arc<PiperTts>`** instead of `tts_portuguese`
   + `tts_english`. `models_for_source(...)` now returns the same
   `Arc` regardless of source language. Total VRAM use drops
   ~1.8 GB on the XTTS path (back below the 4 GB headroom mark).

3. **NaN/Inf guard in the bridge.** `write_pcm_response` now calls
   `np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)`
   before clipping and casting. Even when GPU thrashing somehow
   poisons the tensor again, the failure becomes audible (real
   silence) instead of mysterious (zeros that look like content
   in the log). A peak-amplitude check logs
   `WARNING: synthesised audio peak=...` when the post-clean
   audio is effectively silent — a future session debugging the
   "logs say translation but I hear nothing" report sees the
   warning immediately and knows it's a NaN → silence path, not a
   playback or routing bug.

### Streaming TTS protocol — multi-frame response

After the single-instance collapse fixed the silence, voice fidelity
was excellent but per-fragment latency rose to 4-11 s end-to-end vs
~1-2 s on the previous Kokoro path. Steady-state RTF was healthy
(~1.0-1.5) — the visible delay came from buffering an entire
fragment's audio in the bridge before sending it to Rust as one
PCM blob. With clauses of 8-15 words at PT speech rate that's
~3-5 s of audio waiting to land before the listener hears the
first phoneme.

XTTS-v2 already exposes `model.inference_stream(...)` — an iterator
that yields ~250-400 ms PCM tensors as the GPT decoder produces
them. We re-architected the bridge → Rust protocol around it.

**Wire format (multi-frame).** Each TTS request now triggers N
response frames instead of 1. Each frame:

```
{"sample_rate": 24000, "num_samples": M, "final": <bool>}\n
<M * 2 bytes int16 LE PCM>
```

The Rust client reads frames in a loop until `final=true`. Atomic
bridges (Kokoro `tts_bridge.py`) emit exactly ONE frame whose body
carries the full PCM and `final=true`; streaming bridges
(XTTS `xtts_bridge.py`) emit N body frames with `final=false`
followed by a single empty terminator frame with `final=true`.
Same client code handles both — the legacy header without `final`
deserialises with `is_final=true` by default for one-version-skew
safety.

**Mid-phrase chunks bypass the mixer envelope.** The mixer's
per-chunk equal-power crossfade (12 ms each side, ADR 0013
follow-up) was designed for atomic TTS chunks where every chunk
is a complete utterance. For streaming chunks it would chop the
waveform between adjacent fragments of the same word. We added
`is_streaming_chunk: bool` to `shared::AudioChunk`; the mixer's
phrase-aligned ingester now skips its envelope when this flag is
set. The LAST chunk of a streamed fragment is still emitted with
`is_streaming_chunk=false` so it gets the boundary fade-out a
phrase end deserves.

**Per-chunk dispatch in v2.rs.** `flush_phrase` replaces
`tts.synthesize(...)` with `tts.synthesize_stream(..., |chunk|
audio_output.send(chunk))`. Each chunk lands in the playback
mixer ~250-500 ms after the bridge starts decoding the fragment,
not ~700-2000 ms after it finishes the whole fragment. TCC
(`apply_tcc_if_eligible`) only runs on non-streaming chunks
(atomic Kokoro path or the trailing fragment of an XTTS phrase) —
running it per mid-stream chunk would compute tone-color from a
truncated utterance and produce edge artefacts.

**Warmup covers BOTH paths.** cudnn.benchmark autotunes
convolutions per (input shape, kernel) pair. Atomic
`inference(...)` and streaming `inference_stream(...)` exercise
different shapes / sequence lengths — warming only one leaves
the other paying ~50 s on its first real request. The bridge
warmup now runs both paths in both languages (4 dummy
inferences total).

**Expected impact.** Time-to-first-audio per translation
fragment drops from ~`fragment_inference_ms` (700-2000 ms typical
on RTX 3050) to ~`first_chunk_ms` (250-500 ms typical). Voice
fidelity is unchanged — the model and reference conditioning are
identical; only the *delivery* changes from one big frame to N
small ones.
