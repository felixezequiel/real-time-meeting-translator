# ADR 0009 — Streaming translation via Qwen 2.5 1.5B (llama.cpp)

- **Status:** Accepted
- **Date:** 2026-04-27
- **Deciders:** felix
- **Supersedes:** ADR 0001 (CTranslate2 + NLLB-200-distilled-600M)
- **Related:** ADR 0010 (Kokoro TTS), ADR 0004 (streaming STT)

## Context

The previous translator was NLLB-200-distilled-600M run via CTranslate2
with int8 quantisation. It was atomic by design: the Rust pipeline
sent a sentence, the bridge ran one `translate_batch` call, returned
the full translation. Latency was reasonable (~150–250 ms per
sentence), but the *time-to-first-audio* was bottlenecked by the
"wait for full translation" pattern: TTS only started after NLLB
finished.

The friend's voiceMaster spike (cloud-based, llama-3.3-70b on Groq)
showed how much latency the streaming pattern can claw back. With
token-by-token streaming and *fragment commits* (chunks ending on
punctuation or 25-char word boundary), TTS starts speaking the first
clause while the LLM is still generating the rest of the sentence.

For our use case the question was: can we get the same streaming
benefit *locally*, without going to llama-3.3-70b on dedicated
hardware?

## Decision

Replace NLLB + CTranslate2 with **Qwen 2.5 1.5B Instruct** (Q4_K_M
GGUF), running locally through **llama-cpp-python** with
`stream=True`. The translation bridge is rewritten to:

1. Keep the same one-line-in JSON request shape (no Rust-side change).
2. Stream the response: each commit-eligible fragment is emitted as a
   separate JSON line (`{"fragment": "...", "is_final": false}`),
   terminated by `{"fragment": "", "is_final": true}`.
3. Apply the same fragment commit rule as voiceMaster
   (`mt_groq.py:_should_commit_fragment`): commit on punctuation or
   when the buffer reaches 25 chars and contains a space.
4. Use a translation-engine system prompt + 8 few-shot examples
   (PT↔EN, mixing technical jargon and short fragments). This is
   the prompt-engineering trick that makes a small instruction-tuned
   model behave like a translation engine — without it, models in the
   1–3B range conversationally elaborate ("Sure, here's the
   translation: …") and break the streaming protocol.

The Rust client (`crates/translation/src/lib.rs`) gains a streaming
API:

```rust
pub fn translate_stream<F>(&self, segment: &TextSegment, on_fragment: F)
    -> Result<(), TranslationError>
where F: FnMut(&TranslationFragment);
```

The pipeline's translate worker (`start_translate_worker` in
`crates/pipeline/src/lib.rs`) uses this API: each fragment triggers a
separate TTS call whose output goes straight to the audio mixer. This
keeps the pre-existing per-utterance ordering (no reorder buffer
needed — fragments arrive in order from a single Python subprocess
and are pushed sequentially to the audio output channel).

The atomic `translate()` API is kept as a wrapper that buffers all
fragments — used by tests and callers that don't care about latency.

## Why Qwen 2.5 1.5B

| Candidate | Params | PT-BR | Streaming | VRAM Q4 | Tok/s 3050 |
|---|---|---|---|---|---|
| **Qwen 2.5 1.5B Instruct** | 1.5B | strong | yes | ~1 GB | 60–80 |
| Qwen 2.5 3B Instruct | 3B | strong | yes | ~2 GB | 40–50 |
| Phi-3-mini-4k | 3.8B | weak | yes | ~2.4 GB | 30–40 |
| Llama-3.2-3B-Instruct | 3B | adequate | yes | ~2 GB | 40–50 |
| Gemma-2-2B-it | 2B | adequate | yes | ~1.5 GB | 50–60 |
| NLLB-200-distilled-600M (CT2) | 600M | strong | no (atomic) | ~600 MB | n/a |

Qwen 2.5 was trained on multilingual data including extensive PT-BR;
the family's PT translation quality is competitive with NLLB on
sentence-level benchmarks and **substantially better** on context-
sensitive cases (idioms, technical jargon, ambiguous pronouns) thanks
to the LLM's contextual reasoning. Q4_K_M keeps memory and speed in
the sweet spot for a 6 GB GPU sharing space with whisper-small,
Sepformer (opt-in) and Kokoro.

3B variants would give incremental quality, but at ~50% of the
tokens/sec — first-token latency becomes a problem when the user
expects sub-1-second TTFA. We can revisit if the 1.5B output proves
insufficient on real-meeting traffic.

## Why llama-cpp-python (not CTranslate2 / vLLM / native ORT)

- llama-cpp-python ships pre-built CUDA wheels for Windows; CTranslate2
  doesn't have a Qwen converter that handles the architecture cleanly.
- vLLM is server-grade overhead — multiple GB of dependencies, designed
  for batched throughput, not single-stream latency on a laptop GPU.
- ONNX Runtime would need a custom export and per-token sampling loop
  reimplemented; llama.cpp already has all of that.
- llama-cpp-python's `create_chat_completion(..., stream=True)` API
  yields delta dicts compatible with our fragment commit logic in ~5
  lines of Python.

## Consequences

### Positive
- **Streaming**: TTS first audio comes ~150–200 ms after STT commit
  (vs ~400–500 ms with atomic NLLB). End-to-end TTFA drops by the
  same delta.
- **Quality**: contextual reasoning — "Vou fazer o commit" no longer
  becomes "I will make the commitment", idioms translate to natural
  equivalents, technical terms preserved verbatim through few-shot
  conditioning.
- **Same bridge protocol**: only the response shape changed (multi-line
  streaming). Rust callers that only need atomic translation (tests)
  get a wrapper that buffers fragments — zero forced refactor.
- **Pipeline simplification**: the translate worker is now sequential
  (one utterance at a time, fragments stream within); the previous
  semaphore + reorder buffer is gone. Less code, easier to reason
  about.

### Negative
- **+400 MB on disk** vs NLLB-CT2 (1 GB GGUF vs ~600 MB CT2 model).
  Negligible for a project that already ships ~3 GB of weights.
- **+~400 MB VRAM** vs NLLB-CT2 (Qwen Q4 ~1 GB vs NLLB-CT2 ~600 MB).
  Still well inside the 6 GB budget alongside whisper + kokoro.
- **Prompt engineering matters**: a small instruction-tuned model can
  fall out of "translation engine" mode if the input pattern is
  unusual (e.g. very short ALL-CAPS interjections). The few-shot
  examples cover the obvious cases but real-world testing will likely
  surface a few corrections.
- **First-token cold start**: ~80–150 ms on the very first request
  per session (KV cache empty). After that ~15 ms per token. Below
  the previous NLLB baseline either way.

### Neutral
- The `translate()` atomic API is preserved as a wrapper. External
  users (tests, debug tools) keep working with no source-change.

## Rollout

1. Rewrite `scripts/translation_bridge.py` as a streaming bridge.
2. Replace `OpusMtTranslator` with `LlmTranslator` in
   `crates/translation/src/lib.rs`; keep `pub type OpusMtTranslator =
   LlmTranslator;` for compatibility.
3. Add `TranslationFragment` + `translate_stream` API.
4. Refactor `start_translate_worker` to sequential streaming.
5. `scripts/install.ps1`: `Install-LlmModel` downloads
   `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` into `models/`.
6. `scripts/requirements.txt`: drop `ctranslate2`, add
   `llama-cpp-python`.

## Rollback

The previous CTranslate2 + NLLB stack is preserved in git history
(commit before this ADR landed). Revert `translation_bridge.py` and
`crates/translation/src/lib.rs`, plus the install/requirements
changes. `OpusMtTranslator` typedef means downstream callers don't
change either way.

## Amendment 2026-05-07 — Streaming reactivated post-V2

When ADR 0013 cleaned up V1, `translate_stream` temporarily lost
its only caller and `OpusMtTranslator::translate` (atomic) became
the V2 default. Field testing then showed TTFA was the dominant
"perceived delay" axis — translate-full-then-tts-full cost
~900-2000 ms before the listener heard anything. ADR 0013's
streaming-flush amendment puts `translate_stream` back into the
hot path inside `SpeakerPipelineV2::flush_phrase`, dispatching TTS
per clause fragment as it arrives. The mechanism described in this
ADR is unchanged — bridge protocol, `split_commit_point` logic,
`TranslationFragment` type, all stay. Only the *consumer* moved
from V1's streaming-STT pipeline to V2's accumulator-driven phrase
flush.

## Amendment 2026-05-07 — Interpreter-style compression in the prompt

Listening to V2 output revealed a quality ceiling that wasn't a
model capability problem — Qwen 1.5B was *faithfully* doing what
the original prompt asked, which was literal translation
("Do not omit, summarize, or paraphrase. Every content word must
be represented in the output."). Field examples like
*"yeah yeah yeah, looking forward to it"* came out as
*"sim sim sim, espero muito"* — 1:1 with the source, including the
verbal tics and stutters. Real human interpreters drop those.

`SYSTEM_PROMPT` and `FEWSHOT` in `scripts/translation_bridge.py`
were rewritten to frame the model as a *professional simultaneous
interpreter* (the TV-news kind that does live political speeches).
The new rules explicitly authorise dropping fillers ("uh", "um",
"you know", "tipo", "sabe", "né"), collapsing repetition
("yeah yeah yeah" → "sim"), compressing disfluencies and self-
corrections, and being concise without inventing content. Register
matching is preserved (casual stays casual). Proper nouns / brand
names / technical jargon in the target language stay verbatim — so
"Huge Conversations" stops becoming "Conversas Grandes".

Twelve new few-shot examples demonstrate the behaviour, taken from
real transcripts the user shipped (Apple Vision Pro podcast,
"Huge Conversations" announcement). At 1.5B parameters the verbal
prompt alone is not enough — Qwen needs the pattern in-context.

Side benefit: shorter outputs mean less work for downstream Kokoro
TTS, so the streaming pipeline naturally gets faster too. Estimated
compression ratio on filler-heavy spoken English: 1.3–1.7×.

Risks (acknowledged):
- Aggressive compression could drop nuance the speaker intended.
  `temperature=0.0` plus the "preserve meaning" rule (#2) is the
  guardrail; field-validate.
- Behaviour is prompt-engineering, not a hard constraint. A
  smaller model could ignore the rules in edge cases. If quality
  regresses, revert this commit and Qwen returns to literal mode.
- No automated test — interpreter quality is by definition
  subjective. The validation loop is "user listens, reports back".

## Amendment 2026-05-08 — Single-instance translator (collapse from two)

Field testing with XTTS-v2 (ADR 0014) on a 6 GB GPU surfaced a
broader VRAM-saturation pattern that this ADR's prior design
contributed to: `load_models()` was creating TWO `OpusMtTranslator`
instances, one per direction (EN→PT and PT→EN). With Qwen 1.5B Q4
that's ~1 GB × 2 = 2 GB of VRAM held by translation alone, on top
of Whisper × 2, diariser, and the new XTTS bridge. The 6 GB card
went into CUDA paging, cuDNN autotune chose unstable algorithms,
and one of the visible side effects was silent NaN-laced output
from the TTS path (see ADR 0014 amendment for the full diagnosis).

The translation bridge always accepted `source_lang` and
`target_lang` per request — same as the TTS bridge. The two-
instance arrangement was a Rust-side convenience: each
`OpusMtTranslator` stored a `direction: TranslationDirection` on
the struct and used it to fill in the JSON request. Cheap when
the model was small (NLLB-CT2 ~700 MB), expensive once Qwen 1.5B
joined the GPU at ~1 GB per instance, and VERY expensive once we
needed to coexist with a 1.8 GB XTTS model.

Fix mirrors the TTS one (ADR 0014 amendment "Collapse to a single
TTS instance"):

1. **Drop `direction` field from `LlmTranslator`.** Constructor
   keeps `_direction: TranslationDirection` for API compatibility
   (existing callers pass it) but ignores the value. New helper
   `invert_language(source) → target` fills in the missing target
   inside `translate_stream`.
2. **`LoadedModels.translator: Arc<OpusMtTranslator>`** instead
   of `translator_en_pt` + `translator_pt_en`. `models_for_source`
   returns the same `Arc` regardless of source — the per-segment
   `TextSegment.language` carries the direction information.
3. **Single bridge subprocess** spawned at startup. The internal
   `Mutex<BridgeProcess>` already serialises concurrent callers,
   so Mic + Speaker translating in parallel just queue (rare in
   practice — Mic translates only when the user speaks).

VRAM saved: ~1 GB. Trade-off: when Mic and Speaker both translate
simultaneously, the second waits for the first (typically 200-500
ms of streaming). Acceptable in a single-user meeting context.

The collapse pattern — "one Python bridge process per model
class, language/direction per request" — is now the project's
default convention for any new ML bridge. Explicitly NOT
splitting per-direction unless a per-instance state genuinely
diverges (Whisper-rs's WhisperState being the lone hold-out today
because each STT pipeline genuinely consumes a different audio
stream).

## Amendment 2026-05-08 — Compression target in the prompt

The 2026-05-07 amendment ("Interpreter-style compression") moved
the prompt from literal translation to professional-interpreter
behaviour (drop filler, collapse repetition, trim self-corrections).
Field listening showed Qwen 1.5B did follow the *direction* but not
the *intensity* — its outputs hovered at 90-100% of the source word
count even when the source was 50% filler. The prompt asked the
model to "be concise" without giving a target, and a 1.5B model
without an explicit anchor reverts to its training-distribution
default (which for translation tasks is roughly 1:1).

Two prompt changes:

1. **Rule 5 sharpened with a concrete target.** "When the source
   contains fillers, repetition, or self-corrections, aim for
   ≤ 70% of the source word count. When the source is already
   clean and information-dense (read text, formal speech), translate
   at ~1:1." The conditional is critical — without it, the model
   compresses dense formal speech too, dropping content-words.
2. **Five new few-shots demonstrate aggressive compression** (some
   reaching 19-29% of source length on filler-heavy English/PT) plus
   **two counter-examples** (clean technical English / formal
   Portuguese narrative) anchored at ~1:1. The asymmetry is what
   teaches the 1.5B model "compression is conditional on filler",
   not "always translate short".

Expected impact on the *latency* axis: indirect but real. Shorter
output → shorter TTS audio → shorter playback queue when the
speaker is on a monologue. The TTFA itself is unchanged (we still
emit fragments at the same commit cadence), but the listener no
longer falls progressively behind real time during a long answer.

Risks (acknowledged):
- The 70% number is a heuristic, not a hard contract. The model
  may overshoot or undershoot on inputs that don't match the
  few-shot pattern. Field-validate; if compression is too
  aggressive on, say, narration, walk the figure to 80% or weaken
  the rule's wording.
- No automated test (same reasoning as the 2026-05-07 amendment —
  interpreter quality is subjective).
