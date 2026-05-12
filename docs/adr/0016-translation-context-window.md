# ADR 0016 — Translation context window for the Qwen bridge

- **Status:** Accepted — Implementation landing 2026-05-12.
- **Date:** 2026-05-12
- **Deciders:** felix
- **Related:** ADR 0009 (Qwen streaming translation), ADR 0013 (V2 hybrid pipeline), ADR 0015 (streaming STT)

## Context

The 2026-05-12 captures showed the translation pipeline producing
isolated mistakes that share a common cause: each phrase is sent to
Qwen 1.5B-Q4 as a **fresh chat** with no memory of what was said in
the previous phrases of the same conversation. Examples from the
real log:

- *"Tony Stark glasses"* → *"óculos Tony Stark reais"* (the model
  added "reais" / "real" — it had no clue the speaker meant the
  fictional character's AR glasses, so it half-translated "real-life
  Tony Stark" to a literal phrase).
- *"escrow on a tower"* (Whisper misheard "out a stroll on a tower")
  → *"Escrow para duas ligações"*. With the prior context *"from
  mainframe computers to..."*, the model would at least have a hint
  this is computing-related vocabulary.
- *"glasses I think that exist in the world"* → *"acho que existem"*
  — the model dropped "glasses" and "in the world" because in
  isolation the fragment looked like an aside.
- *"Ten years of work right there"* alternated between *"Dez anos"*
  and *"Doze anos"* across runs — typical 1.5B Q4 instability that
  context would anchor.

The Qwen bridge already feeds 30+ few-shot examples per chat
completion, but those are *static* (filler removal, unit anchoring,
compression targets). They don't tell the model what the *current
speaker* has been saying. Conversational anaphora ("these",
"it", "that") and topic continuity ("from mainframe to phones
to glasses") need session-local memory, not a frozen prompt.

The natural place to inject this memory is the same chat-completion
API: prepend the last few `(user, assistant)` exchanges before the
new user turn. The model is already used to "the previous turn is
the user's text and my reply is the translation" — adding two real
examples on top of the few-shots costs no architecture change.

## Decision

The translation bridge maintains a per-direction ring of recent
phrase pairs (`(source_text, target_text)`) and prepends the last
**K = 2** entries to every chat completion, *after* the static
few-shots and *before* the new user turn.

The ring is keyed by `(source_lang, target_lang)` so the EN→PT and
PT→EN directions don't pollute each other's context (relevant for
multi-pipeline setups where both Mic and Speaker translate
simultaneously in opposite directions).

A new pair is pushed to the ring only after `translate_streaming`
finishes the full translation successfully and emits a non-empty
output. Degenerate, empty, or error-fallback outputs are not
recorded — they would mislead the next translation more than
helping it.

The ring is in-process only. No persistence across restarts: the
bridge boots with empty history and rebuilds it as the user speaks.

## Architecture

```
build_messages(text, source_lang, target_lang)
    │
    ├── system prompt   ──── interpreter-style guidance
    ├── few-shot pairs  ──── 30+ static (filler removal, units, …)
    ├── HISTORY (NEW)   ──── last K=2 (user=src_prev, asst=tgt_prev) for THIS direction
    └── user turn       ──── current source text
                              ↓
                       Qwen chat_completion(...)
                              ↓
                  stream → full target_text
                              ↓
                  append (src, target_text) to history ring
```

## Why K = 2

- **K = 0** is the current behaviour — proven to lose continuity.
- **K = 1** carries one prior exchange. Helps with immediate
  anaphora ("these glasses", "it works") but not with topic shifts
  that need >1 sentence of buildup ("we built X", "Y of them",
  "but the third one was different").
- **K = 2** spans the *typical* unit of conversational coherence:
  one statement + one elaboration is usually enough for anaphora,
  topic continuity, and unit consistency without making the prompt
  bloated.
- **K = 3+** risks the model treating the history as something to
  *continue* rather than translate — at K=3 with Qwen 1.5B small,
  early field tests showed the model occasionally extending the
  previous sentence instead of translating the new one.

Per-phrase token cost at K=2:
- Each historical pair: ~30 tokens source + ~30 tokens target = ~60 tokens.
- Two pairs: ~120 tokens.
- On top of ~1200 tokens of system prompt + few-shots.
- Total prompt grows ~10 %, decoder TTFA grows ~80-150 ms
  proportional to KV-cache fill. Acceptable inside the V2
  accumulator's existing 800-2000 ms hold window.

## Consequences

### Positive

- Anaphora-heavy passages translate coherently: "Tony Stark glasses
  […] I think they exist" no longer drops the noun.
- Topic continuity across phrases: technical vocabulary stays in
  the same register once seen once.
- Numeric and named-entity consistency: "Ten years" once translated
  as "10 anos" tends to stay "10 anos" in the same conversation
  instead of randomly becoming "Doze".
- Cost is hyper-local: only the bridge changes; the Rust pipeline
  and the V2 accumulator are unchanged. Zero risk to the audio
  path stability we just achieved.

### Negative

- Polluted history can amplify mistakes: if Qwen mis-translates one
  phrase, that mis-translation enters the history and the model may
  mimic the same error on follow-ups. Mitigated partially by the
  degenerate-output filter — bad translations don't get cached.
- Prompt length grows linearly with K; at K=2 we're well under the
  4 k context limit, but values above 3 start risking truncation
  on rambling input.
- The bridge becomes stateful in a way it wasn't before. Any test
  that calls `build_messages` twice for the same direction has to
  account for the order or reset the history explicitly. New unit
  tests cover this.

### Neutral

- No new model, no new VRAM cost.
- No interaction with the SBD / accumulator layer — those decide
  *when* a phrase ships; this changes *how* the phrase translates
  once it arrives.

## Rollout

1. **PR 1 — unit tests for the history ring** (red phase, the
   bridge does not yet implement the ring).
2. **PR 2 — implement the ring + `build_messages` change** (green
   phase). Confirms tests pass and `cargo build` still green.
3. **PR 3 — field validate**: capture one log and compare against
   the 2026-05-12 baseline. Specifically watch for the recurring
   anaphora errors flagged in the Context section. If the listener
   reports regressions ("Qwen looped" or "translated the previous
   sentence again"), drop K back to 1.

PR 1 and PR 2 land in the same commit because the change is small
and tested together.

## Alternatives considered

1. **Trade Qwen 1.5B for Qwen 2.5 3B / 7B Q4.** Much larger model
   = much better translations in isolation. Rejected for this ADR
   (kept on the roadmap): 7B Q4 alone needs ~5 GB VRAM, on a 6 GB
   GPU already running Whisper + Kokoro + OpenVoice TCC. The
   memory budget is tight; OOM on heavy phrases would be worse
   than the context-less mistakes we have today. Worth revisiting
   on a larger GPU or once XTTS-style heavy components are off
   the GPU.
2. **Persist history across restarts (disk).** Adds reliability
   risk (corrupt file → bad translations on every boot) without a
   clear win — the model has no memory of WHO the speakers are
   anyway, so reusing yesterday's translations doesn't compound.
3. **Pass full conversation transcript every turn.** Trivially
   blows the 4 k context limit. K=2 is the largest window where
   prompt growth stays sub-linear in conversation length.
4. **Use the history at a different layer (e.g., accumulator).**
   The accumulator already keeps a sticky reference path and a
   speaker history for routing — adding semantic history there
   crosses concerns. Translation-side history is the right home.
