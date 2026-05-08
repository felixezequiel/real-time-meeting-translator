# ADR 0013 — Hybrid sliding-window pipeline with diariser-first routing

- **Status:** Accepted (Phase 1 + Phase 2 + Phase 3 all landed 2026-05-07)
- **Date:** 2026-05-07
- **Deciders:** felix
- **Supersedes:** ADR 0004 (streaming STT local-agreement), ADR 0012 (permutation tracker)
- **Amends:** ADR 0007 (Sepformer — becomes conditional opt-in), ADR 0009 (Qwen streaming — reduced role)
- **Related:** ADR 0005 (ECAPA diarisation — becomes central), ADR 0008 (Silero VAD), ADR 0010/0006 (TTS), ADR 0011 (OpenVoice TCC)

## Context

The current pipeline streams audio chunk-by-chunk (~500 ms) through
Whisper using local-agreement (ADR 0004) and triggers Qwen streaming
translation per fragment (ADR 0009). When the user enabled Sepformer
(ADR 0007) for source separation, the system collapsed:

- **Streaming STT fragility** — local-agreement requires consecutive
  chunks with consistent re-transcription. Sepformer outputs are
  permutation-invariant (ADR 0012 attempted to mitigate this) and
  every chunk-boundary perturbation breaks the agreement window.
- **GPU contention** — three Whisper-small instances + Sepformer +
  OpenVoice TCC + ECAPA on a 6–8 GB GPU caused intermittent stalls
  of 10–60 seconds, destroying the real-time budget.
- **Translation quality** — Qwen receives 3–8 word fragments out of
  context, produces semantically wrong Portuguese.
- **Wrong premise** — Sepformer separates *simultaneous* speakers,
  but in real meetings the dominant case is *alternating* speakers.
  Diarisation (ADR 0005) is the right primitive for that.

Field testing showed only ~1 STT commit per 8 seconds of speech
during a 70-second test, with audible translation quality "horrível".
The user described the desired UX as a **live news interpreter**:
3–5 s tolerable delay, coherent sentence-level output, voice
distinction, audible-quality translation.

## Decision

Refactor the pipeline around three principles:

### 1. Adaptive VAD-driven phrase windows

Replace fixed 500 ms streaming chunks with **adaptive phrase windows**:

- A window opens when Silero VAD detects speech onset.
- The window grows while VAD reports continuous speech, up to
  `MAX_WINDOW_MS = 5000`.
- The window closes when VAD reports `SILENCE_TAIL_MS = 400` of
  silence, or when the max cap is reached.
- Closed windows are dispatched downstream for STT/translate/TTS
  as a complete unit.
- Windows of the same speaker that overlap in *playback time* (because
  TTS for window N is still finishing when window N+1's TTS arrives)
  are blended by the crossfade mixer (V3 of the design space — audio
  always fluid).

This eliminates local-agreement entirely. Each window is a
syntactically meaningful chunk of speech, processed once.

### 2. Diariser-first routing

- The ECAPA diariser runs over each window and assigns a `speaker_id`.
- Per-speaker buffers accumulate windows for the **same** speaker, so
  consecutive phrases by the same person reuse OpenVoice TCC voice
  state and stay tonally consistent.
- A single Whisper instance services all speakers via a FIFO queue.
  No more parallel STT instances, no more GPU contention.

### 3. Sepformer as a conditional surgical tool

Sepformer (ADR 0007) is no longer always-on when the flag is set.
It runs **only when** the diariser reports ≥2 distinct speaker IDs
within the same window with comparable energy (overlap detection
threshold: |rms_a − rms_b| < `OVERLAP_RMS_DELTA`). On true overlap
it splits the window into two channels for parallel pipelines; on
single-speaker windows the audio bypasses Sepformer entirely.

### 4. Subtitle overlay UI

A new transparent always-on-top window renders translated text
incrementally (faster updates than audio). The user reads while the
TTS catches up, addressing the "delay para fazer sentido" requirement
without sacrificing audio quality.

## Architecture

```
[Loopback / Mic]
        │
        ▼
[Silero VAD continuous] ──signals──┐
        │                           ▼
        ▼                  [PhraseSegmenter]
                          (adaptive window, max 5 s)
                                    │
                                    ▼
                          [ECAPA diariser per window]
                                    │
                                    ▼
                          [Overlap detector]
                          ├── 1 speaker → mono path
                          └── ≥2 with similar energy → Sepformer (opt-in)
                                    │
                                    ▼
                          [Per-speaker buffers]
                                    │
                                    ▼
                          [PhraseModeStt — single Whisper, FIFO queue]
                                    │
                                    ▼
                          [TranslationStage (NLLB or Qwen, full window)]
                                    │
                          ┌─────────┴─────────┐
                          ▼                   ▼
              [SubtitleOverlay]      [TTS + OpenVoice TCC]
              (text streaming UI)    (audio per window)
                                              │
                                              ▼
                                  [CrossfadeMixer + ducking]
                                              │
                                              ▼
                                          [Output]
```

## Alternatives considered

1. **V1 — Texto speculativo, áudio só por frase.** Latency 3–5 s on
   audio. Rejected: user wants audio fluid, not gated.
2. **V2 — TTS em sub-frases com correção mid-sentence.** Risks
   audible cuts. Rejected: degrades the listening experience.
3. **Keep current streaming + patch Sepformer further.** Rejected:
   we've stacked two patches (RMS gate, permutation tracker) and the
   underlying premise is wrong.
4. **Drop Sepformer entirely.** Tempting, but real meetings do have
   overlap. Conditional opt-in keeps the option without the cost.

## Consequences

### Positive
- **Single Whisper instance** — GPU is no longer thrashed; STT
  latency becomes deterministic.
- **Coherent translation** — Qwen/NLLB receives a complete phrase,
  produces grammatical Portuguese.
- **No more permutation problem** — diarisation tracks identity by
  voice fingerprint, immune to Sepformer's chunk-level invariance.
- **Voice consistency** — same speaker keeps the same TCC reference
  across consecutive windows.
- **Faster perceived feedback** — subtitle overlay updates within ~1 s
  even when audio takes 3–4 s.
- **Simpler debug** — windows are units that can be logged, replayed,
  and inspected end-to-end.

### Negative / Limits
- **Audio latency floor of ~1.5–3 s** — adaptive window adds
  `SILENCE_TAIL_MS` + STT + translate + TTS. Not "live", but matches
  professional interpreter pacing.
- **Major refactor** — `SpeakerPipeline` is rewritten; ADR 0004
  streaming code goes away.
- **New components** — `PhraseSegmenter`, `PhraseModeStt`,
  `CrossfadeMixer`, `SubtitleOverlay`. Each is small but together it's
  a substantial diff.
- **Feature parity gap during rollout** — until the migration is
  complete, the user runs either the old pipeline (current) or the
  new (after refactor). No half-state.

### Neutral
- The hexagonal layering is preserved: domain code (the new
  `PhraseSegmenter`, `PhraseModeStt`, `SpeakerPipelineV2`) doesn't
  know anything about the audio adapter or the UI overlay.

## Migration plan (PR sequence)

PRs are ordered by dependency and can be reviewed independently.

1. **ADR 0013** — this document (lands first to anchor reviews).
2. **`PhraseSegmenter`** in `crates/audio` — pure logic, fully unit-
   tested, no integration yet. Idempotent addition.
3. **`PhraseModeStt`** wrapper around the existing Whisper STT —
   accepts a complete segment, returns a single string. Reuses the
   STT process; just changes the call shape.
4. **`SpeakerPipelineV2`** in `crates/pipeline` — new pipeline using
   segmenter + phrase-mode STT + diariser-first routing. Introduced
   side-by-side with the existing `SpeakerPipeline` (feature flag
   `pipeline_v2 = true` in `config.toml`).
5. **`CrossfadeMixer`** — extends `crates/audio/src/playback.rs` with
   a per-source crossfade buffer when chunks of the same source
   collide. Doesn't break the current contract.
6. **`SubtitleOverlay`** — new module in `crates/ui` using eframe.
   Receives `(speaker_id, text)` over an MPSC channel. Toggle in
   tray menu.
7. **Conditional Sepformer** — `start_separation_worker` becomes a
   diariser-aware router (Sepformer only on overlap windows).
8. **Cleanup** — once `pipeline_v2 = true` is the default, remove the
   streaming local-agreement code (ADR 0004 path), remove
   `PermutationTracker` (ADR 0012), mark those ADRs Superseded.

## Rollback

Set `pipeline_v2 = false` in `config.toml`. The legacy
`SpeakerPipeline` and the streaming local-agreement code remain
available until step 8 (cleanup). After cleanup, rollback is `git
revert` of the cleanup PR plus disabling the flag.

## Implementation status (2026-05-07)

### Phase 1 — landed
- `PhraseSegmenter` in `crates/audio/src/phrase_segmenter.rs` with
  9 unit tests.
- `SpeakerPipelineV2` in `crates/pipeline/src/v2.rs`.
- `pipeline_v2`, `phrase_max_window_ms`, `phrase_silence_tail_ms`,
  `phrase_min_window_ms` config fields.
- `main.rs` branches V1/V2 on the flag for both Speaker and Mic.

### Phase 2 — landed
- Diariser-first routing in V2: ECAPA runs per closed window,
  `VoiceProfileRegistry` (reused from V1, methods raised to
  `pub(crate)`) auto-enrols ~6 s of clean audio per speaker.
- Voice cloning: V2 picks a sticky Kokoro voice per `speaker_id`,
  feeds the running-mean F0 to the bridge, applies OpenVoice TCC
  with the auto-enrolled WAV (or `mic_voice_profile_path` when set).
- `SubtitleOverlay` in `crates/ui/src/subtitle_overlay.rs` —
  transparent always-on-top eframe window. Pipeline pushes
  `SubtitleEvent` into a `std::sync::mpsc` channel; a bridge thread
  in `main.rs` converts to `SubtitleMessage` for the overlay.
  **Opt-in via `subtitle_overlay = true` in config.toml** (default
  off). Field-confirmed limitation 2026-05-07: eframe + winit on
  Windows cannot run two `run_native` event loops in parallel — when
  the overlay is up, the Configurações window silently fails to open.
  Multi-viewport refactor scheduled for Phase 3.
- Per-chunk fade envelope in `MixerPlayback::spawn_ingester` (12 ms
  linear fade-in/fade-out) — smooths the click at phrase boundaries
  that V2's once-per-phrase chunks would otherwise produce.
- **Resampler fix** (separate ADR-worthy bug, captured as memory):
  `FftFixedIn` is now constructed once per stream and reused across
  blocks. The previous code recreated it every 1024 samples, which
  produced ~47 Hz boundary artifacts audible as "underwater radio
  robot". Affects both V1 and V2.

### Phase 3 — landed
- **3.1 Multi-viewport eframe app** (`crates/ui/src/combined_window.rs`):
  one `eframe::run_native` hosts the subtitle overlay as the primary
  viewport AND Configurações as a `show_viewport_immediate` secondary
  viewport. `SettingsApp::render_ui` was extracted from `update`
  (`pub(crate)`) so the host can drive it without owning a separate
  Frame. `subtitle_overlay::SubtitleState` exposes pure rendering
  state for the same reason. Settings + overlay coexist, fixing the
  field-confirmed two-event-loop conflict on Windows.
  `subtitle_overlay = true` is now safe to enable.
- **3.2 Conditional Sepformer.** `start_separation_worker` consults
  the diariser per chunk; Sepformer fires only when ≥2 distinct
  speaker IDs appeared in the last 2.5 s, with a 1.5 s
  hysteresis hold. Single-speaker meetings pay zero Sepformer cost.
  `PermutationTracker` (ADR 0012) is still used inside the armed
  window — it stays.
- **3.3 Equal-power crossfade** in `apply_chunk_envelope`:
  fade-in via `sin(t·π/2)`, fade-out via `cos(t·π/2)`. When two
  phrase chunks of the same speaker overlap sample-by-sample, total
  power = sin² + cos² = 1 — no audible loudness dip at the boundary.
- **3.4 V1 cleanup.** Removed: `crates/stt/src/streaming.rs`
  (StreamingSession), V1 `SpeakerPipeline` + worker functions in
  `crates/pipeline/src/lib.rs` (~795 lines), V1-only constants and
  the `pipeline_v2` config flag. ADR 0004 marked Superseded. V2 is
  now the only pipeline path; the legacy fallback is gone.

### What is still TODO (post-Phase-3)
- Field-validate combined window UX on multi-monitor setups (the
  initial overlay position assumes a 1080p primary display; user
  drags it where they want for now).
- Settings save/cancel flow inside the secondary viewport — confirm
  TrayAction-based saves still propagate correctly when the panel is
  embedded rather than standalone.

## Amendment 2026-05-07 — Inter-window phrase accumulator

Field testing exposed a quality regression after we shrank
`phrase_silence_tail_ms` from 400 → 280 ms (latency tuning). The
shorter tail closes a window on natural mid-sentence pauses (breath,
word search), so Whisper hands the translator fragments like
*"for this conversation is to try to figure out what that future"*
or *"of the future Mark Zuckerberg is trying to build so that you"*.
Qwen translates faithfully but the output is incoherent because the
input is incoherent.

The accumulator restores phrase-level coherence WITHOUT walking back
the latency improvement. `Accumulator` struct in `v2.rs` holds an
in-progress phrase across multiple closed windows. `process_segment`
now does only diariser + STT + echo check, then merges the new text
into the accumulator. Translation, TTS and audio output happen in a
new `flush_phrase` helper that fires on three conditions:

1. **Punctuation reached** — `ends_with_punctuation` matches `.!?;。；`.
   The natural ending of a sentence.
2. **Speaker changed** — early-flush the previous speaker's pending
   text, then start fresh with the new speaker's contribution.
3. **Hard caps** — `ACCUMULATOR_MAX_HOLD_MS = 3000` (3 s wall-clock)
   or `ACCUMULATOR_MAX_WORDS = 35`. Protect against monologues that
   never produce a clean punctuation boundary.

When none of these fire, the new text stays in the accumulator and
the function logs `V2 STT (held): "..."` for diagnostics; no audio
is emitted yet. Effective per-sentence latency stays similar for
short utterances but rises moderately on longer ones — the trade
the user explicitly asked for ("um pouco mais de delay para fazer
sentido na frase").

The accumulator is reset on `PipelineCommand::Stop` so leftover
text doesn't leak across sessions. Each accumulator caches the
F0 / speaker_id / TCC reference path observed when the speaker last
contributed, so `flush_phrase` can pick the right voice profile and
TCC reference without re-querying the diariser.

This is a refinement of the V2 design, not a return to V1 streaming.
V1 accumulated PER-WORD across the streaming-STT local-agreement
path; V2 accumulates PER-PHRASE-WINDOW (the whole adaptive segment
is the unit of merge). No `MIN_WORDS_FOR_PAUSE_FLUSH` heuristics,
no streaming partial-text logic — just sentence boundary detection.

## Amendment 2026-05-07 — Streaming translation in `flush_phrase`

Once the accumulator restored translation coherence, the natural
next bottleneck became time-to-first-audio (TTFA). Atomic translate
+ atomic TTS per phrase meant the listener heard nothing for
~translate (300–500 ms) + ~tts (600–1500 ms) = ~900–2000 ms after
the speaker stopped, even when the rest of the pipeline ran fast.

`flush_phrase` now uses `OpusMtTranslator::translate_stream` (already
exposed by the translation bridge — used to be V1's streaming path
under ADR 0009, dormant after the V1 cleanup). The translator emits
one `TranslationFragment` per commit-eligible clause boundary
(comma, period, semicolon). Per fragment we:

1. **Update the subtitle channel** with the cumulative text — UI
   reflects translation progress at clause granularity.
2. **Synthesise + dispatch THIS fragment alone** through TTS and
   `audio_output`. The mixer's per-source FIFO buffer naturally
   serialises consecutive fragments into a single playback line,
   while later fragments are still being generated.
3. **Apply TCC** via `apply_tcc_if_eligible` (extracted helper) —
   the `TCC_MIN_DURATION_MS = 500` guard skips the bridge for very
   short fragments that historically tripped OpenVoice
   preprocessing (see ADR 0011 amendment).

Quality is preserved because the LLM **sees the full phrase at
once** — the accumulator delivers the complete sentence as a single
prompt. Streaming only changes when output starts flowing back; the
underlying decoder context is unchanged.

The `translate_first_fragment` metric was added to the metrics
aggregator so the operator can verify the win in the settings panel:
typical value should drop from ~translate-full-time to ~50–150 ms.

Trade-off: the post-stream `is_translation_degenerate` check now
runs *after* the audio has already played. If the translator
produces a degenerate output, the bad audio leaked. Mitigation:
Qwen at temperature 0.0 + few-shot prompt rarely produces
degenerate output for normal input; when it does, we still suppress
the echo-buffer record so the bad text doesn't pollute future echo
detection. Acceptable cost for the TTFA improvement.

## Amendment 2026-05-07 — Internal sentence splitting in the accumulator

Field logs revealed a 9-second TTFA spike on multi-sentence speech
like *"I'd love to tell you what my goal is of this conversation.
Go for it. So we have"*. The accumulator only flushed when the
ENTIRE buffer ended in punctuation, so a buffer with two complete
sentences plus an open trailing fragment held all three until the
fragment finally closed.

The accumulator now uses `split_complete_sentences(text) →
(complete, remaining)`. After every ingest:
- Pull all sentences from the front that end in `.!?;` followed by
  whitespace or EOL — those flush immediately.
- Keep the trailing in-progress fragment in the accumulator with a
  fresh `started_at` so MAX_HOLD applies to the fragment alone.
- If no complete sentences exist but a hard cap fires (`aged_out`
  or `long_enough`), the whole buffer flushes anyway.

Boundary rule guards against false positives: periods inside
acronyms ("U.S.A is") or numbers ("3.14") are NOT boundaries
because the next byte is alphanumeric, not whitespace. Byte-level
scan is UTF-8-safe — Portuguese accents on the *next* word don't
trip it because the byte right after the period is the ASCII space.

Multi-sentence flush behaviour: a single ingest can now produce
TWO flushes (early on speaker change, then main on internal
boundary), and `process_segment` already handled multiple flush
returns from the lock-scoped block.

Seven unit tests in `v2.rs::tests::split_*` lock the boundary
behaviour, including the acronym false-positive case and the
Portuguese-accent UTF-8 case.

## Amendment 2026-05-07 — Subtitle `phrase_id` and draggable overlay

Two related UX issues from the streaming subtitle path:

**Stacking.** The streaming translator emits N `SubtitleEvent`s per
phrase (one per clause boundary), each carrying the cumulative
translation so far. The overlay was `push`ing every event as a new
line, so a 3-clause phrase produced 3 stacked lines — overflowing
the visible area.

**Fix.** `SubtitleEvent` and `SubtitleMessage` carry a `phrase_id:
u64` generated once per `flush_phrase` invocation (atomic counter
in `v2.rs::PHRASE_ID_COUNTER`). `SubtitleState::push` now updates
the most recent line in place when the new event's `phrase_id`
matches; it only opens a new line when the id changes. Streaming
within one phrase becomes a single growing line; new phrase = new
line at the bottom.

**Fixed-position window.** The overlay was created with
`decorations(false)` so the user couldn't drag it. Field complaint:
the overlay sometimes covered the speaker's slides.

**Fix.** `SubtitleState::render` allocates an interaction area
covering the central panel with `Sense::click_and_drag()`. On
drag, it issues `ctx.send_viewport_cmd(ViewportCommand::StartDrag)`
which delegates window movement to winit. The user can now grab
the overlay anywhere (text, empty space, heartbeat indicator) and
move it to any screen position.

Tests in `subtitle_overlay::tests::push_*` cover the new
update-in-place semantics, including the cap on visible lines and
the timestamp refresh on update so a streaming phrase doesn't fade
out mid-flow.

## Configuration additions

```toml
# config.toml — new fields with defaults
pipeline_v2 = true                  # use the new sliding-window pipeline
phrase_max_window_ms = 5000         # adaptive window upper bound
phrase_silence_tail_ms = 400        # silence required to close window
phrase_min_window_ms = 600          # below this, treat as noise
overlap_rms_delta = 0.05            # threshold for true overlap detection
subtitle_overlay = true             # show translation as on-screen subtitles
subtitle_position = "bottom"        # bottom | top
crossfade_ms = 150                  # equal-power crossfade between TTS chunks
```
