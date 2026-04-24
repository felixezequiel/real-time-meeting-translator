# ADR 0002 — Audio ducking via WASAPI Session API

- **Status:** Accepted
- **Date:** 2026-04-24
- **Deciders:** felix
- **Related:** ADR 0001 (translation optimization)

## Context

In the default setup (`loopback_device == headphones_device`), the user hears
two overlapping streams on the same device:

1. The original meeting audio (produced by Teams/Zoom/Chrome directly)
2. The translated TTS output (produced by this app)

Both streams compete at the same volume, producing a "choppy" listening
experience whenever a translation overlaps the original speech. The user
wants the original audio to **continue flowing untouched** and only be
"intercepted" transparently when a translation is spoken.

## Decision

Implement **automatic audio ducking** via the Windows Audio Session API:

- When TTS output becomes active, enumerate all audio sessions on the
  default render endpoint and **reduce their volume** (via
  `ISimpleAudioVolume::SetMasterVolume`) to a configurable target (default
  25%).
- When TTS output stops (after a short debounce), **restore** each session
  to its original volume.
- Our own process is **excluded** from ducking so the TTS itself plays at
  full volume.

A dedicated OS thread owns COM initialization and the WASAPI calls. The
pipeline signals TTS activity through the existing
`AudioPlayback::with_playing_flag(Arc<AtomicBool>)` hook, which flips
whenever the playback buffer is non-empty.

## Alternatives considered

1. **Route meeting audio through VB-Cable, make app the single arbiter.**
   - Requires user to change Windows audio routing for every meeting app.
   - Friction: too high for a zero-setup translator.
2. **Separate output device for TTS (e.g. second headphones).**
   - Needs extra hardware. Doesn't match the "transparent" request.
3. **`AudioDuckingManager` / setting our session as `AudioCategory_Communications`.**
   - Implicit ducking is controlled by Windows and is not tunable
     (fixed 80% attenuation, abrupt transitions, can be disabled by user
     in Sound Control Panel).
   - Less predictable than explicit per-session control.
4. **Full mute of other sessions during TTS.**
   - Too aggressive — cuts original voices/music completely, worse UX
     than fade-to-25%.

## Consequences

### Positive
- Zero setup for the user — no Windows audio routing changes needed.
- Original meeting audio keeps flowing uninterrupted; only its gain
  changes during translation playback.
- Works with any meeting app (Teams, Zoom, Chrome, etc.) — ducking
  targets sessions by PID, not by app name.
- Tunable: target volume and debounce window are constants we can evolve.

### Negative
- Only the **speaker pipeline** triggers ducking (mic TTS goes to
  VB-Cable, doesn't affect user's ears).
- Windows Audio Session API is COM-based — requires a dedicated thread
  with `CoInitializeEx(COINIT_MULTITHREADED)`.
- If the user launches a new audio app mid-session, its first seconds
  play at full volume until our next enumeration pass.
- Ducking uses `ISimpleAudioVolume`, which alters the per-session volume
  slider visible in Windows Sound Mixer. We restore on exit via the
  `Drop` impl, but a crash could leave other apps at reduced volume —
  acceptable since the user can reset them manually.

### Neutral
- No change to the STT/translation/TTS pipeline — the ducker is a pure
  side-effect layer around playback.

## Rollout

1. Add `windows` crate to `crates/audio/Cargo.toml` with features
   `Win32_Media_Audio`, `Win32_System_Com`, `Win32_Foundation`.
2. New module `crates/audio/src/ducking.rs` exposing `AudioDucker`.
3. Wire `Arc<AtomicBool>` from `AudioPlayback::with_playing_flag` into a
   debounced supervisor task in `main.rs` that calls
   `ducker.duck()` / `ducker.restore()`.
4. Only the speaker pipeline is wired — mic pipeline outputs to
   VB-Cable which does not reach the user's ears.

## Rollback

If ducking causes issues (wrong apps muted, volumes not restored, user
complaints), disable by not creating the supervisor task. The
`AudioDucker` `Drop` restores all original volumes. No persistent state
survives process exit.
