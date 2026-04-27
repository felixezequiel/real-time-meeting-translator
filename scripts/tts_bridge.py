"""
TTS bridge using Kokoro v1.0 ONNX + pyworld pitch/formant shifting.

Why Kokoro (vs the previous Piper baseline):

  Piper sounds clearly synthetic — fine for proof-of-life translation
  but the robotic colour comes through every utterance. Kokoro v1.0
  is an 82M-parameter ONNX model with substantially more natural
  prosody and breathing, and it's the same TTS the friend's
  voiceMaster spike runs. Both are local; Kokoro is just bigger and
  newer.

Why pyworld stays on top:

  Kokoro itself produces a fixed voice per `voice` argument. To keep
  per-speaker pitch differentiation (a feature voiceMaster doesn't
  have), we run the same WORLD analysis-synthesis we ran on Piper
  before: extract F0, swap for the speaker's running F0, optionally
  shift the spectral envelope, resynthesise. Multi-voice routing also
  stays — for an English female speaker we pick `af_bella` (already
  ~210 Hz) so the residual shift is small and stays inside pyworld's
  clean range.

Protocol (binary-framed, unchanged from the Piper era):

- Startup (text line on stdout):
    {"status": "ready"}\\n

- Request (text line on stdin):
    {"text": "...",
     "language": "en|pt|en-us|pt-br",
     "target_f0": 180.0,            (optional Hz; speaker's running F0)
     "formant_shift": 1.0           (optional ratio; 1.0 = no change)
    }\\n

- Response (binary):
    {"sample_rate": 24000, "num_samples": N}\\n
    <N * 2 bytes int16 little-endian PCM>

Requires:
  pip install kokoro-onnx pyworld numpy
Models (~330 MB total: kokoro-v1.0.onnx + voices-v1.0.bin) are
downloaded by `Install-KokoroModel` in scripts/install.ps1 into the
project's `models/kokoro/` directory.
"""

import json
import os
import sys
import time

import numpy as np


stdout_bin = sys.stdout.buffer
sys.stdin = __import__("io").TextIOWrapper(sys.stdin.buffer, encoding="utf-8")


# Multi-voice routing per language. We list MORE voices than we strictly
# need so that documentaries with several speakers of the same gender
# can each get a distinct voice (round-robin among the closest matches).
# At synthesis time we pick the unused voice whose measured F0 is
# closest to the target — see `pick_voice_with_uniqueness` below.
#
# Kokoro v1.0 voice naming: `{lang}{gender}_{name}`.
#   am_*  : American English, male       af_*  : American English, female
#   bm_*  : British English, male        bf_*  : British English, female
#   pm_*  : Portuguese (Brazil), male    pf_*  : Portuguese (Brazil), female
#
# All these voices ship with the kokoro-v1.0.onnx + voices-v1.0.bin
# bundle. Voices not present at runtime are filtered by `setup_piper`
# during boot.
KOKORO_VOICES_BY_LANG = {
    "en": [
        "am_michael",   # male, baritone (~120 Hz)
        "am_eric",      # male, mid (~135 Hz)
        "am_adam",      # male, lighter (~150 Hz)
        "af_bella",     # female, mid (~210 Hz)
        "af_sarah",     # female, brighter (~225 Hz)
        "af_nicole",    # female, lower (~190 Hz)
    ],
    "en-us": [
        "am_michael",
        "am_eric",
        "am_adam",
        "af_bella",
        "af_sarah",
        "af_nicole",
    ],
    "pt": [
        "pm_alex",      # male (~135 Hz)
        "pm_santa",     # male, alternate (~125 Hz if shipped)
        "pf_dora",      # female (~205 Hz if shipped)
    ],
    "pt-br": [
        "pm_alex",
        "pm_santa",
        "pf_dora",
    ],
}

# Kokoro's internal language tag. Different from the wire-protocol code
# we receive from Rust (which uses ISO-639 short forms).
LANG_TO_KOKORO = {
    "en": "en-us",
    "en-us": "en-us",
    "pt": "pt-br",
    "pt-br": "pt-br",
}

# Kokoro's HiFi-GAN-style decoder emits 24 kHz mono float32. We keep
# this rate end-to-end through the pitch shifter; the playback resampler
# on the Rust side handles 24→48 kHz.
KOKORO_OUTPUT_SAMPLE_RATE = 24_000

# Empirically measured F0 (cached after the first calibration call):
# {voice_name: mean_F0_hz}. Used at runtime to pick the base voice
# closest to the target F0, minimising the shift required.
_KOKORO_F0_CACHE: dict[str, float] = {}

# Sticky voice routing: {(language, speaker_id): voice_name}. Once a
# speaker is assigned a voice, the bridge keeps using it across
# utterances unless the running F0 has drifted far enough to make
# switching obviously the right call (see VOICE_SWITCH_DELTA_HZ).
# Without this, the per-chunk pyworld F0 jitter alternated the same
# speaker between male and female voices on consecutive sentences —
# the listener heard 4 different "people" when there was actually
# only one. The hysteresis trades a (very rare) wrong-voice lock-in
# for a stable, identifiable voice per speaker.
_VOICE_PER_SPEAKER: dict[tuple[str, int], str] = {}

# Minimum distance (Hz) by which the target F0 must favour the
# alternative voice over the currently-locked voice before we accept a
# switch. Picked so that normal vocal expression (~30 Hz of jitter for
# an excited speaker) doesn't trigger a flip; only a sustained shift
# into the other gender's range (typical delta ~80 Hz between male
# and female voice baselines) does.
VOICE_SWITCH_DELTA_HZ = 50.0


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def write_json_line(obj: dict) -> None:
    stdout_bin.write((json.dumps(obj, ensure_ascii=True) + "\n").encode("utf-8"))
    stdout_bin.flush()


def write_pcm_response(samples_np: np.ndarray, sample_rate: int) -> None:
    samples_np = np.clip(samples_np, -1.0, 1.0)
    int16 = (samples_np * 32767.0).astype(np.int16)
    pcm_bytes = int16.tobytes()
    header = {"sample_rate": int(sample_rate), "num_samples": int(int16.size)}
    stdout_bin.write((json.dumps(header) + "\n").encode("utf-8"))
    stdout_bin.write(pcm_bytes)
    stdout_bin.flush()


# ─── Model loading ──────────────────────────────────────────────────────────

def find_kokoro_paths() -> tuple[str, str]:
    """Locate the Kokoro model + voices files, in the same search order
    other bridges use for their assets."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    candidates = [
        (os.path.join(project_root, "models", "kokoro", "kokoro-v1.0.onnx"),
         os.path.join(project_root, "models", "kokoro", "voices-v1.0.bin")),
        (os.path.join(os.getcwd(), "models", "kokoro", "kokoro-v1.0.onnx"),
         os.path.join(os.getcwd(), "models", "kokoro", "voices-v1.0.bin")),
    ]
    env_model = os.environ.get("KOKORO_MODEL_PATH")
    env_voices = os.environ.get("KOKORO_VOICES_PATH")
    if env_model and env_voices:
        candidates.insert(0, (env_model, env_voices))

    for model, voices in candidates:
        if os.path.isfile(model) and os.path.isfile(voices):
            return model, voices
    raise FileNotFoundError(
        "Kokoro model files not found. Run scripts/install.ps1 to download "
        "them, or set KOKORO_MODEL_PATH / KOKORO_VOICES_PATH. Tried: "
        + ", ".join(f"{m}+{v}" for m, v in candidates)
    )


def load_kokoro():
    """Instantiate kokoro-onnx with CUDA when available. Returns the
    Kokoro object; voice selection happens per-call."""
    from kokoro_onnx import Kokoro
    import onnxruntime as ort

    # Diagnostic: log what ORT is offering. When the user sees `tts`
    # P50 stuck at >500 ms, the most common cause is CUDAExecutionProvider
    # being absent here (= no `onnxruntime-gpu`, or cuDNN missing) and
    # Kokoro silently running on CPU.
    available = ort.get_available_providers()
    log(f"[init] ORT available providers: {available}")
    log(f"[init] ORT device: {ort.get_device()}")
    if "CUDAExecutionProvider" not in available:
        log(
            "[init] WARNING: ORT cannot see CUDAExecutionProvider — Kokoro "
            "will run on CPU. Reinstall onnxruntime-gpu or check cuDNN 9 + "
            "CUDA 12.x DLLs are on PATH."
        )

    model_path, voices_path = find_kokoro_paths()
    log(f"Loading Kokoro v1.0 (model={model_path})…")
    # kokoro-onnx 0.3.6+ picks the provider list from onnxruntime
    # automatically: GPU if available, else CPU. No override exposed in
    # the constructor, so we rely on the installed onnxruntime variant.
    kokoro = Kokoro(model_path, voices_path)
    log("Kokoro ready")
    return kokoro


def list_available_voices(kokoro) -> list[str]:
    """Best-effort enumeration of voices the loaded model knows. Used
    only to filter our PIPER_VOICES_BY_LANG list down to what's
    actually available — different model snapshots ship different
    voice subsets."""
    try:
        # kokoro-onnx exposes `voices` as a dict-like or list, depending
        # on version. Cover both shapes.
        voices = getattr(kokoro, "voices", None)
        if voices is None:
            return []
        if hasattr(voices, "keys"):
            return list(voices.keys())
        return list(voices)
    except Exception as e:
        log(f"Could not enumerate Kokoro voices: {e}")
        return []


# ─── Synthesis ──────────────────────────────────────────────────────────────

def synthesize_kokoro(
    kokoro,
    text: str,
    voice: str,
    language: str,
) -> tuple[np.ndarray, int]:
    """Run Kokoro TTS synchronously. Returns (samples_f32, sample_rate).

    `language` here is the Rust-wire code; we translate to Kokoro's
    `lang` parameter which uses the `en-us` / `pt-br` long form.

    `speed=1.20` because Portuguese translations are reliably 15-30%
    longer (in spoken duration) than the English source. 1.15 was the
    first attempt; the playback queue still grew over sustained
    monologue (10+ s of continuous talking). 1.20 keeps queue depth
    under the 6-second drop-at-ingress threshold even during long
    explanatory passages and still sounds natural — not perceptibly
    rushed for an interpreter-style translation.
    """
    lang = LANG_TO_KOKORO.get(language, "en-us")
    samples, sample_rate = kokoro.create(
        text,
        voice=voice,
        speed=1.20,
        lang=lang,
    )
    samples = np.asarray(samples, dtype=np.float32)
    return samples, int(sample_rate)


def measure_voice_f0(samples: np.ndarray, sample_rate: int) -> float:
    """Mean F0 over voiced frames of `samples`. Used once per Kokoro
    voice to compute the baseline pitch we shift away from."""
    import pyworld as pw

    f0, _t = pw.harvest(
        np.ascontiguousarray(samples.astype(np.float64)),
        sample_rate,
        f0_floor=70.0,
        f0_ceil=400.0,
        frame_period=10.0,
    )
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        return 120.0
    return float(np.mean(voiced))


def measure_all_voices(kokoro, voice_universe: list[str]) -> None:
    """Synthesise a calibration phrase per voice and cache its F0.
    Voice routing in the hot path is then a dict lookup."""
    for voice_name in voice_universe:
        # Pick a calibration phrase that matches the voice's likely
        # native language so Kokoro's phonemiser doesn't fight us.
        if voice_name.startswith(("p", "P")):
            calib = "Olá mundo, isto é um teste."
            lang = "pt"
        else:
            calib = "Hello world, this is a test."
            lang = "en"
        try:
            samples, sr = synthesize_kokoro(kokoro, calib, voice_name, lang)
        except Exception as e:
            log(f"[init] could not measure {voice_name}: {type(e).__name__}: {e}")
            continue
        if samples.size < int(sr * 0.5):
            f0 = 0.0
        else:
            f0 = measure_voice_f0(samples, sr)
        _KOKORO_F0_CACHE[voice_name] = f0
        log(f"[init] kokoro voice {voice_name} measured F0={f0:.1f} Hz")


def pick_voice_for_f0(
    voices_for_lang: list[str],
    target_f0: float,
) -> str | None:
    """Pick the loaded voice whose measured F0 is closest to `target_f0`.
    Falls back to the first available when the cache is cold or the
    caller didn't supply a target."""
    if not voices_for_lang:
        return None
    if target_f0 <= 0:
        return voices_for_lang[0]
    best_name, best_dist = None, float("inf")
    for name in voices_for_lang:
        f0 = _KOKORO_F0_CACHE.get(name)
        if f0 is None or f0 <= 0:
            continue
        dist = abs(f0 - target_f0)
        if dist < best_dist:
            best_name, best_dist = name, dist
    return best_name or voices_for_lang[0]


def pick_voice_with_hysteresis(
    voices_for_lang: list[str],
    target_f0: float,
    language: str,
    speaker_id: int | None,
) -> str | None:
    """Sticky voice selection per speaker, biased toward uniqueness.

    Goals (in priority order):
      1. The same `speaker_id` always gets the same voice across
         utterances (sticky → no flapping per chunk-by-chunk F0 noise).
      2. Two DIFFERENT speakers get DIFFERENT voices when possible
         (uniqueness → listener can tell them apart in a documentary).
      3. Among unused voices, prefer the one whose measured F0 is
         closest to the speaker's target F0 (so a male speaker still
         maps to a male voice, not a random female one).

    `speaker_id` of None falls through to the stateless F0 picker —
    used when diarisation is off or the chunk was too quiet to
    identify.
    """
    if not voices_for_lang:
        return None
    if speaker_id is None:
        return pick_voice_for_f0(voices_for_lang, target_f0)

    key = (language, speaker_id)
    locked = _VOICE_PER_SPEAKER.get(key)

    if locked is None or locked not in voices_for_lang:
        # First time we're routing for this speaker. Choose a voice
        # that is both (a) close in F0 and (b) not already taken by
        # another speaker. We sort all voices by F0 distance to target
        # and walk the list until we find one that's unused. If every
        # voice is already taken (more speakers than voices in the
        # catalog), fall back to the closest match — distinct-voice
        # is a soft preference, not a hard guarantee.
        used_voices = {
            v for (lang, _), v in _VOICE_PER_SPEAKER.items()
            if lang == language
        }
        ranked = _voices_by_f0_distance(voices_for_lang, target_f0)
        chosen = next(
            (v for v in ranked if v not in used_voices),
            ranked[0] if ranked else None,
        )
        if chosen is not None:
            _VOICE_PER_SPEAKER[key] = chosen
            chosen_f0 = _KOKORO_F0_CACHE.get(chosen, 0.0)
            already_used = chosen in used_voices
            log(
                f"[voice] speaker {speaker_id} ({language}) → "
                f"assigned {chosen} (voice_f0={chosen_f0:.0f}, "
                f"target_f0={target_f0:.0f}"
                + (", reused — out of unique voices" if already_used else "")
                + ")"
            )
        return chosen

    # Already locked. Decide whether to switch.
    if target_f0 <= 0:
        return locked
    locked_f0 = _KOKORO_F0_CACHE.get(locked, 0.0)
    if locked_f0 <= 0:
        return locked
    candidate = pick_voice_for_f0(voices_for_lang, target_f0)
    if candidate is None or candidate == locked:
        return locked
    candidate_f0 = _KOKORO_F0_CACHE.get(candidate, 0.0)
    if candidate_f0 <= 0:
        return locked

    # Switch only when the new voice is decisively closer to target.
    locked_dist = abs(target_f0 - locked_f0)
    candidate_dist = abs(target_f0 - candidate_f0)
    if (locked_dist - candidate_dist) >= VOICE_SWITCH_DELTA_HZ:
        log(
            f"[voice] speaker {speaker_id} ({language}): switching "
            f"{locked} → {candidate} "
            f"(target_f0={target_f0:.0f}, locked_dist={locked_dist:.0f}, "
            f"candidate_dist={candidate_dist:.0f})"
        )
        _VOICE_PER_SPEAKER[key] = candidate
        return candidate
    return locked


def _voices_by_f0_distance(
    voices_for_lang: list[str],
    target_f0: float,
) -> list[str]:
    """Return voices sorted by absolute F0 distance to `target_f0`.
    Voices without a cached F0 are appended at the end (we still want
    them as fallbacks when every measured voice is exhausted)."""
    measured = [
        (v, abs(_KOKORO_F0_CACHE.get(v, 0.0) - target_f0))
        for v in voices_for_lang
        if _KOKORO_F0_CACHE.get(v, 0.0) > 0
    ]
    measured.sort(key=lambda pair: pair[1])
    unmeasured = [
        v for v in voices_for_lang
        if _KOKORO_F0_CACHE.get(v, 0.0) <= 0
    ]
    return [v for v, _ in measured] + unmeasured


# ─── pyworld pitch + formant shift (unchanged from the Piper version) ───────

def shift_pitch_and_formant(
    samples: np.ndarray,
    sample_rate: int,
    source_f0: float,
    target_f0: float,
    formant_shift: float,
) -> np.ndarray:
    """Apply pitch and formant shift via WORLD analysis-synthesis. See
    the Piper-era version of this comment block for the longer rationale;
    the only thing that changed is the input source rate (24 kHz from
    Kokoro vs 22 kHz from Piper)."""
    import pyworld as pw

    try:
        x = np.ascontiguousarray(samples.astype(np.float64))
        f0_floor = max(60.0, source_f0 * 0.5) if source_f0 > 0 else 70.0
        f0_ceil = min(500.0, source_f0 * 2.5) if source_f0 > 0 else 400.0
        f0, t = pw.harvest(
            x, sample_rate,
            f0_floor=f0_floor, f0_ceil=f0_ceil,
            frame_period=10.0,
        )
        f0 = np.ascontiguousarray(f0)
        t = np.ascontiguousarray(t)
        sp = pw.cheaptrick(x, f0, t, sample_rate)
        ap = pw.d4c(x, f0, t, sample_rate)

        ratio = target_f0 / source_f0 if source_f0 > 0 else 1.0
        ratio = float(np.clip(ratio, 0.7, 1.4))
        f0_shifted = np.ascontiguousarray(f0 * ratio)

        if abs(formant_shift - 1.0) > 1e-3:
            n_bins = sp.shape[1]
            old_idx = np.arange(n_bins)
            new_idx = old_idx / formant_shift
            new_idx = np.clip(new_idx, 0, n_bins - 1)
            lower = np.floor(new_idx).astype(np.int32)
            upper = np.minimum(lower + 1, n_bins - 1)
            frac = new_idx - lower
            sp = np.ascontiguousarray(
                sp[:, lower] * (1.0 - frac) + sp[:, upper] * frac
            )

        sp = np.ascontiguousarray(sp)
        ap = np.ascontiguousarray(ap)
        y = pw.synthesize(f0_shifted, sp, ap, sample_rate, frame_period=10.0)
        return y.astype(np.float32)
    except Exception as e:
        log(f"pitch/formant shift failed, returning original: {type(e).__name__}: {e}")
        return samples


# ─── Main loop ──────────────────────────────────────────────────────────────

def main() -> None:
    log("Setting up Kokoro TTS…")
    kokoro = load_kokoro()
    available = set(list_available_voices(kokoro))

    # Filter our voice catalog down to what the model actually shipped
    # with. Different snapshots include different voices; if the user's
    # model lacks `pf_dora` we just fall back to `pm_alex`.
    voices_for_lang: dict[str, list[str]] = {}
    universe: set[str] = set()
    for lang, names in KOKORO_VOICES_BY_LANG.items():
        present = [n for n in names if (not available) or n in available]
        if not present:
            log(f"[init] WARNING: no Kokoro voices for {lang} (looked for {names})")
            continue
        voices_for_lang[lang] = present
        universe.update(present)
    if not voices_for_lang:
        log("FATAL: no Kokoro voices loaded — model file has no usable voices.")
        sys.exit(2)

    log(f"Kokoro ready ({len(universe)} unique voices: {sorted(universe)})")
    log("[init] measuring per-voice F0 (one-time calibration)…")
    measure_all_voices(kokoro, sorted(universe))

    write_json_line({"status": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            text = request.get("text", "").strip()
            language = request.get("language", "en")
            target_f0 = float(request.get("target_f0", 0.0))
            formant_shift = float(request.get("formant_shift", 1.0))
            speaker_id_raw = request.get("speaker_id")
            speaker_id = int(speaker_id_raw) if speaker_id_raw is not None else None

            if not text:
                write_pcm_response(np.zeros(1, dtype=np.float32), KOKORO_OUTPUT_SAMPLE_RATE)
                continue

            voices = (voices_for_lang.get(language)
                      or voices_for_lang.get(language.split("-")[0])
                      or [])
            if not voices:
                log(f"No voice available for language {language!r}, sending silence.")
                write_pcm_response(np.zeros(1, dtype=np.float32), KOKORO_OUTPUT_SAMPLE_RATE)
                continue

            chosen_voice = pick_voice_with_hysteresis(
                voices, target_f0, language, speaker_id
            )

            t0 = time.monotonic()
            samples, sr = synthesize_kokoro(kokoro, text, chosen_voice, language)
            t_kokoro = time.monotonic() - t0

            # Run pyworld only when the residual pitch shift is large
            # enough to be perceptible AND the formant shift is non-
            # trivial. Multi-voice routing already picks the base voice
            # closest to the target F0; if the residual is small, the
            # WORLD analysis-synthesis pass introduces artefacts ("fanha"
            # robotic colour) that outweigh the differentiation it
            # would provide. Threshold 25 Hz: roughly the pitch step
            # between adjacent semitones at human-voice frequencies —
            # below that, the listener cannot reliably tell the shifted
            # output from the unshifted one anyway.
            shift_skip_threshold_hz = 25.0
            source_f0 = _KOKORO_F0_CACHE.get(chosen_voice, 0.0)
            f0_delta = abs(target_f0 - source_f0) if (target_f0 > 0 and source_f0 > 0) else 0.0
            formant_warps = abs(formant_shift - 1.0) > 1e-3
            should_pyworld = (
                samples.size > int(sr * 0.2)
                and source_f0 > 0
                and (f0_delta > shift_skip_threshold_hz or formant_warps)
            )

            if should_pyworld:
                t1 = time.monotonic()
                samples = shift_pitch_and_formant(
                    samples, sr, source_f0, target_f0, formant_shift
                )
                t_shift = time.monotonic() - t1
                log(
                    f"[tts] voice={chosen_voice} kokoro={t_kokoro * 1000:.0f}ms "
                    f"shift={t_shift * 1000:.0f}ms "
                    f"src_f0={source_f0:.0f} → target_f0={target_f0:.0f} "
                    f"formant={formant_shift:.2f}"
                )
            else:
                # Pristine Kokoro output — no pyworld touch. This is the
                # path 80-90 % of utterances take after multi-voice
                # routing picks an appropriate base.
                log(
                    f"[tts] voice={chosen_voice} kokoro={t_kokoro * 1000:.0f}ms "
                    f"(natural; src_f0={source_f0:.0f}, target_f0={target_f0:.0f}, "
                    f"delta={f0_delta:.0f}Hz < {shift_skip_threshold_hz:.0f}Hz)"
                )

            write_pcm_response(samples, sr)
        except Exception as e:
            log(f"TTS error: {type(e).__name__}: {e}")
            import traceback
            log(traceback.format_exc())
            write_pcm_response(np.zeros(1, dtype=np.float32), KOKORO_OUTPUT_SAMPLE_RATE)


if __name__ == "__main__":
    main()
