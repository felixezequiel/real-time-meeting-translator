"""
TTS bridge using Piper + pyworld pitch/formant shifting.

Why this approach (vs. the CosyVoice 2 zero-shot cloning we tried before):

  CosyVoice 2 generates each phoneme conditioned on a reference voice
  via a 500M-parameter LLM. The result is a faithful clone, but on a
  laptop GPU (RTX 3050 6 GB) it costs 2-4 s per utterance even fully
  optimised — well outside the simultaneous-translation budget of
  ~2 s end-to-end.

  Piper synthesises in ~150 ms from a 25 MB ONNX voice. We then run
  pyworld analysis-synthesis on the output: extract F0 (pitch), the
  spectral envelope (timbre/formants) and aperiodicity, swap the F0
  for the target speaker's running F0, optionally shift the spectral
  envelope, and re-synthesise. Total TTS step: ~250-350 ms. End-to-
  end latency stays under 2 s and the listener hears voices that
  shift in pitch and vocal weight when different speakers talk —
  enough to "tell who's speaking" without a real clone.

Protocol (binary-framed, mirrors previous bridges):

- Startup (text line on stdout):
    {"status": "ready"}\\n

- Request (text line on stdin):
    {"text": "...",
     "language": "en|pt|en-us|pt-br",
     "target_f0": 180.0,            (optional, Hz; speaker's running mean
                                     F0 — TTS output is pitch-shifted to
                                     match. Omit for default Piper voice.)
     "formant_shift": 1.0           (optional ratio; >1 enlarges vocal
                                     tract → deeper voice, <1 narrows →
                                     thinner voice. 1.0 = no change.)
    }\\n

- Response (binary):
    {"sample_rate": 22050, "num_samples": N}\\n
    <N * 2 bytes int16 little-endian PCM>

Requires:
  pip install piper-tts pyworld numpy
Models (~25 MB each) are downloaded by Install-PiperVoices in
`scripts/install.ps1` to %TEMP%/piper_voices/.
"""

import json
import os
import sys
import tempfile
import time

import numpy as np


# stdout MUST stay binary so we can interleave a JSON header with raw
# PCM bytes in a single response. Mixing TextIOWrapper writes and
# binary writes corrupts the framing.
stdout_bin = sys.stdout.buffer
sys.stdin = __import__("io").TextIOWrapper(sys.stdin.buffer, encoding="utf-8")


# Multi-voice routing: each language has both a male and (when
# available on rhasspy/piper-voices) a female Piper voice. At synthesis
# time we pick whichever base voice has the F0 closest to the target,
# then apply a SMALLER pitch shift on top — large shifts (>1.3x) in
# pyworld sound robotic, but a 1.05-1.15x shift over a closer base
# stays clean. PT-BR currently has no female Piper voice on the
# rhasspy registry; high-F0 PT speakers fall back to shifting Faber
# more aggressively (best we can do until a female PT voice ships).
PIPER_VOICES_BY_LANG = {
    "en": [
        "en_US-ryan-medium",    # male, ~120 Hz
        "en_US-amy-medium",     # female, ~200 Hz
    ],
    "en-us": [
        "en_US-ryan-medium",
        "en_US-amy-medium",
    ],
    "pt": [
        "pt_BR-faber-medium",   # male, ~150 Hz
    ],
    "pt-br": [
        "pt_BR-faber-medium",
    ],
}

PIPER_VOICE_URLS = {
    "en_US-ryan-medium": [
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
    ],
    "en_US-amy-medium": [
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
    ],
    "pt_BR-faber-medium": [
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx.json",
    ],
}

# Empirically measured F0 (cached after the first calibration call):
# {voice_name: mean_F0_hz}. Used at runtime to pick the base voice
# closest to the target F0, minimising the shift required.
_PIPER_F0_CACHE: dict[str, float] = {}


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


def download_piper_voice(voice_name: str, voices_dir: str) -> None:
    import urllib.request

    urls = PIPER_VOICE_URLS.get(voice_name, [])
    for url in urls:
        filename = os.path.basename(url)
        local_path = os.path.join(voices_dir, filename)
        if not os.path.exists(local_path):
            log(f"Downloading {filename}…")
            urllib.request.urlretrieve(url, local_path)
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            log(f"  Downloaded: {size_mb:.1f} MB")


def setup_piper(voices_dir: str) -> dict:
    """Load every Piper voice referenced by `PIPER_VOICES_BY_LANG`.

    Returns: {language: [(voice_name, PiperVoice), ...]}. The list is
    in arbitrary order — picking the right voice for a given target_f0
    happens at synthesis time via `pick_voice_for_f0`.
    """
    from piper import PiperVoice

    loaded: dict[str, list[tuple[str, "PiperVoice"]]] = {}
    voice_cache: dict[str, "PiperVoice"] = {}

    for language, voice_names in PIPER_VOICES_BY_LANG.items():
        for voice_name in voice_names:
            if voice_name in voice_cache:
                voice = voice_cache[voice_name]
            else:
                onnx_path = os.path.join(voices_dir, f"{voice_name}.onnx")
                json_path = os.path.join(voices_dir, f"{voice_name}.onnx.json")
                if not os.path.exists(onnx_path):
                    download_piper_voice(voice_name, voices_dir)
                if not os.path.exists(onnx_path):
                    log(f"  WARNING: {voice_name} not available (download failed?), skipping")
                    continue
                log(f"Loading Piper voice: {voice_name}")
                voice = PiperVoice.load(onnx_path, config_path=json_path)
                voice_cache[voice_name] = voice
                log(f"  ready (sr={voice.config.sample_rate})")
            loaded.setdefault(language, []).append((voice_name, voice))
    return loaded


def synthesize_piper(
    piper_voices: dict,
    text: str,
    language: str,
    voice_name: str | None = None,
) -> tuple[np.ndarray, int, str]:
    """Run Piper TTS synchronously with a specific voice.

    When `voice_name` is None, the first voice registered for the
    language is used (whichever happened to load first — typically the
    male voice). The caller normally selects the right voice via
    `pick_voice_for_f0` before calling this.

    Returns: (samples_f32, sample_rate, voice_name_used).
    """
    voices = (piper_voices.get(language)
              or piper_voices.get(language.split("-")[0])
              or [])
    if not voices:
        return np.zeros(1, dtype=np.float32), 22050, ""

    # Find the requested voice; fall back to the first available.
    selected = None
    if voice_name:
        for vn, v in voices:
            if vn == voice_name:
                selected = (vn, v)
                break
    if selected is None:
        selected = voices[0]
    name_used, voice = selected
    sr = voice.config.sample_rate

    chunks = []
    for chunk in voice.synthesize(text):
        chunks.append(chunk.audio_float_array)
    if not chunks:
        return np.zeros(1, dtype=np.float32), sr, name_used
    return np.concatenate(chunks).astype(np.float32), sr, name_used


def pick_voice_for_f0(
    piper_voices: dict,
    language: str,
    target_f0: float,
) -> str | None:
    """Return the name of the Piper voice with the closest mean F0 to
    `target_f0`. Requires every voice to have been measured already
    (call `measure_all_voices` once at startup). Returns None when no
    voices are loaded for the language.
    """
    voices = (piper_voices.get(language)
              or piper_voices.get(language.split("-")[0])
              or [])
    if not voices:
        return None
    if target_f0 <= 0:
        return voices[0][0]
    best_name, best_dist = None, float("inf")
    for name, _v in voices:
        f0 = _PIPER_F0_CACHE.get(name)
        if f0 is None or f0 <= 0:
            continue
        dist = abs(f0 - target_f0)
        if dist < best_dist:
            best_name, best_dist = name, dist
    return best_name or voices[0][0]


def measure_all_voices(piper_voices: dict) -> None:
    """Synthesise a calibration phrase for every loaded voice and cache
    its mean F0. Done once at startup so per-call routing is just a
    dict lookup."""
    seen = set()
    for language, voices in piper_voices.items():
        for voice_name, _v in voices:
            if voice_name in seen:
                continue
            seen.add(voice_name)
            calibration_text = (
                "Hello world, this is a test."
                if language.startswith("en")
                else "Olá mundo, isto é um teste."
            )
            samples, sr, _ = synthesize_piper(piper_voices, calibration_text, language, voice_name)
            f0 = measure_voice_f0(samples, sr) if samples.size > int(sr * 0.5) else 0.0
            _PIPER_F0_CACHE[voice_name] = f0
            log(f"[init] piper voice {voice_name} measured F0={f0:.1f} Hz")


def measure_voice_f0(samples: np.ndarray, sample_rate: int) -> float:
    """Mean F0 over voiced frames of `samples`. Used once per Piper voice
    to compute the baseline pitch we shift away from. ~30 ms for 1 s of
    audio on CPU."""
    import pyworld as pw

    f0, _t = pw.harvest(
        samples.astype(np.float64),
        sample_rate,
        f0_floor=70.0,
        f0_ceil=400.0,
        frame_period=10.0,
    )
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        return 120.0  # adult male default
    return float(np.mean(voiced))


def shift_pitch_and_formant(
    samples: np.ndarray,
    sample_rate: int,
    source_f0: float,
    target_f0: float,
    formant_shift: float,
) -> np.ndarray:
    """Apply pitch and formant shift via WORLD analysis-synthesis.

    Args:
        samples: mono f32 in [-1, 1].
        sample_rate: input/output sample rate (24 kHz works best for WORLD).
        source_f0: mean F0 of `samples` (precomputed).
        target_f0: desired mean F0 in Hz.
        formant_shift: spectral-envelope warp ratio (1.0 = no change).

    Returns: mono f32 with the shifted pitch/formants, same sample rate.
    Falls back to the input on any internal error so the pipeline never
    drops a TTS chunk.
    """
    import pyworld as pw

    try:
        # pyworld is a C-extension that requires C-contiguous float64
        # arrays for every input. astype() and arithmetic ops can leave
        # the result non-contiguous (e.g. a strided slice or a transpose);
        # wrap every array we hand to pw.* in ascontiguousarray to make
        # the contract explicit. Skipping these wrappers is what produced
        # the "ValueError: ndarray is not C-contiguous" we used to see on
        # every synthesis call.
        x = np.ascontiguousarray(samples.astype(np.float64))
        # Tighten F0 search range around the source voice. Tighter
        # bounds give pyworld DIO/Harvest cleaner pitch tracks (fewer
        # octave errors), which directly translates to fewer artefacts
        # in the resynthesised output.
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

        # Pitch shift ratio: tighter clamp than before — pyworld stays
        # natural up to ~1.4x; beyond that timbre starts to feel
        # "underwater" / robotic. With multi-voice routing we should
        # rarely need more than 1.2x because the closest base voice is
        # already picked.
        ratio = target_f0 / source_f0 if source_f0 > 0 else 1.0
        ratio = float(np.clip(ratio, 0.7, 1.4))
        f0_shifted = np.ascontiguousarray(f0 * ratio)

        # Formant shift: warp the spectral envelope along the frequency
        # axis. >1 stretches the envelope (deeper / "bigger" voice),
        # <1 compresses (thinner). 1.0 means leave envelope alone.
        if abs(formant_shift - 1.0) > 1e-3:
            n_bins = sp.shape[1]
            old_idx = np.arange(n_bins)
            new_idx = old_idx / formant_shift
            new_idx = np.clip(new_idx, 0, n_bins - 1)
            lower = np.floor(new_idx).astype(np.int32)
            upper = np.minimum(lower + 1, n_bins - 1)
            frac = new_idx - lower
            # Fancy-indexing with `sp[:, lower]` returns a non-contiguous
            # view; the multiplication produces a non-contiguous array
            # too. Wrap the result so pw.synthesize sees C-contiguous.
            sp = np.ascontiguousarray(
                sp[:, lower] * (1.0 - frac) + sp[:, upper] * frac
            )

        sp = np.ascontiguousarray(sp)
        ap = np.ascontiguousarray(ap)
        y = pw.synthesize(f0_shifted, sp, ap, sample_rate, frame_period=10.0)
        return y.astype(np.float32)
    except Exception as e:  # noqa: BLE001
        log(f"pitch/formant shift failed, returning original: {type(e).__name__}: {e}")
        return samples


def main() -> None:
    voices_dir = os.path.join(tempfile.gettempdir(), "piper_voices")
    os.makedirs(voices_dir, exist_ok=True)

    log("Setting up Piper TTS…")
    piper_voices = setup_piper(voices_dir)
    if not piper_voices:
        log("FATAL: no Piper voices loaded — check the voices_dir and download URLs.")
        sys.exit(2)
    unique_voices = {name for vs in piper_voices.values() for name, _ in vs}
    log(f"Piper ready ({len(unique_voices)} unique voices: {sorted(unique_voices)})")

    log("[init] measuring per-voice F0 (one-time calibration)…")
    measure_all_voices(piper_voices)

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

            if not text:
                write_pcm_response(np.zeros(1, dtype=np.float32), 22050)
                continue

            # Voice routing: pick the loaded base voice with the closest
            # mean F0 to the target. For an English female speaker
            # (~220 Hz) this picks Amy (~200 Hz) instead of Ryan
            # (~120 Hz), which means the subsequent shift is ~1.1x
            # instead of ~1.8x — pyworld stays well inside its clean
            # range and the output sounds natural rather than robotic.
            chosen_voice_name = pick_voice_for_f0(piper_voices, language, target_f0)

            t0 = time.monotonic()
            samples, sr, voice_used = synthesize_piper(
                piper_voices, text, language, chosen_voice_name
            )
            t_piper = time.monotonic() - t0

            if target_f0 > 0 and samples.size > int(sr * 0.2):
                source_f0 = _PIPER_F0_CACHE.get(voice_used, 0.0)
                if source_f0 > 0:
                    t1 = time.monotonic()
                    samples = shift_pitch_and_formant(
                        samples, sr, source_f0, target_f0, formant_shift
                    )
                    t_shift = time.monotonic() - t1
                    log(
                        f"[tts] voice={voice_used} piper={t_piper * 1000:.0f}ms "
                        f"shift={t_shift * 1000:.0f}ms "
                        f"src_f0={source_f0:.0f} → target_f0={target_f0:.0f} "
                        f"formant={formant_shift:.2f}"
                    )
                else:
                    log(f"[tts] voice={voice_used} piper={t_piper * 1000:.0f}ms (no source F0)")
            else:
                log(f"[tts] voice={voice_used} piper={t_piper * 1000:.0f}ms (no shift)")

            write_pcm_response(samples, sr)
        except Exception as e:  # noqa: BLE001
            log(f"TTS error: {type(e).__name__}: {e}")
            import traceback
            log(traceback.format_exc())
            write_pcm_response(np.zeros(1, dtype=np.float32), 22050)


if __name__ == "__main__":
    main()
