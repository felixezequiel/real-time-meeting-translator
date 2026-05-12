"""
XTTS-v2 voice-cloning TTS bridge — replaces Kokoro + OpenVoice TCC
with a single zero-shot voice-cloning model in one stage (ADR 0014).

The wire protocol matches `tts_bridge.py` so the Rust client
(`crates/tts/src/lib.rs::PiperTts`) doesn't change — main.rs just
spawns this script instead based on `config.tts_engine = "xtts"`.

Key behavioural differences vs Kokoro:
  * `reference_wav_path` is REQUIRED. Without it the bridge emits
    silence (Kokoro fell back to a default voice; XTTS-v2 has no
    "default" — its output IS the reference voice).
  * `target_f0`, `formant_shift`, `speaker_id` are silently ignored
    (XTTS infers prosody and timbre from the reference audio).
  * Output sample rate is fixed at 24 kHz, like Kokoro. The
    playback resampler on the Rust side handles 24 → 48 kHz.

Protocol:

  Startup: {"status": "ready"}\\n

  Request (one JSON line, no PCM payload):
    {"text": "...",
     "language": "en" | "pt",
     "reference_wav_path": "C:\\\\path\\\\to\\\\ref.wav",
     ...kokoro fields ignored...}\\n

  Response (multi-frame streaming):
    For each chunk produced by `model.inference_stream`:
      {"sample_rate": 24000, "num_samples": N, "final": false}\\n
      <N * 2 bytes int16 LE PCM mono>
    Then exactly one terminator frame at the end:
      {"sample_rate": 24000, "num_samples": 0, "final": true}\\n
      <0 bytes>

    The Rust client reads frames in a loop until `final=true`. Atomic
    bridges (Kokoro `tts_bridge.py`) emit a single frame with the full
    audio payload AND `final=true`; the same client code handles both.

Install:
  pip install coqui-tts          # maintained fork of Coqui TTS
  pip install torch torchaudio   # already required by other bridges

Model: tts_models/multilingual/multi-dataset/xtts_v2 (~1.8 GB).
Downloaded automatically by `Install-XttsModel` in install.ps1, or
on first run if the model is missing (HF fallback).

Device: respects `XTTS_DEVICE` env var (defaults to cuda when
available, cpu otherwise — same convention as ADR 0011 follow-up
for the Sepformer bridge).
"""

import json
import os
import sys
import time
import warnings

import numpy as np


stdout_bin = sys.stdout.buffer
stdin_bin = sys.stdin.buffer
sys.stdin = __import__("io").TextIOWrapper(sys.stdin.buffer, encoding="utf-8")

XTTS_OUTPUT_SAMPLE_RATE = 24_000

# Per-process cache of (gpt_cond_latent, speaker_embedding) keyed by
# the absolute reference path. Computing latents from a 6-second WAV
# costs ~150-300 ms on GPU; for consecutive clauses of the same
# speaker we'd pay that on every fragment without the cache.
_LATENT_CACHE: dict[str, tuple] = {}


def find_fallback_reference() -> str | None:
    """Resolve a fallback reference WAV used when the request gives
    no `reference_wav_path` (e.g. a brand-new speaker_id whose
    auto-enrolment hasn't finished yet). The first phrase of any
    new speaker is otherwise emitted as silence — the user reasonably
    asked to hear *something* even if the voice isn't yet calibrated
    to that person.

    Search order (project-relative, mirrors how other bridges find
    their assets):
      1. `models/xtts_v2/fallback_voice.wav` — explicit fallback,
         shippable, takes precedence.
      2. `voice_profile/user.wav` — the user's own recorded profile.
         Translations in the user's voice while waiting for the real
         speaker enrolment is unusual but unmistakably "real human"
         audio, and the listener instantly knows it's a placeholder.
    Returns None when nothing is available — bridge then falls back
    to silence (legacy behaviour).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    candidates = [
        os.path.join(project_root, "models", "xtts_v2", "fallback_voice.wav"),
        os.path.join(project_root, "voice_profile", "user.wav"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def write_json_line(obj: dict) -> None:
    stdout_bin.write((json.dumps(obj, ensure_ascii=True) + "\n").encode("utf-8"))
    stdout_bin.flush()


def write_pcm_frame(samples_np: np.ndarray, sample_rate: int, final: bool) -> None:
    """Emit one PCM frame in the streaming protocol. Each frame has a
    JSON header carrying `final`; the Rust client reads frames in a
    loop until it sees `final=true`. Mid-stream frames carry audio;
    the terminator frame is empty (`num_samples=0`).
    """
    # Defensive: replace NaN / Inf with silence BEFORE clipping.
    # Field test 2026-05-08: GPU under heavy contention (Whisper +
    # Qwen + 2× XTTS + diariser on 6 GB) produced NaN-laced output
    # tensors because cuDNN autotune picked numerically unstable
    # algorithms. NaN * 32767 → 0 on int16 cast → completely silent
    # output even though the bridge logged a non-zero sample count.
    # `nan_to_num` converts those into proper zeros so at minimum
    # the failure is visible (audible silence) instead of mysterious.
    samples_np = np.nan_to_num(samples_np, nan=0.0, posinf=1.0, neginf=-1.0)
    samples_np = np.clip(samples_np, -1.0, 1.0)
    # Diagnostic: log the peak amplitude so the operator can tell
    # silence-from-NaN apart from silence-from-zero-energy at a glance.
    # Only check on non-terminator frames — the terminator is meant
    # to be empty and a peak warning there is noise.
    if samples_np.size > 0 and not final:
        peak = float(np.max(np.abs(samples_np)))
        if peak < 0.001:
            log(f"[xtts] WARNING: synthesised audio peak={peak:.4f} (effectively silent)")
    int16 = (samples_np * 32767.0).astype(np.int16)
    pcm_bytes = int16.tobytes()
    header = {
        "sample_rate": int(sample_rate),
        "num_samples": int(int16.size),
        "final": bool(final),
    }
    stdout_bin.write((json.dumps(header) + "\n").encode("utf-8"))
    stdout_bin.write(pcm_bytes)
    stdout_bin.flush()


def write_pcm_terminator(sample_rate: int) -> None:
    """Emit the empty terminator frame that marks end-of-stream."""
    write_pcm_frame(np.zeros(0, dtype=np.float32), sample_rate, final=True)


# ─── Model loading ──────────────────────────────────────────────────────────

def select_device() -> str:
    """Resolve the runtime device. Override via `XTTS_DEVICE` env var
    (mirrors `SEPFORMER_DEVICE` from ADR 0011 follow-up). Falls back
    to CPU when CUDA is requested but unavailable."""
    import torch
    requested = os.environ.get("XTTS_DEVICE", "").strip()
    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            log(f"[xtts] CUDA requested via XTTS_DEVICE but unavailable — using CPU")
            return "cpu"
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def find_xtts_model_dir() -> str | None:
    """Locate the on-disk XTTS-v2 model directory. Search order:
       1. `models/xtts_v2/` next to the executable (installed mode)
       2. `models/xtts_v2/` next to the script (dev mode)
       3. `XTTS_MODEL_DIR` environment variable
    Returns None if nothing is found — the caller then falls back to
    auto-download via TTS's ModelManager.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    candidates = [
        os.path.join(project_root, "models", "xtts_v2"),
        os.path.join(os.getcwd(), "models", "xtts_v2"),
    ]
    env_dir = os.environ.get("XTTS_MODEL_DIR")
    if env_dir:
        candidates.insert(0, env_dir)
    for candidate in candidates:
        config_json = os.path.join(candidate, "config.json")
        if os.path.isfile(config_json):
            return candidate
    return None


class XttsBridge:
    """Wraps the XTTS-v2 model with a per-reference latent cache and
    a streaming-aware synthesise call. The model is initialised once
    on bridge startup; subsequent requests share the loaded weights.
    """

    def __init__(self) -> None:
        # Suppress Coqui's loud "best practices" prints on import —
        # they interleave with our stderr logs and confuse the Rust
        # side that watches for the "ready" line on stdout.
        warnings.filterwarnings("ignore")

        device = select_device()
        log(f"[xtts] device: {device}")

        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        model_dir = find_xtts_model_dir()
        if model_dir is None:
            log("[xtts] no local model dir found — downloading via TTS.api …")
            from TTS.api import TTS as TtsApi
            # Trigger Coqui's model download into ~/.local/share/tts/.
            # We then resolve the cache path from the manager.
            api = TtsApi()
            mgr = api.manager
            local_path, _, _ = mgr.download_model(
                "tts_models/multilingual/multi-dataset/xtts_v2"
            )
            model_dir = os.path.dirname(local_path)
            log(f"[xtts] model cached at {model_dir}")

        config_path = os.path.join(model_dir, "config.json")
        log(f"[xtts] loading config from {config_path}")
        config = XttsConfig()
        config.load_json(config_path)

        # cudnn.benchmark autotunes convolution algorithms on the
        # first few forward passes. Enabling it BEFORE the warmup
        # below means the autotune cost is paid during init, not
        # during the user's first real translation. Field test
        # 2026-05-08 saw the first inference take 53 s (RTF ~9!)
        # because cudnn was discovering optimal kernels live; that
        # cold-start cost is what this flag + the warmup eliminate.
        import torch
        torch.backends.cudnn.benchmark = True

        log(f"[xtts] loading XTTS-v2 weights …")
        t = time.monotonic()
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_dir=model_dir,
            use_deepspeed=False,
        )
        if device.startswith("cuda"):
            model.cuda()
        load_ms = (time.monotonic() - t) * 1000
        log(f"[xtts] weights loaded ({load_ms:.0f} ms)")

        self._model = model
        self._device = device

        # Pre-resolve the fallback reference once at startup so the
        # main loop doesn't pay a `os.path.isfile` per request.
        self._fallback_reference = find_fallback_reference()
        if self._fallback_reference:
            log(
                f"[xtts] fallback reference: "
                f"{os.path.basename(self._fallback_reference)} "
                f"(used when a request has no reference_wav_path)"
            )
        else:
            log("[xtts] no fallback reference found — silence on enrolment gap")

        # Warmup: run 2 dummy inferences so cudnn.benchmark and
        # model-internal lazy initialisation pay their costs HERE,
        # not on the user's first real phrase. Without this, field
        # logs showed conditioning latents taking 51 s and the first
        # inference 53 s — the queue piled up faster than the bridge
        # drained it and nothing reached the listener.
        if self._fallback_reference:
            self._warmup()
        else:
            log(
                "[xtts] skipping warmup (no fallback reference) — first "
                "real synthesis WILL pay the cold-start cost"
            )

        log(f"[xtts] ready (total init: {(time.monotonic() - t) * 1000:.0f} ms)")

    def _warmup(self) -> None:
        """Pay cuDNN autotune up front by running dummy inferences in
        BOTH directions and across BOTH the atomic and streaming code
        paths. cudnn picks different convolution algorithms for the
        two paths (different shapes / sequence lengths), so warming
        only one leaves the other paying ~50 s on its first real
        request — which is what triggered the 2026-05-08 latency
        regression after we switched the hot path to streaming.
        """
        log("[xtts] warming up (atomic + streaming, 2 langs) …")
        warmup_start = time.monotonic()
        try:
            gpt_cond, spk_emb = self.get_conditioning(self._fallback_reference)
            for i, (text, lang) in enumerate([
                ("Olá mundo.", "pt"),
                ("Hello world.", "en"),
            ]):
                t_atomic = time.monotonic()
                self._model.inference(
                    text=text,
                    language=lang,
                    gpt_cond_latent=gpt_cond,
                    speaker_embedding=spk_emb,
                    temperature=0.65,
                    enable_text_splitting=False,
                )
                log(
                    f"[xtts] warmup {i + 1}/2 atomic ({lang}) "
                    f"in {(time.monotonic() - t_atomic) * 1000:.0f} ms"
                )

                t_stream = time.monotonic()
                stream_chunks = self._model.inference_stream(
                    text=text,
                    language=lang,
                    gpt_cond_latent=gpt_cond,
                    speaker_embedding=spk_emb,
                    stream_chunk_size=20,
                    temperature=0.65,
                    enable_text_splitting=False,
                )
                # Drain the iterator so cudnn pays the streaming
                # autotune cost here too.
                chunk_count = 0
                for _ in stream_chunks:
                    chunk_count += 1
                log(
                    f"[xtts] warmup {i + 1}/2 streaming ({lang}) "
                    f"{chunk_count} chunks "
                    f"in {(time.monotonic() - t_stream) * 1000:.0f} ms"
                )
        except Exception as e:  # noqa: BLE001
            log(f"[xtts] warmup failed (non-fatal): {type(e).__name__}: {e}")
        log(f"[xtts] warmup done in {(time.monotonic() - warmup_start) * 1000:.0f} ms")

    def get_conditioning(self, ref_path: str):
        """Cache (gpt_cond_latent, speaker_embedding) per reference
        WAV. The cache invalidation strategy is "process lifetime":
        VoiceProfileRegistry rewrites the file when it has fresh
        audio for a speaker, but the path stays the same and the
        latents stay valid for the full session.
        """
        cached = _LATENT_CACHE.get(ref_path)
        if cached is not None:
            return cached
        t = time.monotonic()
        gpt_cond_latent, speaker_embedding = self._model.get_conditioning_latents(
            audio_path=[ref_path],
            gpt_cond_len=6,
        )
        dt = (time.monotonic() - t) * 1000
        log(f"[xtts] conditioning latents for {os.path.basename(ref_path)} in {dt:.0f} ms")
        _LATENT_CACHE[ref_path] = (gpt_cond_latent, speaker_embedding)
        return gpt_cond_latent, speaker_embedding

    def synthesize_stream(
        self,
        text: str,
        language: str,
        ref_path: str,
        on_chunk,
    ) -> int:
        """Streaming synthesis: invokes `on_chunk(samples_np)` for every
        ~250 ms PCM fragment XTTS produces, returning the total number
        of samples emitted across all fragments.

        Why streaming again (after the 2026-05-08 atomic regression):
        the previous atomic switch hid the fragment-internal RTF cost
        (~RTF 0.6 on RTX 3050) by accumulating the full inference
        before sending. Once we collapsed to a SINGLE PiperTts and
        reduced VRAM pressure, the RTF stabilised low enough that
        emitting chunks as they're produced cuts time-to-first-audio
        per fragment from ~`full_inference_ms` to ~`first_chunk_ms`
        (~250-500 ms typical). Adjacent chunks of the same utterance
        play back-to-back without gaps because the mixer skips its
        per-chunk envelope on `is_streaming_chunk` chunks.

        Raises if the reference is missing.
        """
        if not os.path.isfile(ref_path):
            raise FileNotFoundError(
                f"reference WAV missing: {ref_path}"
            )

        gpt_cond_latent, speaker_embedding = self.get_conditioning(ref_path)

        chunks = self._model.inference_stream(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            # `stream_chunk_size` is in tokens. 20 produces ~250-400 ms
            # PCM chunks at XTTS's 24 kHz output — short enough for
            # snappy time-to-first-audio, long enough that decode-loop
            # overhead doesn't dominate. Lowering to 10 made first
            # audio land sooner but pushed RTF >1.5 (the previous
            # atomic-switch trigger).
            stream_chunk_size=20,
            temperature=0.65,
            # `speed` is XTTS-v2's prosody-rate knob (multiplier on the
            # internal duration prediction). 1.0 produces ~15 chars/s
            # in pt-BR — noticeably slower than native conversational
            # cadence (18-25 chars/s). 1.25 brings it back to natural
            # speed and as a side-effect shortens the synthesised
            # audio by ~20 %, which directly reduces the mixer-queue
            # depth that drives our backlog. Tunable: 1.20 (gentler)
            # to 1.35 (sounds rushed) work; >1.4 starts distorting
            # vowels. Matches Kokoro's 1.30 default set on 2026-05-10.
            speed=1.25,
            enable_text_splitting=False,  # the V2 accumulator already split
        )

        total_samples = 0
        for chunk in chunks:
            # `inference_stream` yields a torch Tensor on the model's
            # device. Move to CPU and float32 for protocol emission.
            try:
                wav = chunk.detach().cpu().numpy().astype(np.float32)
            except AttributeError:
                # Fallback if the iterator yielded a numpy array (older
                # Coqui versions) — passthrough.
                wav = np.asarray(chunk, dtype=np.float32)
            if wav.size == 0:
                continue
            on_chunk(wav)
            total_samples += int(wav.size)
        return total_samples


# ─── Main loop ─────────────────────────────────────────────────────────────

LANGUAGE_MAP = {
    "en": "en",
    "en-us": "en",
    "pt": "pt",
    "pt-br": "pt",
}


def main() -> None:
    bridge = XttsBridge()
    write_json_line({"status": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            text = (request.get("text") or "").strip()
            raw_lang = (request.get("language") or "en").strip().lower()
            language = LANGUAGE_MAP.get(raw_lang, "en")
            ref_path = (request.get("reference_wav_path") or "").strip()

            if not text:
                # Empty text: emit just the terminator so the Rust
                # client unblocks immediately with zero audio.
                write_pcm_terminator(XTTS_OUTPUT_SAMPLE_RATE)
                continue

            if not ref_path:
                # No real reference yet for this speaker. Use the
                # session-wide fallback so the listener still hears
                # SOMETHING (the user explicitly asked: "mesmo que
                # robotica"). When auto-enrolment finishes ~6 s
                # later, real reference replaces this and the voice
                # switches to the actual speaker's timbre.
                if bridge._fallback_reference is not None:
                    ref_path = bridge._fallback_reference
                    log(
                        f"[xtts] no reference for text=\"{text[:40]}…\" "
                        f"({raw_lang}) — using fallback "
                        f"{os.path.basename(ref_path)}"
                    )
                else:
                    log(
                        f"[xtts] no reference for text=\"{text[:40]}…\" "
                        f"({raw_lang}) and no fallback configured — "
                        f"emitting silence"
                    )
                    write_pcm_terminator(XTTS_OUTPUT_SAMPLE_RATE)
                    continue

            # Stream chunks as they're produced. Each on-chunk callback
            # writes one PCM frame with `final=false`; after the
            # iterator drains we emit the terminator.
            t = time.monotonic()
            first_chunk_logged = False
            chunks_emitted = 0

            def emit(wav: np.ndarray) -> None:
                nonlocal first_chunk_logged, chunks_emitted
                if not first_chunk_logged:
                    log(
                        f"[xtts] first chunk in "
                        f"{(time.monotonic() - t) * 1000:.0f} ms "
                        f"({wav.size} samples, {language}, "
                        f"ref={os.path.basename(ref_path)})"
                    )
                    first_chunk_logged = True
                chunks_emitted += 1
                write_pcm_frame(wav, XTTS_OUTPUT_SAMPLE_RATE, final=False)

            total_samples = bridge.synthesize_stream(
                text, language, ref_path, emit,
            )
            dt = (time.monotonic() - t) * 1000
            log(
                f"[xtts] {total_samples} samples in {chunks_emitted} chunks "
                f"({XTTS_OUTPUT_SAMPLE_RATE} Hz) for {len(text)} chars / "
                f"{language} in {dt:.0f} ms (ref={os.path.basename(ref_path)})"
            )
            write_pcm_terminator(XTTS_OUTPUT_SAMPLE_RATE)

        except Exception as e:  # noqa: BLE001 — bridge must stay alive
            log(f"[xtts] synthesis error: {type(e).__name__}: {e}")
            import traceback
            log(traceback.format_exc())
            # Rust expects a frame stream that ends in `final=true` for
            # every request — emit just the terminator so the protocol
            # stays in sync without queueing bogus audio.
            write_pcm_terminator(XTTS_OUTPUT_SAMPLE_RATE)


if __name__ == "__main__":
    main()
