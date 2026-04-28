"""
Tone-color conversion bridge — OpenVoice v2 TCC.

Why this exists: Kokoro TTS produces a fixed catalog of voices
(am_michael, af_bella, …). Even with multi-voice routing and pyworld
pitch shifting, two different speakers in a documentary who share the
same gender and rough pitch end up sounding like the SAME Kokoro
voice. The user can't tell them apart.

This bridge takes Kokoro's output PCM + a reference WAV recorded
from the actual person and rewrites the timbre (vocal-tract
resonance, formants) onto the synthesised audio. The prosody, words
and rhythm stay Kokoro's; the "voice colour" becomes the real
person's. Good enough that two adult men in the same panel sound
like distinct individuals.

Trade-offs (vs the WORLD pitch-shift path that runs in tts_bridge):
  - Quality: substantially better. WORLD only swaps F0 and (used to)
    warp a single global formant ratio; OpenVoice TCC is a learned
    flow model trained on multilingual speech so it captures detailed
    spectral envelope dynamics that WORLD cannot.
  - Latency: +150-250 ms per fragment on GPU. Within the budget.
  - Setup: requires cloning the OpenVoice v2 repo into
    `third_party/OpenVoice/` and downloading their TCC checkpoint
    (~50 MB) into `models/openvoice/`. Both are handled by
    `Install-OpenVoice` in `scripts/install.ps1`.
  - Failure mode: if either the repo or the checkpoint is missing,
    the bridge logs a warning and exits cleanly. The Rust client
    treats the bridge as dead and the pipeline degrades to
    pyworld-only voice differentiation — never a hard failure.

Protocol (binary-framed, mirrors the tts_bridge protocol):

- Startup (text on stdout):
    {"status": "ready"}\\n

- Request:
    {"source_sr": 24000, "source_num_samples": N,
     "reference_wav_path": "C:/.../ref.wav",
     "speaker_id": 3}\\n
    <N * 2 bytes int16 LE PCM (the synthesised audio from Kokoro)>

- Response:
    {"sample_rate": 22050, "num_samples": M}\\n
    <M * 2 bytes int16 LE PCM (the converted audio)>

The output sample rate is fixed at 22050 because OpenVoice TCC
operates natively at that rate and resampling on the Python side
saves a stage from the Rust playback resampler.

Caches:
  - `_REF_SE_CACHE`: speaker-embedding tensor per reference WAV path.
    Computed once per ref (~150 ms); subsequent calls with the same
    path are instantaneous.
  - `_SOURCE_SE`: a single SE for the Kokoro output domain, computed
    from the FIRST request's source audio. Stays valid for the
    lifetime of the bridge — Kokoro voices share enough spectral
    statistics that a single source SE works for all of them.
"""

import json
import os
import sys
import tempfile
import time
import wave

import numpy as np


stdout_bin = sys.stdout.buffer
stdin_bin = sys.stdin.buffer


# OpenVoice TCC native sample rate. We resample input to this rate and
# return output at this rate — the pipeline's downstream resampler
# (in `crates/audio/src/playback.rs`) handles the device rate.
TCC_SAMPLE_RATE = 22_050

# How many seconds of reference audio TCC's SE extractor needs to
# produce a stable embedding. Less than ~3 s and the embedding is
# noisy; more than ~10 s and we waste time on diminishing returns.
MIN_REF_SECONDS = 3.0
MAX_REF_SECONDS = 10.0

# Per-call caches (process lifetime).
_REF_SE_CACHE: dict[str, "torch.Tensor"] = {}
_SOURCE_SE: "torch.Tensor | None" = None


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def write_json_line(obj: dict) -> None:
    stdout_bin.write((json.dumps(obj, ensure_ascii=True) + "\n").encode("utf-8"))
    stdout_bin.flush()


def read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError(f"Short read: got {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def read_line_binary(stream) -> bytes | None:
    """Byte-wise newline-terminated read on a binary stream — see
    diarization_bridge.py for the TextIOWrapper interaction this avoids."""
    buf = bytearray()
    while True:
        ch = stream.read(1)
        if not ch:
            return None if not buf else bytes(buf)
        if ch == b"\n":
            return bytes(buf)
        buf.extend(ch)


def write_pcm_response(samples_np: np.ndarray, sample_rate: int) -> None:
    samples_np = np.clip(samples_np, -1.0, 1.0)
    int16 = (samples_np * 32767.0).astype(np.int16)
    pcm_bytes = int16.tobytes()
    header = {"sample_rate": int(sample_rate), "num_samples": int(int16.size)}
    stdout_bin.write((json.dumps(header) + "\n").encode("utf-8"))
    stdout_bin.write(pcm_bytes)
    stdout_bin.flush()


# ─── Path resolution ───────────────────────────────────────────────────────

def stub_wavmark() -> None:
    """Replace `wavmark` with a stub before OpenVoice imports it.

    Why: OpenVoice's `ToneColorConverter.__init__` calls
    `wavmark.load_model().to(device)` unconditionally on this version
    of the repo (the `enable_watermark` kwarg is not propagated past
    `OpenVoiceBaseClass.__init__`, which doesn't accept it). The real
    `wavmark.load_model()` downloads ~200 MB of weights on first call
    and instantiates a heavy diffusion model on CPU/GPU that we never
    use — we always invoke `convert(message="")` which skips the
    watermarking branch.

    The stub returns a tiny `nn.Identity`-like object: enough to call
    `.to(device)` on without raising, so the constructor finishes
    immediately. The module reference is never invoked elsewhere in
    our pipeline.
    """
    import types

    class _NoopWavmarkModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            raise RuntimeError(
                "wavmark stub was invoked — voice_convert_bridge does not "
                "expect watermarking. Pass message=\"\" to convert()."
            )

    fake_wavmark = types.ModuleType("wavmark")
    fake_wavmark.load_model = lambda *a, **k: _NoopWavmarkModel()
    sys.modules["wavmark"] = fake_wavmark


def patch_librosa_filters_mel() -> None:
    """Replace `librosa.filters.mel` with a torchaudio-backed equivalent
    BEFORE OpenVoice imports it.

    Why: on this Python 3.12 / Windows install, the very first
    `from librosa.filters import mel` triggers a Numba JIT chain that
    spins on a single CPU thread for >5 minutes without progress (we
    confirmed it consumes ~600 s of CPU producing no output). The
    `mel()` function only builds a deterministic mel filterbank matrix
    — it doesn't need Numba at all. `torchaudio.functional.melscale_fbanks`
    builds the same matrix in a few milliseconds with no JIT step.

    By planting a fake `librosa.filters` module in `sys.modules` before
    `openvoice.mel_processing` runs `from librosa.filters import mel`,
    Python's import machinery hands back our fake instead of triggering
    the real (broken) import. OpenVoice's `mel_processing.py` only
    calls `mel(...)` to build the filterbank — semantically identical
    output, no behavioural change.
    """
    import types

    def _mel_replacement(
        sr,
        n_fft,
        n_mels=128,
        fmin=0.0,
        fmax=None,
        htk=False,
        norm="slaney",
        dtype=np.float32,
    ):
        import torchaudio  # type: ignore

        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=float(fmin),
            f_max=float(fmax) if fmax is not None else float(sr) / 2.0,
            n_mels=int(n_mels),
            sample_rate=int(sr),
            norm="slaney" if norm == "slaney" else None,
            mel_scale="htk" if htk else "slaney",
        )
        # librosa returns [n_mels, n_freqs]; torchaudio returns
        # [n_freqs, n_mels]. Transpose to match librosa's contract.
        return fb.T.numpy().astype(dtype)

    # Replacement for `librosa.load` — same semantics OpenVoice uses
    # (target sample rate via `sr=...`, mono downmix), implemented on
    # top of `soundfile.read` + torchaudio resample. Avoids librosa's
    # lazy chain into `audioread`/`soxr` which deadlocks the same way
    # librosa.filters did on this Python 3.12/Windows combo.
    def _load_replacement(path, sr=None, mono=True, **_kwargs):
        import soundfile as sf  # type: ignore
        import torch  # type: ignore
        import torchaudio  # type: ignore

        audio, native_sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1 and mono:
            audio = audio.mean(axis=1).astype(np.float32, copy=False)
        if sr is not None and int(native_sr) != int(sr):
            t = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
            t = torchaudio.functional.resample(
                t, orig_freq=int(native_sr), new_freq=int(sr)
            )
            audio = t.squeeze(0).contiguous().numpy()
            native_sr = int(sr)
        return audio, int(native_sr)

    # Force the real `librosa` package shell to load first — its module
    # object is what `import librosa` resolves to inside OpenVoice.
    import librosa  # noqa: F401

    fake = types.ModuleType("librosa.filters")
    fake.mel = _mel_replacement
    sys.modules["librosa.filters"] = fake
    sys.modules["librosa"].filters = fake
    # Override `librosa.load` with the soundfile-based replacement so
    # `extract_se` and `convert` don't trigger the audioread/soxr
    # lazy import that hangs at 100 % CPU on this Windows install.
    sys.modules["librosa"].load = _load_replacement


def add_openvoice_repo_to_path() -> None:
    """Make `from openvoice.api import ToneColorConverter` importable.

    We don't pip-install OpenVoice — its pip package is unmaintained
    and pulls heavy optional deps (jieba, pypinyin, mecab) that we
    don't need for English/Portuguese tone-color conversion. Instead
    `Install-OpenVoice` clones the upstream repo into
    `third_party/OpenVoice/` and we add it to sys.path here.
    """
    bridge_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(bridge_dir)
    candidate = os.path.join(project_root, "third_party", "OpenVoice")
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)


def find_tcc_checkpoint() -> tuple[str, str]:
    """Locate the TCC checkpoint + config under `models/openvoice/`.

    Returns (checkpoint_path, config_path). Raises if either is
    missing.
    """
    bridge_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(bridge_dir)
    base = os.path.join(project_root, "models", "openvoice", "converter")

    ckpt = os.path.join(base, "checkpoint.pth")
    cfg = os.path.join(base, "config.json")
    if not os.path.isfile(ckpt) or not os.path.isfile(cfg):
        raise FileNotFoundError(
            f"OpenVoice TCC checkpoint not found at {base}. Run "
            f"`Install-OpenVoice` in scripts/install.ps1 to download it."
        )
    return ckpt, cfg


# ─── TCC bootstrap ─────────────────────────────────────────────────────────

class TccBridge:
    def __init__(self) -> None:
        t_total = time.monotonic()
        add_openvoice_repo_to_path()
        # Lazy imports: OpenVoice and torch together pull ~3 s on first
        # boot. Doing them here lets the bridge fail loud (with a
        # readable Python traceback) when something is missing instead
        # of stalling Rust's "wait for ready" forever.
        t = time.monotonic()
        import torch  # type: ignore
        log(f"[init] import torch: {(time.monotonic() - t) * 1000:.0f} ms")

        # Block the real librosa.filters before OpenVoice imports it
        # (Numba JIT loop), and stub out wavmark so the TCC constructor
        # doesn't try to load the 200 MB watermark model.
        patch_librosa_filters_mel()
        stub_wavmark()

        t = time.monotonic()
        from openvoice.api import ToneColorConverter  # type: ignore
        log(f"[init] import openvoice.api: {(time.monotonic() - t) * 1000:.0f} ms")

        self._torch = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"[init] TCC device: {device}")

        ckpt_path, cfg_path = find_tcc_checkpoint()
        log(f"[init] loading TCC checkpoint: {ckpt_path}")
        t = time.monotonic()
        # `enable_watermark=False` skips wavmark — we don't need
        # output watermarking and wavmark adds another ~200 MB model
        # whose CPU load alone takes 2-4 minutes on first boot. Without
        # this kwarg the constructor calls `wavmark.load_model()` even
        # though we never invoke it later, which is what blocks the
        # bridge's "ready" handshake on startup.
        # `enable_watermark` is intentionally NOT passed: this version
        # of OpenVoice doesn't propagate it to the base class. The
        # 200 MB wavmark download is short-circuited by stub_wavmark()
        # above, so the constructor finishes in milliseconds anyway.
        self._tcc = ToneColorConverter(cfg_path, device=device)
        log(f"[init] ToneColorConverter ctor: {(time.monotonic() - t) * 1000:.0f} ms")
        t = time.monotonic()
        self._tcc.load_ckpt(ckpt_path)
        log(f"[init] load_ckpt: {(time.monotonic() - t) * 1000:.0f} ms")
        log(f"[init] TCC ready (total: {(time.monotonic() - t_total) * 1000:.0f} ms)")

    def get_target_se(self, ref_wav_path: str) -> "torch.Tensor | None":
        """Compute or retrieve the cached speaker embedding for the
        reference WAV. Returns None when the file is missing or the
        SE extractor failed (caller falls back to no conversion).

        We call `ToneColorConverter.extract_se(path)` directly instead
        of going through `openvoice.se_extractor.get_se`. The latter
        segments the WAV with `whisper_timestamped` (an extra ~150 MB
        dep that is pip-broken on Python 3.12 Windows) and averages
        SE across short clips. Calling extract_se on the whole 15-30s
        reference yields a single embedding from the entire utterance
        — slightly less robust against bad clips, much simpler, no
        external dependency.
        """
        cached = _REF_SE_CACHE.get(ref_wav_path)
        if cached is not None:
            return cached
        if not os.path.isfile(ref_wav_path):
            log(f"[se] reference wav not found: {ref_wav_path}")
            return None
        try:
            t = time.monotonic()
            se = self._tcc.extract_se(ref_wav_path)
            log(
                f"[se] extracted target SE for {os.path.basename(ref_wav_path)} "
                f"in {(time.monotonic() - t) * 1000:.0f} ms"
            )
            _REF_SE_CACHE[ref_wav_path] = se
            return se
        except Exception as e:  # noqa: BLE001
            import traceback
            log(f"[se] extract_se failed for {ref_wav_path}: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            return None

    def get_source_se(self, kokoro_wav_path: str) -> "torch.Tensor":
        """Return the source-side SE used as the 'from' embedding for
        every conversion. We compute it once from the first Kokoro
        sample we see (Kokoro voices share enough spectral statistics
        that a single embedding suffices). Subsequent calls reuse.

        Same `extract_se` shortcut as `get_target_se` above — avoids
        the whisper_timestamped dependency entirely.
        """
        global _SOURCE_SE
        if _SOURCE_SE is not None:
            return _SOURCE_SE
        t = time.monotonic()
        se = self._tcc.extract_se(kokoro_wav_path)
        log(
            f"[se] extracted Kokoro source SE in "
            f"{(time.monotonic() - t) * 1000:.0f} ms (cached for the session)"
        )
        _SOURCE_SE = se
        return se

    def convert(
        self,
        source_pcm_f32: np.ndarray,
        source_sr: int,
        ref_wav_path: str,
    ) -> tuple[np.ndarray, int]:
        """Run the actual tone-color conversion. Returns (audio_f32, sr).

        On any internal failure (missing reference, TCC inference error)
        the source audio is returned unchanged at its original rate so
        the pipeline degrades gracefully to plain Kokoro output.
        """
        # OpenVoice's API operates on file paths. We write the source
        # audio to a temp WAV, call convert(), and ask for an in-memory
        # array back (output_path=None makes it return the array).
        src_path = os.path.join(
            tempfile.gettempdir(),
            f"vc_src_{os.getpid()}_{int(time.time() * 1000)}.wav",
        )
        try:
            _write_mono_wav(src_path, source_pcm_f32, source_sr)
            target_se = self.get_target_se(ref_wav_path)
            if target_se is None:
                log(
                    f"[vc] FALLBACK: target SE missing for "
                    f"{os.path.basename(ref_wav_path)} — playing raw TTS"
                )
                return source_pcm_f32, source_sr
            source_se = self.get_source_se(src_path)

            t = time.monotonic()
            out_audio = self._tcc.convert(
                audio_src_path=src_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=None,  # return array
                tau=0.3,
                message="",  # disable watermark string
            )
            log(
                f"[vc] converted {len(source_pcm_f32)} src samples → "
                f"{len(out_audio)} out samples in "
                f"{(time.monotonic() - t) * 1000:.0f} ms"
            )
            return np.asarray(out_audio, dtype=np.float32), TCC_SAMPLE_RATE
        except Exception as e:  # noqa: BLE001
            log(f"[vc] conversion failed: {type(e).__name__}: {e}")
            return source_pcm_f32, source_sr
        finally:
            try:
                os.remove(src_path)
            except OSError:
                pass


def _write_mono_wav(path: str, samples_f32: np.ndarray, sample_rate: int) -> None:
    int16 = (np.clip(samples_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(int16.tobytes())


# ─── Main loop ─────────────────────────────────────────────────────────────

def main() -> None:
    bridge = TccBridge()
    write_json_line({"status": "ready"})

    while True:
        header_bytes = read_line_binary(stdin_bin)
        if header_bytes is None:
            return
        line = header_bytes.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            action = request.get("action", "convert")

            # Preload requests have no PCM payload — they only ask the
            # bridge to extract and cache the speaker embedding for the
            # given reference WAV so the first real conversion fires
            # without paying the ~150 ms SE-extraction cost. The reply
            # is a single JSON line, no audio.
            if action == "preload":
                ref_path = request.get("reference_wav_path", "")
                t = time.monotonic()
                se = bridge.get_target_se(ref_path) if ref_path else None
                ok = se is not None
                write_json_line({
                    "status": "preloaded" if ok else "preload_failed",
                    "reference_wav_path": ref_path,
                    "elapsed_ms": int((time.monotonic() - t) * 1000),
                })
                continue

            source_sr = int(request["source_sr"])
            source_num_samples = int(request["source_num_samples"])
            reference_wav_path = request.get("reference_wav_path", "")
            _speaker_id = int(request.get("speaker_id", 0))

            pcm_bytes = read_exact(stdin_bin, source_num_samples * 2)
            source_np = (
                np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if not reference_wav_path:
                write_pcm_response(source_np, source_sr)
                continue

            converted, out_sr = bridge.convert(source_np, source_sr, reference_wav_path)
            write_pcm_response(converted, out_sr)

        except Exception as e:  # noqa: BLE001 — bridge must stay alive
            log(f"VC error: {type(e).__name__}: {e}")
            import traceback
            log(traceback.format_exc())
            # Send a single-sample silence frame so the Rust client
            # doesn't block on stdin.read_exact for the PCM payload.
            write_pcm_response(np.zeros(1, dtype=np.float32), TCC_SAMPLE_RATE)


if __name__ == "__main__":
    main()
