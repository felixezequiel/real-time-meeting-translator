"""
Source-separation bridge using SpeechBrain Sepformer.

When two speakers talk at the same time on the meeting audio (which
the loopback captures as a single mono mix), Whisper produces a
garbled salad of both voices. This bridge takes each mono chunk and
returns TWO separated mono chunks — channel A and channel B — so the
downstream pipeline can run two parallel STT/translation/TTS branches
and the listener hears each speaker translated separately.

Trade-offs:
  - Sepformer is trained on exactly 2 simultaneous speakers. With 3+
    overlapping speakers, expect quality degradation.
  - On a 6 GB laptop GPU, libri2mix Sepformer runs at ~RTF 0.05 — a
    1-second chunk separates in ~50 ms. Plenty of headroom for live.
  - When only one speaker is active, one channel comes back near-
    silent. The Rust side detects that via the per-channel RMS in the
    response header and skips the empty channel.

Protocol (binary-framed, mirrors the diarization bridge pattern):

- Startup:
    {"status": "ready"}\\n

- Request:
    {"num_samples": N, "sample_rate": 16000}\\n
    <N * 4 bytes f32 LE mono audio>

- Response:
    {"sample_rate": 16000, "num_samples": N,
     "rms_a": 0.052, "rms_b": 0.001}\\n
    <N * 4 bytes f32 LE for channel A>
    <N * 4 bytes f32 LE for channel B>

Requires:
  pip install speechbrain torch torchaudio
Model (~120 MB) is downloaded by `Install-SeparationModel` in
`scripts/install.ps1` to `~/.cache/speechbrain/sepformer-libri2mix/`.
"""

import json
import os
import sys
import time

import numpy as np


stdout_bin = sys.stdout.buffer
stdin_bin = sys.stdin.buffer

EXPECTED_SAMPLE_RATE = 16_000


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def write_json_line(obj: dict) -> None:
    stdout_bin.write((json.dumps(obj, ensure_ascii=True) + "\n").encode("utf-8"))
    stdout_bin.flush()


def write_response(channel_a: np.ndarray, channel_b: np.ndarray, sample_rate: int) -> None:
    rms_a = float(np.sqrt(np.mean(channel_a ** 2))) if channel_a.size else 0.0
    rms_b = float(np.sqrt(np.mean(channel_b ** 2))) if channel_b.size else 0.0
    header = {
        "sample_rate": int(sample_rate),
        "num_samples": int(channel_a.size),
        "rms_a": rms_a,
        "rms_b": rms_b,
    }
    stdout_bin.write((json.dumps(header) + "\n").encode("utf-8"))
    stdout_bin.write(channel_a.astype(np.float32).tobytes())
    stdout_bin.write(channel_b.astype(np.float32).tobytes())
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
    buf = bytearray()
    while True:
        ch = stream.read(1)
        if not ch:
            return None if not buf else bytes(buf)
        if ch == b"\n":
            return bytes(buf)
        buf.extend(ch)


class SeparationBridge:
    def __init__(self) -> None:
        import torch
        from speechbrain.inference.separation import SepformerSeparation as Sep

        self._torch = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_dir = os.environ.get(
            "SEPFORMER_PRETRAINED_DIR",
            os.path.join(
                os.path.expanduser("~"),
                ".cache", "speechbrain", "sepformer-libri2mix",
            ),
        )
        log(f"Loading Sepformer-libri2mix on {device} (cache={cache_dir})…")
        # libri2mix: trained at 16 kHz on 2-speaker LibriSpeech mixes.
        # Output is (batch, T, 2) — two channel-separated waveforms.
        self._sep = Sep.from_hparams(
            source="speechbrain/sepformer-libri2mix",
            savedir=cache_dir,
            run_opts={"device": device},
        )
        self._device = device
        log("Sepformer ready")

    def separate(self, audio_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Pad short inputs: Sepformer's internals choke on chunks
        # shorter than ~0.5 s. We pad with zeros and trim back after.
        min_samples = EXPECTED_SAMPLE_RATE // 2
        original_size = audio_f32.size
        if original_size < min_samples:
            audio_f32 = np.concatenate(
                [audio_f32, np.zeros(min_samples - original_size, dtype=np.float32)]
            )

        with self._torch.inference_mode():
            tensor = self._torch.from_numpy(audio_f32).unsqueeze(0).to(self._device)
            est_sources = self._sep.separate_batch(tensor)
            # est_sources: (1, T, 2)
            ch_a = est_sources[0, :, 0].cpu().numpy()
            ch_b = est_sources[0, :, 1].cpu().numpy()

        if original_size < min_samples:
            ch_a = ch_a[:original_size]
            ch_b = ch_b[:original_size]

        return ch_a, ch_b


def main() -> None:
    bridge = SeparationBridge()
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
            num_samples = int(request["num_samples"])
            _sample_rate = int(request.get("sample_rate", EXPECTED_SAMPLE_RATE))
            raw = read_exact(stdin_bin, num_samples * 4)
            audio = np.frombuffer(raw, dtype=np.float32)

            t = time.monotonic()
            ch_a, ch_b = bridge.separate(audio)
            dt = (time.monotonic() - t) * 1000

            log(f"[sep] {num_samples} samples → 2 ch in {dt:.0f} ms")
            write_response(ch_a, ch_b, EXPECTED_SAMPLE_RATE)
        except Exception as e:  # noqa: BLE001 — bridge must stay alive
            log(f"Separation error: {type(e).__name__}: {e}")
            import traceback
            log(traceback.format_exc())
            # Send back the original audio on both channels — graceful
            # degradation: pipeline runs one channel through STT, the
            # other channel adds noise but nothing crashes.
            zero = np.zeros(num_samples, dtype=np.float32)
            write_response(zero, zero, EXPECTED_SAMPLE_RATE)


if __name__ == "__main__":
    main()
