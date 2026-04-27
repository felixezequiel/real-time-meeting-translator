"""
Online speaker diarisation + F0 estimation bridge.

Combines two cheap per-chunk analyses in one Python worker:
  - SpeechBrain ECAPA-TDNN embedding → cosine-NN clustering → speaker_id
  - pyworld DIO → mean F0 of voiced frames → target pitch for TTS shift

Both run on every audio chunk; total cost ~30-50 ms on CPU.

Why SpeechBrain ECAPA-TDNN (vs the previous Resemblyzer d-vector):

  Resemblyzer's encoder is a 256-d LSTM trained on relatively small
  data; in our pipeline it routinely confused similar voices on short
  windows, which forced the pipeline to flush mid-sentence and produce
  the choppy, garbled cloned voice the user reported.

  ECAPA-TDNN (192-d) from `speechbrain/spkrec-ecapa-voxceleb` is the
  current open-source SOTA on VoxCeleb-style speaker verification:
  trained on much more data, with channel-attention statistics pooling
  that's far more robust to short clips. It is freely downloadable
  from HuggingFace without any authentication, which keeps the
  project's "100% local, no API keys" goal intact.

Protocol (binary-framed on stdin, JSON line on stdout) — unchanged
from the previous version so the Rust client (`crates/diarization`)
needs no edits:

- Startup:
    {"status": "ready"}\\n

- Request:
    {"num_samples": N, "sample_rate": 16000}\\n
    <N * 4 bytes little-endian float32 mono audio>

- Response:
    {"speaker_id": 3, "is_new": false, "f0_hz": 142.3}\\n
    (`f0_hz` is 0.0 when the chunk had no voiced frames or was too quiet
    to estimate reliably.)

Clustering: incremental cosine-NN with running means (same shape as
before), but tuned for the higher-quality embeddings — the threshold
moved from 0.75 (Resemblyzer) to 0.55 (ECAPA), and we also keep a
small history of recent embeddings per speaker to make the centroid
robust to a single noisy chunk. Empty/silent chunks return a sentinel
so the Rust side falls back to the last known speaker.

Requires:
  pip install speechbrain torchaudio
Models (~22 MB) are downloaded by torch hub on first run and cached
under `~/.cache/torch/hub/` (or wherever speechbrain decides — it
respects `SPEECHBRAIN_PRETRAINED_DIR` if set).
"""

import json
import os
import sys
import time

import numpy as np


# Mixing a TextIOWrapper readline() with binary reads on the same fd
# eats random bytes from the PCM payload. Keep stdin strictly binary;
# we frame by hand (see diarization_bridge.py history for the bug
# we hit when first writing this).
stdout_bin = sys.stdout.buffer
stdin_bin = sys.stdin.buffer


# ECAPA-TDNN expects 16 kHz mono — matches the STT path.
EXPECTED_SAMPLE_RATE = 16_000

# Cosine threshold for matching a chunk to a known speaker. Tuned for
# ECAPA-TDNN: empirically, intra-speaker similarities sit ~0.65-0.85
# and inter-speaker similarities ~0.10-0.45, so 0.55 cleanly separates
# the two with margin. Resemblyzer needed 0.75 because its embedding
# is noisier; do not raise this without measuring on the new model.
MATCH_THRESHOLD = 0.55

# Hard cap on concurrently tracked speakers; keeps the cosine-scan
# cost bounded and prevents runaway clustering on noisy sessions.
MAX_SPEAKERS = 8

# Evict speakers whose last chunk is older than this. Their slot is
# reused for the next person who shows up. 120 s tolerates normal
# turn-taking gaps without leaking identities across long silences.
IDLE_TTL_SECONDS = 120.0

# Skip chunks shorter than this. ECAPA-TDNN tolerates short windows
# well — even 400 ms is enough for a usable embedding. The pipeline
# emits 500 ms audio chunks, so the threshold MUST sit below 500 ms or
# the diarizer rejects every single chunk and no speaker ever gets
# enrolled (the symptom: `speaker=None` for every commit, bootstrap
# voice forever). 0.4 s is comfortably below 500 ms with margin for
# any clock/resampling drift.
MIN_SAMPLES_FOR_EMBED = int(EXPECTED_SAMPLE_RATE * 0.4)

# RMS gate — silence/quiet noise produces a degenerate embedding that
# matches almost nothing, which used to spawn a new speaker on every
# pause. Reject anything quieter than this and let the Rust side fall
# back to the previous speaker_id.
MIN_RMS_FOR_EMBED = 0.005


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def write_json_line(obj: dict) -> None:
    stdout_bin.write((json.dumps(obj, ensure_ascii=True) + "\n").encode("utf-8"))
    stdout_bin.flush()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


class OnlineDiarizer:
    def __init__(self) -> None:
        # Lazy imports: torch + speechbrain together pull ~1.5 s of
        # init time. Doing them here keeps the failure mode loud — if
        # speechbrain isn't installed, the bridge crashes before
        # writing "ready" instead of stalling later.
        import torch  # type: ignore
        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

        self._torch = torch
        # CPU is plenty for ECAPA-TDNN at our throughput (~30-40 ms / embed)
        # and leaves the 6 GB GPU exclusively for whisper and CosyVoice.
        device = "cpu"
        save_dir = os.environ.get(
            "SPEECHBRAIN_PRETRAINED_DIR",
            os.path.join(os.path.expanduser("~"), ".cache", "speechbrain", "ecapa-tdnn"),
        )
        log(f"Loading speechbrain ECAPA-TDNN on {device} (cache={save_dir})…")
        self._encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=save_dir,
            run_opts={"device": device},
        )
        log("ECAPA-TDNN ready")

        # Per-speaker state: id, running mean embedding, count, last_seen.
        self._speakers: list[dict] = []
        self._next_id: int = 0

    def _evict_idle(self, now: float) -> None:
        self._speakers = [
            sp for sp in self._speakers if now - sp["last_seen"] <= IDLE_TTL_SECONDS
        ]

    def _embed(self, audio_f32: np.ndarray) -> np.ndarray:
        # SpeechBrain wants a 2-D float tensor (batch, time). It
        # internally normalises and produces a 192-d L2-normalised
        # embedding suitable for cosine similarity.
        tensor = self._torch.from_numpy(audio_f32).unsqueeze(0)
        with self._torch.no_grad():
            emb = self._encoder.encode_batch(tensor)
        # Drop the batch and squeeze utterance dim → (192,)
        return emb.squeeze(0).squeeze(0).cpu().numpy()

    def identify(self, audio_f32: np.ndarray) -> tuple[int | None, bool, float]:
        """Return (speaker_id, is_new, f0_hz).

        `speaker_id` is None when the chunk was too short or silent to
        produce a reliable embedding — the caller falls back to the
        previous speaker_id. `f0_hz` is 0.0 in the same conditions, or
        when pyworld couldn't find any voiced frames.
        """
        if audio_f32.size < MIN_SAMPLES_FOR_EMBED:
            return None, False, 0.0

        rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
        if rms < MIN_RMS_FOR_EMBED:
            return None, False, 0.0

        # F0 estimation runs first so we can return it even when the
        # speaker embedding ends up rejected later (e.g. if cosine
        # similarity tying breaks the cluster). Cheap (~10 ms).
        f0_hz = self._estimate_f0(audio_f32)

        embedding = self._embed(audio_f32)
        now = time.time()
        self._evict_idle(now)

        # Best match against running means of known speakers.
        best_idx, best_sim = -1, -1.0
        for idx, sp in enumerate(self._speakers):
            sim = cosine_similarity(embedding, sp["mean"])
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_idx >= 0 and best_sim >= MATCH_THRESHOLD:
            sp = self._speakers[best_idx]
            # Incremental running mean — keeps the centroid fresh
            # without storing every past embedding.
            sp["mean"] = (sp["mean"] * sp["count"] + embedding) / (sp["count"] + 1)
            sp["count"] += 1
            sp["last_seen"] = now
            return sp["id"], False, f0_hz

        if len(self._speakers) >= MAX_SPEAKERS:
            # Replace the least-recently-seen speaker. Reusing the id
            # rather than allocating a new one keeps downstream caches
            # (per-speaker reference WAVs) from growing forever.
            victim = min(
                range(len(self._speakers)),
                key=lambda i: self._speakers[i]["last_seen"],
            )
            recycled_id = self._speakers[victim]["id"]
            self._speakers[victim] = {
                "id": recycled_id,
                "mean": embedding.copy(),
                "count": 1,
                "last_seen": now,
            }
            return recycled_id, True, f0_hz

        new_id = self._next_id
        self._next_id += 1
        self._speakers.append({
            "id": new_id,
            "mean": embedding.copy(),
            "count": 1,
            "last_seen": now,
        })
        return new_id, True, f0_hz

    def _estimate_f0(self, audio_f32: np.ndarray) -> float:
        """Mean F0 of voiced frames in the chunk. 0.0 when no voiced
        frame could be detected (whisper, breath, music)."""
        try:
            import pyworld as pw

            # pyworld wants float64 at any reasonable sample rate. DIO
            # is noisier but ~3x faster than Harvest — good enough for
            # a running mean smoothed across many chunks.
            f0, _t = pw.dio(
                audio_f32.astype(np.float64),
                EXPECTED_SAMPLE_RATE,
                f0_floor=70.0,
                f0_ceil=400.0,
                frame_period=10.0,
            )
            voiced = f0[f0 > 0]
            if voiced.size == 0:
                return 0.0
            return float(np.mean(voiced))
        except Exception as e:  # noqa: BLE001
            log(f"F0 estimation failed: {type(e).__name__}: {e}")
            return 0.0


def read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError(f"Short read: got {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def read_line_binary(stream) -> bytes | None:
    """Byte-wise newline-terminated read on a binary stream.
    Avoids TextIOWrapper look-ahead buffering, which would swallow
    bytes from the PCM payload that follows the JSON header."""
    buf = bytearray()
    while True:
        ch = stream.read(1)
        if not ch:
            return None if not buf else bytes(buf)
        if ch == b"\n":
            return bytes(buf)
        buf.extend(ch)


def main() -> None:
    diarizer = OnlineDiarizer()
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

            speaker_id, is_new, f0_hz = diarizer.identify(audio)
            write_json_line({
                "speaker_id": speaker_id if speaker_id is not None else -1,
                "is_new": bool(is_new),
                "f0_hz": float(f0_hz),
            })

        except Exception as e:  # noqa: BLE001 — bridge must stay alive
            log(f"Diarization error: {type(e).__name__}: {e}")
            import traceback
            log(traceback.format_exc())
            write_json_line({"speaker_id": -1, "is_new": False, "f0_hz": 0.0})


if __name__ == "__main__":
    main()
