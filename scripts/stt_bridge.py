"""
STT bridge using faster-whisper with temp file audio transfer.

Protocol:
- Startup: prints {"status": "ready"}
- Request:  {"audio_file": "path/to/temp.wav", "language": "pt"}
- Response: {"text": "transcribed text"}

Requires: pip install faster-whisper numpy
"""

import json
import sys
import io
import argparse
import os
import wave
import struct
import tempfile
import numpy as np

# Force UTF-8 on Windows pipes
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False


def add_cuda_dll_dirs():
    """Add NVIDIA pip package DLL directories to search path."""
    site_packages = os.path.join(
        os.path.expanduser("~"), "AppData", "Roaming",
        "Python", "Python312", "site-packages", "nvidia"
    )
    for subdir in ["cublas", "cudnn"]:
        bin_dir = os.path.join(site_packages, subdir, "bin")
        if os.path.isdir(bin_dir):
            os.add_dll_directory(bin_dir)
            os.environ["PATH"] = bin_dir + ";" + os.environ.get("PATH", "")


def load_model(model_name):
    if not HAS_WHISPER:
        sys.stderr.write("WARNING: faster-whisper not installed\n")
        sys.stderr.flush()
        return None

    # Try GPU first, fall back to CPU
    add_cuda_dll_dirs()
    try:
        sys.stderr.write(f"Loading Whisper model: {model_name} on GPU...\n")
        sys.stderr.flush()
        model = WhisperModel(model_name, device="cuda", compute_type="float16")
        sys.stderr.write("Whisper model loaded on GPU.\n")
        sys.stderr.flush()
        return model
    except Exception as e:
        sys.stderr.write(f"GPU not available ({e}), falling back to CPU...\n")
        sys.stderr.flush()

    sys.stderr.write(f"Loading Whisper model: {model_name} on CPU...\n")
    sys.stderr.flush()
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    sys.stderr.write("Whisper model loaded on CPU.\n")
    sys.stderr.flush()
    return model


def read_wav_samples(wav_path):
    with wave.open(wav_path, 'rb') as w:
        n_frames = w.getnframes()
        sample_rate = w.getframerate()
        sample_width = w.getsampwidth()
        raw_data = w.readframes(n_frames)

    # Batch conversion with numpy — no Python per-sample loop
    if sample_width == 2:
        return np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0, sample_rate
    elif sample_width == 4:
        return np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0, sample_rate
    else:
        return np.zeros(n_frames, dtype=np.float32), sample_rate


def is_repetitive(text):
    """Detect hallucinated repetitive patterns like 'i'r un i'r un i'r un...'"""
    words = text.split()
    if len(words) < 6:
        return False

    # Check if any short phrase (1-3 words) repeats more than 4 times
    for pattern_len in range(1, 4):
        pattern = " ".join(words[:pattern_len])
        if len(pattern) < 2:
            continue
        count = text.count(pattern)
        if count > 4:
            return True

    # Check if unique words are less than 20% of total (very repetitive)
    unique = set(w.lower().strip(".,!?'\"") for w in words)
    if len(words) > 10 and len(unique) / len(words) < 0.2:
        return True

    return False


def transcribe(model, audio):
    """
    Transcribes audio and auto-detects the spoken language.

    Returns (text, detected_language_code).
    Auto-detection is essential to prevent the feedback loop: when our TTS output
    (e.g. Portuguese) leaks back into the loopback capture, Whisper detects it as
    Portuguese so the pipeline's language guard can drop it instead of re-translating.

    Note: we intentionally do NOT pass `initial_prompt` — it causes Whisper to
    hallucinate the prompt text ("Transcribe the following audio in English.") when
    the audio is silent or in the wrong language.
    """
    if model is None:
        return "[whisper not available]", "unknown"

    max_samples = 16000 * 10
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    segments, info = model.transcribe(
        audio,
        beam_size=1,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=250,
            speech_pad_ms=200,
        ),
        without_timestamps=True,
        condition_on_previous_text=False,
        no_speech_threshold=0.7,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.0,   # Skip segments with repetitive output
        repetition_penalty=1.3,            # Penalize repeated tokens to prevent loops
        temperature=0.0,
    )

    text_parts = []
    for segment in segments:
        text_parts.append(segment.text)

    result = " ".join(text_parts).strip()

    # Detect and discard hallucinated repetitive patterns
    if result and is_repetitive(result):
        sys.stderr.write(f"Hallucination detected, discarding: {result[:80]}...\n")
        sys.stderr.flush()
        result = ""
    detected_language = info.language if info else "unknown"
    return result.encode('utf-8', errors='replace').decode('utf-8'), detected_language


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base", help="Whisper model name or path")
    args = parser.parse_args()

    model = load_model(args.model)
    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)

            if "audio_file" in request:
                audio_path = request["audio_file"]
                audio, sample_rate = read_wav_samples(audio_path)
                # Clean up temp file
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
            elif "samples" in request:
                # Legacy JSON array mode
                samples = request["samples"]
                audio = np.array(samples, dtype=np.float32)
            else:
                print(json.dumps({"text": "", "language": "unknown"}), flush=True)
                continue

            text, detected_language = transcribe(model, audio)
            print(json.dumps({"text": text, "language": detected_language}, ensure_ascii=True), flush=True)
        except Exception as e:
            sys.stderr.write(f"STT error: {e}\n")
            sys.stderr.flush()
            print(json.dumps({"text": f"[error: {e}]"}), flush=True)


if __name__ == "__main__":
    main()
