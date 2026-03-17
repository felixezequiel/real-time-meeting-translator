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

    if sample_width == 2:
        int16_samples = struct.unpack(f"<{n_frames}h", raw_data)
        return np.array([s / 32768.0 for s in int16_samples], dtype=np.float32), sample_rate
    elif sample_width == 4:
        int32_samples = struct.unpack(f"<{n_frames}i", raw_data)
        return np.array([s / 2147483648.0 for s in int32_samples], dtype=np.float32), sample_rate
    else:
        return np.zeros(n_frames, dtype=np.float32), sample_rate


LANGUAGE_PROMPTS = {
    "pt": "Transcreva o seguinte áudio em português brasileiro.",
    "en": "Transcribe the following audio in English.",
}


def transcribe(model, audio, language):
    if model is None:
        return "[whisper not available]"

    max_samples = 16000 * 10
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    initial_prompt = LANGUAGE_PROMPTS.get(language, "")

    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=3,
        best_of=2,
        vad_filter=False,
        without_timestamps=True,
        initial_prompt=initial_prompt,
        condition_on_previous_text=False,
        no_speech_threshold=0.5,
        log_prob_threshold=-0.8,
        temperature=0.0,
    )

    text_parts = []
    for segment in segments:
        if segment.no_speech_prob < 0.7:
            text_parts.append(segment.text)

    result = " ".join(text_parts).strip()
    return result.encode('utf-8', errors='replace').decode('utf-8')


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
                language = request.get("language", "pt")
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
                language = request.get("language", "pt")
            else:
                print(json.dumps({"text": ""}), flush=True)
                continue

            text = transcribe(model, audio, language)
            print(json.dumps({"text": text}, ensure_ascii=True), flush=True)
        except Exception as e:
            sys.stderr.write(f"STT error: {e}\n")
            sys.stderr.flush()
            print(json.dumps({"text": f"[error: {e}]"}), flush=True)


if __name__ == "__main__":
    main()
