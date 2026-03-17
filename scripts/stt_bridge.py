"""
STT bridge for Meeting Translator using faster-whisper.
Runs as a persistent subprocess, communicating via JSON lines on stdin/stdout.

Protocol:
- Startup: prints {"status": "ready"} when model is loaded
- Request:  {"samples": [...], "sample_rate": 16000, "language": "en"}
- Response: {"text": "transcribed text here"}

Requires: pip install faster-whisper
"""

import json
import sys
import argparse
import numpy as np

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False


def load_model(model_path):
    if not HAS_WHISPER:
        sys.stderr.write("WARNING: faster-whisper not installed, using dummy STT\n")
        sys.stderr.flush()
        return None

    sys.stderr.write(f"Loading Whisper model: {model_path}...\n")
    sys.stderr.flush()
    model = WhisperModel(model_path, device="cpu", compute_type="int8")
    sys.stderr.write("Whisper model loaded.\n")
    sys.stderr.flush()
    return model


def transcribe(model, samples, sample_rate, language):
    if model is None:
        return "[whisper not available]"

    audio = np.array(samples, dtype=np.float32)

    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=1,
        best_of=1,
        vad_filter=False,
        without_timestamps=True,
    )

    text_parts = []
    for segment in segments:
        text_parts.append(segment.text)

    return " ".join(text_parts).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base.en", help="Whisper model name or path")
    args = parser.parse_args()

    model = load_model(args.model)
    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            samples = request["samples"]
            sample_rate = request.get("sample_rate", 16000)
            language = request.get("language", "en")

            text = transcribe(model, samples, sample_rate, language)
            # Clean any non-printable characters that break UTF-8
            text = text.encode('utf-8', errors='replace').decode('utf-8')
            response = {"text": text}
            print(json.dumps(response, ensure_ascii=False), flush=True)
        except Exception as e:
            error_response = {"text": f"[stt error: {e}]"}
            sys.stderr.write(f"STT error: {e}\n")
            sys.stderr.flush()
            print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    main()
