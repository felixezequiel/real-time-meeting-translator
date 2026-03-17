"""
TTS bridge for Meeting Translator using Piper TTS.
Runs as a persistent subprocess, communicating via JSON lines on stdin/stdout.

Protocol:
- Startup: prints {"status": "ready"} when models are loaded
- Request:  {"text": "...", "language": "pt-br"}
- Response: {"samples": [...], "sample_rate": 22050}

Requires: pip install piper-tts
"""

import json
import sys
import io
import wave
import struct

try:
    from piper import PiperVoice
    HAS_PIPER = True
except ImportError:
    HAS_PIPER = False

VOICE_MODELS = {
    "pt-br": "pt_BR-faber-medium",
    "en-us": "en_US-lessac-medium",
}

OUTPUT_SAMPLE_RATE = 22050


def load_voices():
    voices = {}
    if not HAS_PIPER:
        sys.stderr.write("WARNING: piper-tts not installed, using dummy TTS\n")
        sys.stderr.flush()
        return voices

    for lang, model_name in VOICE_MODELS.items():
        try:
            sys.stderr.write(f"Loading Piper voice: {model_name}...\n")
            sys.stderr.flush()
            voice = PiperVoice.load(model_name)
            voices[lang] = voice
        except Exception as e:
            sys.stderr.write(f"Failed to load voice {model_name}: {e}\n")
            sys.stderr.flush()
    return voices


def synthesize_piper(voices, text, language):
    if language not in voices:
        return generate_silence(len(text) * 100)

    voice = voices[language]
    audio_buffer = io.BytesIO()

    with wave.open(audio_buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(OUTPUT_SAMPLE_RATE)
        voice.synthesize(text, wav)

    audio_buffer.seek(0)
    with wave.open(audio_buffer, "rb") as wav:
        n_frames = wav.getnframes()
        raw_data = wav.readframes(n_frames)
        sample_rate = wav.getframerate()

    int16_samples = struct.unpack(f"<{n_frames}h", raw_data)
    float_samples = [s / 32768.0 for s in int16_samples]
    return float_samples, sample_rate


def generate_silence(num_samples):
    """Fallback when Piper is not available."""
    return [0.0] * max(num_samples, 1), OUTPUT_SAMPLE_RATE


def main():
    voices = load_voices()
    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            text = request["text"]
            language = request.get("language", "pt-br")

            if voices and language in voices:
                samples, sample_rate = synthesize_piper(voices, text, language)
            else:
                samples, sample_rate = generate_silence(len(text) * 100)

            response = {"samples": samples, "sample_rate": sample_rate}
            print(json.dumps(response), flush=True)
        except Exception as e:
            error_response = {"samples": [0.0] * 100, "sample_rate": OUTPUT_SAMPLE_RATE}
            sys.stderr.write(f"TTS error: {e}\n")
            sys.stderr.flush()
            print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    main()
