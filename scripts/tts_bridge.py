"""
TTS bridge using Windows SAPI via win32com (fast, local, ~80ms).

Protocol:
- Startup: prints {"status": "ready"}
- Request:  {"text": "...", "language": "en-us"}
- Response: {"samples": [...], "sample_rate": 22050}

Requires: pip install pywin32
"""

import json
import sys
import os
import wave
import struct
import tempfile
import pythoncom
import win32com.client

TMP_WAV = os.path.join(tempfile.gettempdir(), "meeting_translator_tts.wav")


def setup_sapi():
    pythoncom.CoInitialize()
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    stream = win32com.client.Dispatch("SAPI.SpFileStream")

    voices = speaker.GetVoices()
    voice_map = {}
    for i in range(voices.Count):
        v = voices.Item(i)
        desc = v.GetDescription()
        sys.stderr.write(f"  Voice {i}: {desc}\n")
        if "PT-BR" in desc or "Portuguese" in desc:
            voice_map["pt-br"] = v
        if "EN-US" in desc or "English" in desc:
            voice_map["en-us"] = v

    sys.stderr.write(f"Voice map: {list(voice_map.keys())}\n")
    sys.stderr.flush()

    return speaker, stream, voice_map


def synthesize(speaker, stream, voice_map, text, language):
    voice = voice_map.get(language, voice_map.get("en-us"))
    if voice:
        speaker.Voice = voice

    stream.Open(TMP_WAV, 3)  # SSFMCreateForWrite
    speaker.AudioOutputStream = stream
    speaker.Rate = 3  # Faster speech
    speaker.Speak(text, 0)
    stream.Close()

    with wave.open(TMP_WAV, 'rb') as w:
        n_frames = w.getnframes()
        sample_rate = w.getframerate()
        channels = w.getnchannels()
        sample_width = w.getsampwidth()
        raw_data = w.readframes(n_frames)

    os.remove(TMP_WAV)

    total_samples = n_frames * channels
    if sample_width == 2:
        int16_samples = struct.unpack(f"<{total_samples}h", raw_data)
        float_samples = [s / 32768.0 for s in int16_samples]
    else:
        float_samples = [0.0] * 100

    if channels > 1:
        mono = []
        for i in range(0, len(float_samples), channels):
            mono.append(sum(float_samples[i:i + channels]) / channels)
        float_samples = mono

    return float_samples, sample_rate


def main():
    speaker, stream, voice_map = setup_sapi()
    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            text = request["text"]
            language = request.get("language", "en-us")

            if text.strip():
                samples, sample_rate = synthesize(speaker, stream, voice_map, text, language)
            else:
                samples, sample_rate = [0.0] * 100, 22050

            response = {"samples": samples, "sample_rate": sample_rate}
            print(json.dumps(response), flush=True)
        except Exception as e:
            sys.stderr.write(f"TTS error: {e}\n")
            sys.stderr.flush()
            print(json.dumps({"samples": [0.0] * 100, "sample_rate": 22050}), flush=True)


if __name__ == "__main__":
    main()
