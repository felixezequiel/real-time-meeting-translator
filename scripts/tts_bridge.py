"""
TTS bridge using Piper neural voices (fast, local, ~150ms).
Falls back to Windows SAPI if Piper is not available.

Protocol:
- Startup: prints {"status": "ready"}
- Request:  {"text": "...", "language": "en-us"}
- Response: {"audio_file": "path/to/wav", "sample_rate": 22050}

Requires: pip install piper-tts pywin32
"""

import json
import sys
import io
import os
import wave
import struct
import tempfile
import numpy as np

# Force UTF-8 on Windows pipes
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# Each TTS process gets a unique temp file to avoid race conditions
# when two TTS instances run concurrently (speaker pipeline + mic pipeline).
TMP_WAV = os.path.join(tempfile.gettempdir(), f"tts_out_{os.getpid()}.wav")

PIPER_VOICE_MAP = {
    "en-us": "en_US-ryan-medium",
    "pt-br": "pt_BR-faber-medium",
}

PIPER_VOICE_URLS = {
    "en_US-ryan-medium": [
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
    ],
    "pt_BR-faber-medium": [
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx.json",
    ],
}

try:
    from piper import PiperVoice
    HAS_PIPER = True
except ImportError:
    HAS_PIPER = False

try:
    import pythoncom
    import win32com.client
    HAS_SAPI = True
except ImportError:
    HAS_SAPI = False


def download_piper_voice(voice_name, voices_dir):
    import urllib.request
    urls = PIPER_VOICE_URLS.get(voice_name, [])
    for url in urls:
        filename = os.path.basename(url)
        local_path = os.path.join(voices_dir, filename)
        if not os.path.exists(local_path):
            sys.stderr.write(f"Downloading {filename}...\n")
            sys.stderr.flush()
            urllib.request.urlretrieve(url, local_path)
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            sys.stderr.write(f"  Downloaded: {size_mb:.1f}MB\n")
            sys.stderr.flush()


def setup_piper(voices_dir):
    loaded_voices = {}
    for language, voice_name in PIPER_VOICE_MAP.items():
        if voice_name in [v[0] for v in loaded_voices.values()]:
            loaded_voices[language] = loaded_voices[
                [k for k, v in loaded_voices.items() if v[0] == voice_name][0]
            ]
            continue

        onnx_path = os.path.join(voices_dir, f"{voice_name}.onnx")
        json_path = os.path.join(voices_dir, f"{voice_name}.onnx.json")

        if not os.path.exists(onnx_path):
            download_piper_voice(voice_name, voices_dir)

        if os.path.exists(onnx_path):
            sys.stderr.write(f"Loading Piper voice: {voice_name}\n")
            sys.stderr.flush()
            voice = PiperVoice.load(onnx_path, config_path=json_path)
            loaded_voices[language] = (voice_name, voice)
            sys.stderr.write(f"  Loaded (sr={voice.config.sample_rate})\n")
            sys.stderr.flush()

    return loaded_voices


def setup_sapi():
    pythoncom.CoInitialize()
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    stream = win32com.client.Dispatch("SAPI.SpFileStream")

    voices = speaker.GetVoices()
    voice_map = {}
    for i in range(voices.Count):
        v = voices.Item(i)
        desc = v.GetDescription()
        sys.stderr.write(f"  SAPI Voice {i}: {desc}\n")
        if "PT-BR" in desc or "Portuguese" in desc:
            voice_map["pt-br"] = v
        if "EN-US" in desc or "English" in desc:
            voice_map["en-us"] = v

    sys.stderr.flush()
    return speaker, stream, voice_map


def synthesize_piper(piper_voices, text, language):
    entry = piper_voices.get(language)
    if entry is None:
        return None, None

    voice_name, voice = entry
    sr = voice.config.sample_rate

    all_samples = []
    for chunk in voice.synthesize(text):
        all_samples.append(chunk.audio_float_array)

    if not all_samples:
        return [0.0] * 100, sr

    audio = np.concatenate(all_samples)
    return audio.tolist(), sr


def synthesize_sapi(speaker, stream, voice_map, text, language):
    sapi_wav = os.path.join(tempfile.gettempdir(), "sapi_tts.wav")
    voice = voice_map.get(language, voice_map.get("en-us"))
    if voice:
        speaker.Voice = voice

    stream.Open(sapi_wav, 3)
    speaker.AudioOutputStream = stream
    speaker.Rate = 0
    speaker.Speak(text, 0)
    stream.Close()

    with wave.open(sapi_wav, 'rb') as w:
        n_frames = w.getnframes()
        sample_rate = w.getframerate()
        channels = w.getnchannels()
        sample_width = w.getsampwidth()
        raw_data = w.readframes(n_frames)

    os.remove(sapi_wav)

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
    voices_dir = os.path.join(tempfile.gettempdir(), "piper_voices")
    os.makedirs(voices_dir, exist_ok=True)

    piper_voices = {}
    sapi_speaker, sapi_stream, sapi_voice_map = None, None, {}

    if HAS_PIPER:
        sys.stderr.write("Setting up Piper TTS...\n")
        sys.stderr.flush()
        piper_voices = setup_piper(voices_dir)
        sys.stderr.write(f"Piper voices loaded: {list(piper_voices.keys())}\n")
        sys.stderr.flush()

    if HAS_SAPI:
        sys.stderr.write("Setting up SAPI fallback...\n")
        sys.stderr.flush()
        sapi_speaker, sapi_stream, sapi_voice_map = setup_sapi()

    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            text = request["text"]
            language = request.get("language", "en-us")

            samples, sample_rate = None, None

            if text.strip():
                # Try Piper first
                if piper_voices:
                    samples, sample_rate = synthesize_piper(piper_voices, text, language)

                # Fall back to SAPI
                if samples is None and sapi_speaker:
                    samples, sample_rate = synthesize_sapi(
                        sapi_speaker, sapi_stream, sapi_voice_map, text, language
                    )

            if samples is None:
                samples, sample_rate = [0.0] * 100, 22050

            # Write WAV with batch numpy conversion (was per-sample Python loop)
            samples_np = np.array(samples, dtype=np.float32)
            samples_np = np.clip(samples_np, -1.0, 1.0)
            int16_data = (samples_np * 32767).astype(np.int16)
            with wave.open(TMP_WAV, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(int16_data.tobytes())

            response = {"audio_file": TMP_WAV, "sample_rate": sample_rate}
            print(json.dumps(response, ensure_ascii=True), flush=True)
        except Exception as e:
            sys.stderr.write(f"TTS error: {e}\n")
            sys.stderr.flush()
            # Return empty audio on error
            with wave.open(TMP_WAV, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(22050)
                w.writeframes(struct.pack('<h', 0) * 100)
            print(json.dumps({"audio_file": TMP_WAV, "sample_rate": 22050}), flush=True)


if __name__ == "__main__":
    main()
