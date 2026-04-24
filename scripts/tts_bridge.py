"""
TTS bridge using Piper neural voices (fast, local, ~150ms).
Falls back to Windows SAPI if Piper is not available.

Protocol (binary-framed, no temp files):
- Startup (text line): {"status": "ready"}\n
- Request  (text line, stdin): {"text": "...", "language": "en-us"}\n
- Response (mixed, stdout):
    {"sample_rate": 22050, "num_samples": N}\n        <-- JSON header
    <N * 2 bytes int16 little-endian PCM>             <-- raw samples

The binary payload is written to sys.stdout.buffer after the JSON header,
skipping the WAV header + temp file round-trip.

Requires: pip install piper-tts pywin32
"""

import json
import sys
import os
import wave
import struct
import tempfile
import numpy as np

# stdin stays text-wrapped; stdout we keep as BINARY on purpose so we can
# write the JSON header as utf-8 bytes followed by raw PCM bytes. Writing
# bytes through TextIOWrapper is not safe.
stdout_bin = sys.stdout.buffer
sys.stdin = __import__('io').TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

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


def write_line(obj):
    """Write a JSON line to the binary stdout (so it can coexist with PCM)."""
    stdout_bin.write((json.dumps(obj, ensure_ascii=True) + "\n").encode("utf-8"))
    stdout_bin.flush()


def write_pcm_response(samples_np, sample_rate):
    """Send header + raw int16 PCM in one framed response."""
    samples_np = np.clip(samples_np, -1.0, 1.0)
    int16 = (samples_np * 32767).astype(np.int16)
    pcm_bytes = int16.tobytes()
    header = {"sample_rate": int(sample_rate), "num_samples": int(int16.size)}
    stdout_bin.write((json.dumps(header) + "\n").encode("utf-8"))
    stdout_bin.write(pcm_bytes)
    stdout_bin.flush()


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

    _voice_name, voice = entry
    sr = voice.config.sample_rate

    all_samples = []
    for chunk in voice.synthesize(text):
        all_samples.append(chunk.audio_float_array)

    if not all_samples:
        return np.zeros(100, dtype=np.float32), sr

    return np.concatenate(all_samples).astype(np.float32), sr


def synthesize_sapi(speaker, stream, voice_map, text, language):
    sapi_wav = os.path.join(tempfile.gettempdir(), f"sapi_tts_{os.getpid()}.wav")
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

    try:
        os.remove(sapi_wav)
    except OSError:
        pass

    if sample_width == 2:
        int16_samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        int16_samples = np.zeros(100, dtype=np.float32)

    if channels > 1:
        # downmix to mono
        int16_samples = int16_samples.reshape(-1, channels).mean(axis=1)

    return int16_samples, sample_rate


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

    write_line({"status": "ready"})

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
                if piper_voices:
                    samples, sample_rate = synthesize_piper(piper_voices, text, language)

                if samples is None and sapi_speaker:
                    samples, sample_rate = synthesize_sapi(
                        sapi_speaker, sapi_stream, sapi_voice_map, text, language
                    )

            if samples is None:
                samples, sample_rate = np.zeros(100, dtype=np.float32), 22050

            write_pcm_response(samples, sample_rate)
        except Exception as e:
            sys.stderr.write(f"TTS error: {e}\n")
            sys.stderr.flush()
            write_pcm_response(np.zeros(100, dtype=np.float32), 22050)


if __name__ == "__main__":
    main()
