"""
Diagnostic script to test STT bridge in isolation.

Usage:
  python scripts/test_stt.py                      # Record from mic and test
  python scripts/test_stt.py --wav path/to/file.wav  # Test with existing WAV
  python scripts/test_stt.py --model small         # Test with different model

Steps:
  1. Records 5 seconds of audio from default mic (or uses provided WAV)
  2. Runs faster-whisper transcription with current and improved settings
  3. Shows results side by side for comparison
"""

import argparse
import json
import sys
import os
import wave
import struct
import time
import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False


SAMPLE_RATE = 16000
RECORD_SECONDS = 5


def record_audio(duration=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
    if not HAS_SOUNDDEVICE:
        print("ERROR: sounddevice not installed. Run: pip install sounddevice")
        sys.exit(1)

    print(f"\nGravando {duration}s de audio... Fale agora!")
    print("=" * 40)
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Gravacao concluida.\n")
    return audio.flatten()


def read_wav(wav_path):
    with wave.open(wav_path, 'rb') as w:
        n_frames = w.getnframes()
        sample_rate = w.getframerate()
        sample_width = w.getsampwidth()
        n_channels = w.getnchannels()
        raw_data = w.readframes(n_frames)

    total_samples = n_frames * n_channels

    if sample_width == 2:
        samples = struct.unpack(f"<{total_samples}h", raw_data)
        audio = np.array([s / 32768.0 for s in samples], dtype=np.float32)
    elif sample_width == 4:
        samples = struct.unpack(f"<{total_samples}i", raw_data)
        audio = np.array([s / 2147483648.0 for s in samples], dtype=np.float32)
    else:
        audio = np.zeros(total_samples, dtype=np.float32)

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    print(f"WAV: {wav_path}")
    print(f"  Sample rate: {sample_rate} Hz, Channels: {n_channels}, Duration: {n_frames/sample_rate:.1f}s")
    return audio, sample_rate


def analyze_audio(audio, sample_rate):
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    duration = len(audio) / sample_rate
    non_silent = np.sum(np.abs(audio) > 0.01) / len(audio) * 100

    print(f"\n--- Audio Analysis ---")
    print(f"  Duration:    {duration:.2f}s")
    print(f"  RMS:         {rms:.6f}")
    print(f"  Peak:        {peak:.6f}")
    print(f"  Non-silent:  {non_silent:.1f}%")

    if rms < 0.001:
        print(f"  WARNING: Audio muito fraco! RMS={rms:.6f}")
        print(f"  Possivel problema: microfone com sinal muito baixo")
    elif rms < 0.01:
        print(f"  AVISO: Audio fraco. Pode afetar precisao do Whisper.")
    else:
        print(f"  OK: Nivel de audio adequado.")

    return rms


def save_temp_wav(audio, sample_rate, path):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for sample in audio:
            clamped = max(-1.0, min(1.0, sample))
            int16 = int(clamped * 32767)
            w.writeframes(struct.pack('<h', int16))


def test_transcription(model, audio, language, label, settings):
    print(f"\n--- {label} ---")
    print(f"  Settings: {settings}")

    start = time.time()
    segments, info = model.transcribe(audio, **settings)

    text_parts = []
    for segment in segments:
        prob_str = f"(no_speech={segment.no_speech_prob:.2f})"
        text_parts.append(f"  [{segment.start:.1f}s-{segment.end:.1f}s] {prob_str} {segment.text}")

    elapsed = time.time() - start

    if text_parts:
        print(f"  Segments:")
        for part in text_parts:
            print(f"    {part}")
    else:
        print(f"  (nenhum segmento detectado)")

    full_text = " ".join([s.text for s in model.transcribe(audio, **settings)[0]]).strip()
    print(f"  Result: \"{full_text}\"")
    print(f"  Time: {elapsed*1000:.0f}ms")

    return full_text


def main():
    parser = argparse.ArgumentParser(description="Test STT bridge in isolation")
    parser.add_argument("--wav", help="Path to WAV file (skip recording)")
    parser.add_argument("--model", default="base", help="Whisper model (base, small, medium)")
    parser.add_argument("--language", default="pt", help="Language code (pt, en)")
    parser.add_argument("--duration", type=int, default=5, help="Record duration in seconds")
    parser.add_argument("--gain", type=float, default=1.0, help="Audio gain multiplier")
    args = parser.parse_args()

    if not HAS_WHISPER:
        print("ERROR: faster-whisper not installed. Run: pip install faster-whisper")
        sys.exit(1)

    if args.wav:
        audio, sample_rate = read_wav(args.wav)
    else:
        audio = record_audio(args.duration)
        sample_rate = SAMPLE_RATE

    if args.gain != 1.0:
        print(f"\nApplying gain: {args.gain}x")
        audio = np.clip(audio * args.gain, -1.0, 1.0)

    analyze_audio(audio, sample_rate)

    temp_wav = os.path.join(os.environ.get('TEMP', '/tmp'), 'stt_test_diag.wav')
    save_temp_wav(audio, sample_rate, temp_wav)
    print(f"\nTemp WAV saved: {temp_wav}")

    print(f"\nLoading model: {args.model}...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    print("Model loaded.\n")

    lang = args.language

    old_settings = {
        "language": lang,
        "beam_size": 1,
        "best_of": 1,
        "vad_filter": True,
        "without_timestamps": True,
    }

    new_settings = {
        "language": lang,
        "beam_size": 5,
        "best_of": 3,
        "vad_filter": True,
        "vad_parameters": dict(
            min_silence_duration_ms=300,
            speech_pad_ms=200,
        ),
        "without_timestamps": True,
        "initial_prompt": "Transcreva o seguinte áudio em português brasileiro." if lang == "pt" else "Transcribe the following audio in English.",
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.5,
        "log_prob_threshold": -0.8,
        "temperature": 0.0,
    }

    print("=" * 60)
    print("COMPARACAO: Config antiga vs Config nova")
    print("=" * 60)

    old_result = test_transcription(model, audio, lang, "CONFIG ANTIGA (beam=1, best_of=1)", old_settings)
    new_result = test_transcription(model, audio, lang, "CONFIG NOVA (beam=5, best_of=3, prompt)", new_settings)

    if args.model == "base":
        print(f"\n\nDica: Teste com --model small para comparar precisao.")
        print(f"  python scripts/test_stt.py --model small")

    print(f"\n{'=' * 60}")
    print(f"RESUMO:")
    print(f"  Antiga: \"{old_result}\"")
    print(f"  Nova:   \"{new_result}\"")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
