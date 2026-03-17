# Real-Time Meeting Translator

Windows application that acts as a real-time audio proxy, translating meeting audio between Portuguese and English with maximum 2-5 second latency. Fully local — no internet, no APIs, no data leaves your machine.

## How it works

```
Speaker Audio → WASAPI Loopback → VAD → Whisper STT → Opus-MT Translation → Piper TTS → VB-Cable Output
```

The app captures audio from your speaker (what others say in the call), transcribes it, translates it, synthesizes speech in the target language, and outputs it through a virtual audio device.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Rust |
| Audio Capture | WASAPI (Windows native) |
| Virtual Audio | VB-Cable |
| Speech-to-Text | Whisper.cpp (local) |
| Voice Activity Detection | Silero VAD |
| Translation | CTranslate2 + Opus-MT |
| Text-to-Speech | Piper TTS |
| UI | Windows System Tray |

## Prerequisites

- Windows 10/11
- [VB-Cable](https://vb-audio.com/Cable/) installed
- ~1.5GB RAM available for models

## Status

🚧 In development — MVP phase.

## License

MIT
