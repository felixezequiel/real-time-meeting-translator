# Real-Time Meeting Translator

Windows application that acts as a real-time audio proxy, translating meeting audio between Portuguese and English with maximum 2-5 second latency. Fully local — no internet, no APIs, no data leaves your machine.

## How it works

```
Speaker Audio -> WASAPI Capture -> VAD -> Whisper STT -> Opus-MT Translation -> Piper TTS -> VB-Cable Output
```

The app captures audio from your speaker (what others say in the call), transcribes it, translates it, synthesizes speech in the target language, and outputs it through a virtual audio device.

## Quick Start

```powershell
# Clone the repository
git clone https://github.com/felixezequiel/real-time-meeting-translator.git
cd real-time-meeting-translator

# Install everything (run as Administrator for VB-Cable)
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

The install script handles:
- Rust toolchain (via rustup)
- Python 3.12 (via winget)
- VB-Cable virtual audio driver
- Python packages (faster-whisper, transformers, piper-tts, torch)
- Compiles the project
- Runs tests
- Optionally launches the app

## Manual Setup

If you prefer to install dependencies manually:

1. Install [Rust](https://rustup.rs)
2. Install [Python 3.10+](https://python.org)
3. Install [VB-Cable](https://vb-audio.com/Cable/)
4. Run: `pip install -r scripts/requirements.txt`
5. Run: `cargo run --release`

First run downloads ML models (~1GB). Subsequent runs start faster.

## Uninstall

```powershell
powershell -ExecutionPolicy Bypass -File scripts\uninstall.ps1
```

Removes Python packages, downloaded models, build artifacts, and optionally VB-Cable and Rust.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Rust + Python bridges |
| Audio Capture | WASAPI via cpal |
| Virtual Audio | VB-Cable |
| Speech-to-Text | faster-whisper (local) |
| Voice Activity Detection | Energy-based RMS |
| Translation | Helsinki-NLP Opus-MT |
| Text-to-Speech | Piper TTS |
| UI | Windows System Tray |

## Architecture

```
Rust (performance + coordination)     Python (ML inference)
+---------------------------+         +----------------------+
| cpal -> AudioCapture      |  JSON   | faster-whisper (STT) |
| EnergyVAD                 |<------->| Opus-MT (Translation)|
| Pipeline Orchestrator     |  stdin/ | Piper TTS (Synthesis)|
| AudioPlayback -> VB-Cable |  stdout |                      |
| System Tray UI            |         |                      |
+---------------------------+         +----------------------+
```

## Configuration

Edit `config.toml` to change settings:

```toml
speaker_source_language = "English"
speaker_target_language = "Portuguese"
chunk_duration_ms = 2000
whisper_model = "base"
tts_speed = 1.1
```

## License

MIT
