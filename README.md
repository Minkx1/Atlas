# Atlas

<p align="center">
  <img src="docs/assets/atlas.svg" alt="Atlas Logo" width="200">
</p>

<p align="center">
  <em>A fast, AI-powered, fully offline voice assistant.</em>
</p>

---

[![Release](https://img.shields.io/github/v/release/Minkx1/Atlas?color=success)](https://github.com/Minkx1/Atlas/releases)
[![License](https://img.shields.io/github/license/Minkx1/Atlas)](https://github.com/Minkx1/Atlas/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)](https://github.com/Minkx1/Atlas/releases)

**Atlas** is a fast, private and 100% offline personal AI Voice Assistant.

## About

Atlas is a Jarvis-inspired local voice assistant built for fast, offline desktop work.
Powered by a modular IPC plugin architecture and local AI, it handles common desktop tasks while keeping your data strictly private and entirely on your machine.

## Main Features

- **Fully Offline & Private:** Your voice data never leaves your computer. Works even in the parking lot.
- **Cross-Platform:** Native support for Windows and Linux.
- **High Performance:** Optimized model inference for low-latency response times.
- **Highly Customizable:** Flexible config and plugin system allows you to customize *Atlas* however you wish.

## Quick Start

The fastest way to try Atlas is downloading a pre-compiled release:

1. Download the latest release from the [Releases page](https://github.com/Minkx1/Atlas/releases).
2. Extract the archive and run the executable (`Atlas.exe` or `./Atlas`).

For full instructions, source building, and system requirements, please refer to the documentation.

## Under the Hood

Atlas is built around a carefully selected stack of fast, lightweight, and purpose-built technologies. Each component handles a specific stage of the local AI pipeline:

- **Speech Recognition** — [Sherpa-ONNX KWS](https://github.com/k2-fsa/sherpa-onnx) handles wake-word detection, [Silero VAD](https://github.com/snakers4/silero-vad) detects speech activity, and [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) converts speech into text. Together, they form a fast and fully local speech-to-text pipeline.

- **Text-to-Speech** — [Piper TTS](https://github.com/OHF-Voice/piper1-gpl) generates natural-sounding speech locally, while [sounddevice](https://github.com/spatialaudio/python-sounddevice/) and [soundfile](https://github.com/bastibe/python-soundfile) handle low-level audio playback and processing.

- **Local AI & Conversations** — [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) provides efficient local LLM inference, while [Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) powers semantic embeddings and related NLP functionality.

All of these components run locally, allowing Atlas to provide voice interaction without relying on cloud-based AI services.

## Documentation

All guides, plugin development tutorials, and architecture overviews have been moved to the dedicated documentation site:

**[Read the Atlas Documentation](https://minkx1.github.io/Atlas/)**

- [System Requirements & Installation](https://minkx1.github.io/Atlas/installation/)
- [Models Setup Guide](https://minkx1.github.io/Atlas/installation/#first-launch-models)
- [Plugin API Overview](https://minkx1.github.io/Atlas/plugins/)

## License

This project is licensed under the [MIT License](LICENSE)
