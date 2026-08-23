# Atlas

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)](https://github.com/Minkx1/Atlas/releases)
[![Release](https://img.shields.io/github/v/release/Minkx1/Atlas?color=success)](https://github.com/Minkx1/Atlas/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Atlas** is a fast, AI-powered, fully offline voice assistant written in Python. It gives you full control over your PC and assistant behavior through a flexible plugin architecture.

**Private · Offline · Highly Customizable**

## Table of Contents

- [Main Features](#main-features)
- [Installation](#installation)
  - [Method I: Pre-compiled Release](#method-i-pre-compiled-release)
  - [Method II: Building from Source (Recommended)](#method-ii-building-from-source-recommended)
- [First Launch & Models](#first-launch--models)
- [Usage & Documentation](#usage--documentation)
- [Contributing](#contributing)
- [License](#license)

## Main Features

- **100% Offline & Private:** Your voice data never leaves your computer. Works even in the parking lot.
- **Cross-Platform:** Native support for Windows and Linux.
- **High Performance:** Optimized model inference for low-latency response times.
- **Highly Customizable:** Flexible config and plugin system allows you to customize _Atlas_ however you wish.

## System Requirements

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **CPU** | 4-Core x86_64 | 8-Core CPU / Dedicated GPU |
| **RAM / VRAM** | 4 GB RAM | 16 GB RAM / 6+ GB VRAM |
| **Disk Space** | ~5 GB (base models) | ~10 GB+ (custom LLMs) |

## Installation

### Method I: Pre-compiled Release

1. Go to the [Releases page](https://github.com/Minkx1/Atlas/releases) and download latest release.
2. Extract archive.
3. Run `Atlas.exe` (Windows) or `./Atlas` binary (Linux).
4. Download a compatible LLM `.gguf` model (e.g., `Llama-3.2-3B-Instruct`) when prompted.

---

### Method II: Building from Source (Recommended)

If you want to modify the code or run _Atlas_ in your own Python environment, follow these steps.

#### Prerequisites

- **Python 3.9+** and **Git**
- _(Linux only)_ Audio drivers & dependencies:

    ```bash
    sudo apt-get install -y libportaudio2 portaudio19-dev libasound2-dev
    ```

#### Setup

```bash
git clone https://github.com/Minkx1/Atlas.git
cd Atlas

# Automated environment setup
install.bat     # Windows
bash install.sh # Linux

# Run Atlas
python main.py

```

## First Launch & Models

On your **very first launch**, _Atlas_ automatically downloads the core neural network models (Whisper, TTS) into the `data/models/` directory.

> [!IMPORTANT]
> Atlas **DOES NOT** automatically download the LLM model.
> You need to download a compatible `.gguf` model (e.g., `Llama-3.2-3B-Instruct`) and place it into the `data/models/llm_models/` directory manually.

Subsequent runs start instantly and operate completely offline.

## Usage & Documentation

Once _Atlas_ is running and models are placed, activate the assistant by saying **`Atlas`**.

For detailed setup guides, plugin development, and architecture overview, visit the **[Atlas Documentation](https://www.google.com/search?q=https://minkx1.github.io/Atlas/)**.

## Contributing

Big thanks to all testers and contributors!
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).
