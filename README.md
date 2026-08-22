# Atlas

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)](https://github.com/Minkx1/Atlas/releases)
[![Release](https://img.shields.io/github/v/release/Minkx1/Atlas?color=success)](https://github.com/Minkx1/Atlas/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Atlas** is a fast, AI-powered, fully offline voice assistant written in Python. It is designed to give you maximum controll of your PC and Atlas itself.

Fast, private, and highly customizable.

## Table of Contents

- [Main Features](#main-features)
- [Installation](#installation)
  - [Method 1: Pre-compiled Release (Recommended)](#method-1-pre-compiled-release-recommended)
  - [Method 2: Building from Source](#method-2-building-from-source)
- [First Launch & Models](#first-launch--models)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Main Features

- **Fully Offline:** Your voice data never leaves your computer.
- **Cross-Platform:** Should run smoothly on both Windows and Linux.
- **Great Performance:** Powered by performance optimized models and algorithms.
- **Highly Customizable:** Nice `config` and `user commands` give you ability to customize _Atlas_ however you wish.

## Installation

You can install _Atlas_ in two ways: downloading a ready-to-use binary (easiest) or running it from the source code.

### Method 1: Pre-compiled Release (Recommended)

Due to the size of the AI libraries (CUDA/PyTorch), the release archive is split into multiple parts to bypass GitHub limits.

**For Windows Users:**

1. Go to the [Releases page](https://github.com/Minkx1/Atlas/releases) and download **all** parts of the latest release (`atlas-xxx.tar.xz.001`, `.002`, etc.).
2. Place all downloaded parts into **one folder**.
3. Make sure you have [7-Zip](https://7-zip.org/) or [WinRAR](https://win-rar.com/) installed.
4. Right-click on the **first file** ONLY (`.tar.xz.001`).
5. Select `7-Zip -> Extract Here`. The archiver will automatically find and merge all the other parts!
6. Run `atlas.exe` (or the provided executable) inside the extracted folder.

**For Linux Users:**

1. Download all parts to a single directory.
2. Open your terminal in that directory and merge the files:

   ```bash
   cat atlas-*.tar.xz.* > atlas.tar.xz
   ```

3. Extract the merged archive:

   ```bash
   tar -xf atlas.tar.xz
   ```

4. Run the executable: `./atlas`

---

### Method 2: Building from Source

If you want to modify the code or run _Atlas_ in a Python environment, follow these steps.

#### Prerequisites

- **Python 3.10** or higher.
- **Git** installed on your system.
- _(Linux only)_ Audio libraries: `sudo apt-get install -y libportaudio2 portaudio19-dev libasound2-dev`

#### Setup

1. Clone the repository:

    ```bash
    git clone [https://github.com/Minkx1/Atlas.git](https://github.com/Minkx1/Atlas.git)
    cd atlas
    ```

2. Run the automated installation script to set up the environment and dependencies:

    **On Windows:**
    Double-click `install.bat` or run it in the Command Prompt / PowerShell:

    ```cmd
    install.bat
    ```

    **On Linux:**

    ```bash
    bash install.sh
    ```

3. Launch the assistant:

```bash
python main.py
```

---

## First Launch & Models

Whether you run _Atlas_ from a release or from source, it comes **without** the heavy neural network models pre-installed to save bandwidth.

On your **very first launch**, _Atlas_ will automatically download the necessary AI models (Whisper, TTS, LLM) into the `data/models/` directory.

> **Note**: This requires an internet connection and will download approximately ~4.5 GB of data. Please be patient! Subsequent launches will be instantaneous and completely offline.*

## Usage

Once _Atlas_ is running and the models are loaded, simply activate the assistant using your microphone.

Say `'Atlas!'` and it will greet you and be **awake** for next 15 seconds(this is customizable).
When _Atlas_ is awake, he listens to all you say and recognizes it as a command. But if you did not said anything before timer ran out, then you need to wake him again.
Also, you can change your assistant's name to another, but this also needs to change `data / keywords.txt` file. Please check the guide.

## Contributing

Big thanks to all testers and contributors.
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the [MIT License](/LICENSE).
