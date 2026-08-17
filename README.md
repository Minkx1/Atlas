# 🦎 Newt

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)]()
[![Release](https://img.shields.io/github/v/release/yourusername/newt?color=success)](https://github.com/yourusername/newt/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Newt** is a fast AI-powered, fully offline voice assistant written in Python. It is designed to give you a completely **NEW T**alk with your piece of scrap some would call a PC. 

Fast, private, and fully customizable.

---

## Table of Contents
- [Main Features](#main-features)
- [Installation](#installation)
  - [Method 1: Pre-compiled Release (Recommended)](#method-1-pre-compiled-release-recommended)
  - [Method 2: Building from Source](#method-2-building-from-source)
- [First Launch & Models](#first-launch--models)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Main Features
* **Fully Offline:** Your voice data never leaves your computer.
* **Cross-Platform:** Should run smoothly on both Windows and Linux.
* **Great Performance:** Powered by performance optimized models and algorithms.
* **Highly Customizable:** Nice `config` and `user commands` give you ability to customize Newt however you wish.

---

## Installation

You can install Newt in two ways: downloading a ready-to-use binary (easiest) or running it from the source code.

### Method 1: Pre-compiled Release (Recommended)

Due to the size of the AI libraries (CUDA/PyTorch), the release archive is split into multiple parts to bypass GitHub limits.

**For Windows Users:**
1. Go to the [Releases page](https://github.com/yourusername/newt/releases) and download **all** parts of the latest release (`newt-xxx.tar.xz.001`, `.002`, etc.).
2. Place all downloaded parts into **one folder**.
3. Make sure you have [7-Zip](https://7-zip.org/) or [WinRAR](https://win-rar.com/) installed.
4. Right-click on the **first file** ONLY (`.tar.xz.001`).
5. Select `7-Zip -> Extract Here`. The archiver will automatically find and merge all the other parts!
6. Run `newt.exe` (or the provided executable) inside the extracted folder.

**For Linux Users:**
1. Download all parts to a single directory.
2. Open your terminal in that directory and merge the files:
   ```bash
   cat newt-*.tar.xz.* > newt.tar.xz
   ```
3. Extract the merged archive:
   ```bash
   tar -xf newt.tar.xz
   ```


4. Run the executable: `./newt`

---

### Method 2: Building from Source

If you want to modify the code or run Newt in a Python environment, follow these steps.

#### Prerequisites

* **Python 3.10** or higher.
* **Git** installed on your system.
* *(Linux only)* Audio libraries: `sudo apt-get install -y libportaudio2 portaudio19-dev libasound2-dev`

#### Setup

1. Clone the repository:
```bash
git clone [https://github.com/yourusername/newt.git](https://github.com/yourusername/newt.git)
cd newt

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

Whether you run Newt from a release or from source, it comes **without** the heavy neural network models pre-installed to save bandwidth.

On your **very first launch**, Newt will automatically download the necessary AI models (Whisper, TTS, LLM) into the `data/models/` directory.

* *Note: This requires an internet connection and will download approximately ~4.5 GB of data. Please be patient! Subsequent launches will be instantaneous and completely offline.*

---

## Usage

Once Newt is running and the models are loaded, simply activate the assistant using your microphone.

Say `'Newt!'` and it will greet you and be **awake** for next 15 seconds(this is customizable).
When Newt is awake, he listens to all you say and recognizes it as a command. But if you did not said anything before timer ran out, then you need to wake him again.
Also, you can change your assistant's name to another, but this also needs to change `data / keywords.txt` file. Please check the guide.

---

## Contributing

Big thanks to all testers and contributors.
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the [MIT License](/LICENSE).

