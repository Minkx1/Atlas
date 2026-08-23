# Installation

This guide will walk you through the system requirements and the installation process for Atlas on different OS.

## System Requirements

Atlas runs completely offline, which means your hardware does all the heavy lifting.

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **CPU** | 4-Core x86_64 | 8-Core CPU / Dedicated GPU |
| **RAM / VRAM** | 4 GB RAM | 8 GB RAM / 6+ GB VRAM |
| **Disk Space** | ~5 GB (base models) | ~10 GB+ (custom LLMs) |

---

## Installation Methods

=== "Method I: Pre-compiled Release"

    This is the easiest way to get started. You don't need Python installed.

    1. Go to the [Releases page](https://github.com/Minkx1/Atlas/releases) and download the latest release for your OS.
    2. Extract the archive into a dedicated folder.
    3. Run `Atlas.exe` (Windows) or the `./Atlas` binary (Linux).
    4. Proceed to the [First Launch & Models](#first-launch-models) section below.

=== "Method II: Building from Source"

    Recommended if you want to modify the code, develop plugins, or run *Atlas* in your own Python environment.

    **Prerequisites:**
    
    * **Python 3.9+** and **Git**
    * *(Linux only)* Audio drivers & dependencies:
        ```bash
        sudo apt-get install -y libportaudio2 portaudio19-dev libasound2-dev
        ```

    **Setup:**

    ```bash
    git clone https://github.com/Minkx1/Atlas.git
    cd Atlas
    ```

    Run the automated environment setup script:

    * **Windows:** run `install.bat`
    * **Linux:** run `bash install.sh`

    Launch the assistant:
    ```bash
    python main.py
    ```

---

## First Launch & Models

Whether you run *Atlas* from a release or from source, it comes **without** the heavy neural network models pre-installed to save bandwidth.

On your **very first launch**, *Atlas* will automatically download the necessary core neural-netowrk modes into the `data/models/` directory.

!!! note "Core Models Download"
    This requires an internet connection and will download approximately ~1 GB of data. Subsequent launches will be almost instantaneous and completely offline.

### Setting up the LLM

!!! warning "Manual LLM Installation Required"
    Atlas **DOES NOT** automatically download the Large Language Model (LLM). You must provide one yourself.

To make Atlas *smart*, you need to download a compatible `.gguf` model:

1. Download a model (e.g., `Llama-3.2-3B-Instruct` or similar) in `.gguf` format from HuggingFace.
2. Place the downloaded `.gguf` file manually into the `data/models/llm_models/` directory.
3. Start Atlas and say **"Atlas"** to activate it!
