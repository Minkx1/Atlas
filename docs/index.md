# Welcome to Atlas Documentation

**Atlas** is a fast, AI-powered, fully offline voice assistant. It was built with one primary goal: to provide a smart, low-latency assistant that respects your privacy and integrates deeply with your operating system.

## How it Works

Atlas operates entirely on your local machine using an optimized pipeline of AI models:

1. **Wake Word Detection:** Sherpna-ONNX KeywordSpotter model checks audio input for phonems, described in dedicated file, often `keywords.txt`.
2. **VAD + Speech-to-Text(STT):** Silero VAD constantly listens for your voice with near-zero CPU usage, and adds each speech chunk to the audio buffer for Faster-Whisper, that transcribes your voice into text accurately and really fast, minimizing latency.
3. **Command Processing:**
    * *Vector Embeddings*: Transcribed text is cleaned and embedded into a vector with sentence transformer model to be contextually compared with command triggers.
    * *Execution*: If command intent was successfuly recognized, Atlas executes it. Mostly it just says some phrases or, in some scenarios, it can go to sleep/quit/execute plugin.
    * *LLM Processing*: If no intent was recongized, command is passed to a local **Large Language Model** (like Llama 3.2 or Qwen) processes your command, accesses local plugins, and generates a response.
4. **Text-to-Speech (TTS):** Piper TTS reads the response back to you naturally or plays pre-recorded sounds.

---

## Where to go next?

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Installation & Setup](installation.md)**

    System requirements, downloading releases, and building from source.

-   :material-brain: **[Models Guide](installation.md#first-launch-models)**

    Learn how and which models to download and configure.

-   :material-puzzle: **[Plugin Development](plugins.md)**

    Write Python scripts to give Atlas new capabilities and system integrations.

-   :material-sitemap: **[Architecture](architecture.md)**

    Deep dive into the core engine and how modules communicate.

</div>