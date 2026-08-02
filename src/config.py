# config.py
from dataclasses import dataclass, field
from typing import Literal

#                 #
#  ===  STT  ===  #
#                 #


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    blocksize: int = 512  # 512 семплів = 32 мс аудіо
    dtype: str = "float32"


@dataclass
class KWSConfig:
    model_dir: str = "./data/models/sherpa_onnx_kws"
    keywords_file: str = "keywords.txt"
    num_threads: int = 2
    score_threshold: float = 0.12


@dataclass
class VADConfig:
    threshold: float = 0.5
    min_silence_duration_ms: int = 600
    speech_pad_ms: int = 60
    preroll_blocks: int = 12  # 12 * 32 =~400 ms preroll before speaking


@dataclass
class WhisperConfig:
    model_size: Literal["small", "small.en", "medium", "large-v3"] = "medium"
    device: Literal["cpu", "cuda"] = "cpu"
    download_root: str = "./data/models/whisper_models_cache"
    compute_type: str = "int8"
    beam_size: int = 5
    cpu_threads: int = 6

    language: str = "en"
    initial_prompt: str = (
        "English language, speech to an assistant. "
        "Terms: Newt, Python, Linux, C++, code, programming, Arch, Cachy, terminal."
    )
    # language: str = "uk"
    # initial_prompt: str = (
    #     "Українська мова, розмова з Асистентом. "
    #     "Терміни: Newt, Python, Linux, C++, код, прога, програмування, Arch, Cachy."
    # )


@dataclass
class AppConfig:
    awake_timeout: float = 10.0  # How many seconds does the assistant wait for a command after the wake word
    min_command_ms: float = 600.0  # Minimal phrase size (random sounds protection)

    audio: AudioConfig = field(default_factory=AudioConfig)
    kws: KWSConfig = field(default_factory=KWSConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)


cfg = AppConfig()
