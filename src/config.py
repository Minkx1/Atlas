# config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tomllib


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    blocksize: int = 512
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
    preroll_blocks: int = 12


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
        "English language, speech to an assistant. Terms: Newt, Python."
    )


@dataclass
class AppConfig:
    awake_timeout: float = 10.0
    min_command_ms: float = 600.0

    audio: AudioConfig = field(default_factory=AudioConfig)
    kws: KWSConfig = field(default_factory=KWSConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)


def load_config(config_path: str = "data/config.toml") -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        print(f"[WARN] Config {config_path} not found. Using defaults.")
        return AppConfig()

    with open(path, "rb") as f:
        data = tomllib.load(f)

    return AppConfig(
        awake_timeout=data.get("app", {}).get("awake_timeout", 10.0),
        min_command_ms=data.get("app", {}).get("min_command_ms", 600.0),
        audio=AudioConfig(**data.get("audio", {})),
        kws=KWSConfig(**data.get("kws", {})),
        vad=VADConfig(**data.get("vad", {})),
        whisper=WhisperConfig(**data.get("whisper", {})),
    )


# Global object
cfg = load_config()
