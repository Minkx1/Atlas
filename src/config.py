# config.py
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tomllib

# GENERAL CONFIGURATIONS

OS_NAME = os.name
if OS_NAME not in {"posix", "nt"}:
    raise OSError(f"Unsupported OS: {OS_NAME}")

DATA_DIR = Path("data/")


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    blocksize: int = 512
    dtype: str = "float32"


@dataclass
class KWSConfig:
    model_dir: str = "models/sherpa_onnx_kws"  # DATA_DIR / "keywords.txt"
    keywords_file: str = "keywords.txt"  # DATA_DIR / "keywords.txt"
    num_threads: int = 1
    score_threshold: float = 0.12


@dataclass
class VADConfig:
    threshold: float = 0.5
    min_silence_duration_ms: int = 600
    speech_pad_ms: int = 60
    preroll_blocks: int = 6


@dataclass
class WhisperConfig:
    model_size: Literal["small", "small.en", "medium", "large-v3"] = "medium"
    device: Literal["cpu", "cuda"] = "cpu"
    download_root: str = "models/whisper_models_cache"
    compute_type: str = "int8"
    beam_size: int = 5
    cpu_threads: int = 6
    language: str = "en"
    initial_prompt: str = (
        "English language, speech to an assistant. Terms: Newt, Python."
    )


@dataclass
class TTSConfig:
    model_path: str = "models/piper/en_US-lessac-medium.onnx"
    use_cuda: bool = False
    volume: float = 0.5
    length_scale: float = 2.0  # twice as slow
    noise_scale: float = 1.0  # more audio variation
    noise_w_scale: float = 1.0  # more speaking variation
    normalize_audio: bool = False  # use raw audio from voice


@dataclass
class AppConfig:
    awake_timeout: float = 10.0
    min_command_ms: float = 600.0
    profiler_debug: bool = False
    stt_pipeline_mode: Literal["KWS", "DIRECT"] = "KWS"

    audio: AudioConfig = field(default_factory=AudioConfig)
    kws: KWSConfig = field(default_factory=KWSConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
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
        profiler_debug=data.get("app", {}).get("profiler_debug", False),
        stt_pipeline_mode=data.get("app", {}).get("stt_pipeline_mode", "KWS"),
        audio=AudioConfig(**data.get("audio", {})),
        kws=KWSConfig(**data.get("kws", {})),
        tts=TTSConfig(**data.get("tts", {})),
        vad=VADConfig(**data.get("vad", {})),
        whisper=WhisperConfig(**data.get("whisper", {})),
    )


# Global object
cfg = load_config()
