# config.py
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tomllib

# GENERAL CONFIGURATIONS : OS Name, Base directory and other directories

OS_NAME = os.name
if OS_NAME not in {"posix", "nt"}:
    raise OSError(f"Unsupported OS: {OS_NAME}")


def get_base_dir() -> Path:
    is_compiled = getattr(sys, "frozen", False) or "__compiled__" in globals()
    if is_compiled:
        exe_dir = Path(sys.executable).resolve().parent
        if exe_dir.name == "bin":
            return exe_dir.parent
        return exe_dir
    return (
        Path(__file__).resolve().parent.parent.parent
    )  # '/src/core/config.py'.parent.parent.parent is '/'


BASE_DIR: Path = get_base_dir()

DATA_DIR = BASE_DIR / "data"
PLUGINS_DIR = BASE_DIR / "plugins"
CONFIG_DIR = BASE_DIR / "config"

DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.toml"


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
    model_path: str = "models/vad/silero_vad.onnx"
    threshold: float = 0.5
    min_silence_duration_ms: int = 600
    speech_pad_ms: int = 60
    preroll_blocks: int = 6


@dataclass
class STTConfig:
    model_size: Literal["small", "small.en", "medium", "large-v3"] = "medium"
    start_state: str = "AWAKE"
    device: Literal["cpu", "cuda"] = "cpu"
    download_root: str = "models/whisper_models_cache"
    beam_size: int = 5
    cpu_threads: int = 6

    awake_timeout: float = 10.0
    min_command_ms: float = 600.0

    language: str = "en"
    initial_prompt: str = (
        "English language, speech to an assistant. Terms: Atlas, Python."
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
    silence_duration: float = 0.2


@dataclass
class LLMConfig:
    model_path: str = "models/llm/gemma2_2b."
    initial_prompt: str = "You are an AI Voice Assistant 'Atlas'. Answer briefly and concisely in English. "
    context_tokens: int = 2048
    max_msg_tokens: int = 512
    temperature: float = 0.7


@dataclass
class OPConfig:
    commands: str = "commands.json"

    def get_commands_path(self, base_dir: str | Path | None = None) -> Path:
        path = Path(self.commands)
        if path.is_absolute():
            return path

        base = Path(base_dir) if base_dir is not None else DATA_DIR
        return base / path

    def load_commands(self) -> dict[str, dict[str, str | list[str] | None]]:
        path = CONFIG_DIR / self.commands
        if not path.exists():
            raise FileNotFoundError(
                "[!] `commands.json` file was not found! Please check path or consider downloading latest version from "
                + "[github repository](https://github.com/Minkx1/Atlas)"
            )

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            return {}

        commands: dict[str, dict[str, str | list[str] | None]] = {}
        for intent, values in payload.items():
            if not isinstance(values, dict):
                continue

            commands[str(intent)] = {
                "sounds": values.get("sounds", []),
                "triggers": values.get("triggers", []),
            }

        return commands


@dataclass
class AppConfig:
    name: str = "Atlas"
    username: str = "Sir"
    log: bool = False

    audio: AudioConfig = field(default_factory=AudioConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    op: OPConfig = field(default_factory=OPConfig)
    kws: KWSConfig = field(default_factory=KWSConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)


def load_config(config_path: str = "config/config.toml") -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        if DEFAULT_CONFIG_PATH.exists():
            path.write_text(
                DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8"
            )
        else:
            raise FileNotFoundError(
                "[!] `config.toml` file was not found! Please consider downloading latest version from [github repository]"
                + "(https://github.com/Minkx1/Atlas)"
            )
        return load_config(str(path))

    with open(path, "rb") as f:
        data = tomllib.load(f)

    app: dict = data.get("app", {})

    res = AppConfig(
        name=app.get("name", "Atlas"),
        username=app.get("username", "Sir"),
        log=app.get("log", False),
        audio=AudioConfig(**data.get("audio", {})),
        llm=LLMConfig(**data.get("llm", {})),
        op=OPConfig(**data.get("op", {})),
        kws=KWSConfig(**data.get("kws", {})),
        tts=TTSConfig(**data.get("tts", {})),
        vad=VADConfig(**data.get("vad", {})),
        stt=STTConfig(**data.get("stt", {})),
    )

    return res


# Global config object
cfg = load_config()

__all__ = ["BASE_DIR", "CONFIG_DIR", "DATA_DIR", "OS_NAME", "PLUGINS_DIR", "cfg"]
