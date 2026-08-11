# config.py
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tomllib

# GENERAL CONFIGURATIONS

OS_NAME = os.name
if OS_NAME not in {"posix", "nt"}:
    raise OSError(f"Unsupported OS: {OS_NAME}")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CONFIG_PATH = DATA_DIR / "config.toml"


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
class STTConfig:
    model_size: Literal["small", "small.en", "medium", "large-v3"] = "medium"
    device: Literal["cpu", "cuda"] = "cpu"
    download_root: str = "models/whisper_models_cache"
    compute_type: str = "int8"
    beam_size: int = 5
    cpu_threads: int = 6

    awake_timeout: float = 10.0
    min_command_ms: float = 600.0
    pipeline_mode: Literal["KWS", "DIRECT"] = "KWS"

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
class LLMConfig:
    model_path: str = "models/llm/gemma2_2b."
    initial_prompt: str = "You are an AI Voice Assistant 'Newt'. Answer briefly and concisely in English. "
    context_tokens: int = 2048
    max_msg_tokens: int = 512
    temperature: float = 0.7


@dataclass
class OPConfig:
    cmd_trigers: str = "cmd_trigers.json"

    def get_triggers_path(self, base_dir: str | Path | None = None) -> Path:
        path = Path(self.cmd_trigers)
        if path.is_absolute():
            return path

        base = Path(base_dir) if base_dir is not None else DATA_DIR
        return base / path

    def load_triggers(self, base_dir: str | Path | None = None) -> dict[str, list[str]]:
        path = self.get_triggers_path(base_dir)
        if not path.exists():
            fallback = DATA_DIR / self.cmd_trigers
            if fallback.exists():
                path = fallback
            else:
                return {}

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            return {}

        return {
            str(intent): [str(trigger) for trigger in triggers]
            for intent, triggers in payload.items()
            if isinstance(triggers, list)
        }


@dataclass
class AppConfig:
    name: str = "Newt"
    profiler: bool = False
    log: bool = False

    audio: AudioConfig = field(default_factory=AudioConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    op: OPConfig = field(default_factory=OPConfig)
    kws: KWSConfig = field(default_factory=KWSConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)


def load_config(config_path: str = "data/config.toml") -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        if DEFAULT_CONFIG_PATH.exists():
            path.write_text(
                DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8"
            )
        else:
            path.write_text(
                """[app]\nname = \"Newt\"\nprofiler = false\ndebug_server = false\n\n[op]\ncmd_trigers = \"cmd_trigers.json\"\n""",
                encoding="utf-8",
            )
        return load_config(str(path))

    with open(path, "rb") as f:
        data = tomllib.load(f)

    app: dict = data.get("app", {})

    res = AppConfig(
        name=app.get("name", "Newt"),
        profiler=app.get("profiler", False),
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
