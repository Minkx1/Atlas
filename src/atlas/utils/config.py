"""
utils / config.py

Config API has two layers:

- `Config.load_raw` / `Config.write_from_example` -- the low-level escape
  hatch. Reads/writes a file under CONFIG_DIR verbatim (toml/json/txt/cfg),
  generating it from an "origin" string if missing. Use this directly for
  files that don't have a fixed schema (commands.json's intents are
  user-defined) or that are a single raw value (keybinds.cfg).

- `@config(file=..., table=...)` -- a class decorator for the common case:
  a typed dataclass mapped onto one TOML table. It generates its own
  example content from the dataclass's field defaults (so the example can
  never drift from what the code actually reads) and exposes `.load()`,
  returning an instance you can also construct directly with overrides
  (e.g. in tests: `TtsConfig(volume=0.1)`).

`GeneralConfig` (below) is a small hand-written composite on top of
`Config.load_raw` for `general.toml` -- settings that more than one module
needs (assistant identity, shared audio format). It isn't built with
`@config` because that decorator assumes one dataclass owns a whole TOML
table; `general.toml` has multiple tables shared across modules, and
generating its example content has to happen for the file as a whole, in
one shot -- see `GeneralConfig.load`.

"""

from __future__ import annotations

import json
import logging
import os
import sys
import tomllib
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Literal, get_args

log = logging.getLogger(__name__)

# GENERAL CONFIGURATIONS : OS Name, Base directory and other directories

OS_NAME = os.name
if OS_NAME not in {"posix", "nt"}:
    raise OSError(f"Unsupported OS: {OS_NAME}")


def _get_base_dir() -> Path:
    is_compiled = getattr(sys, "frozen", False) or "__compiled__" in globals()
    if is_compiled:
        exe_dir = Path(sys.executable).resolve().parent
        if exe_dir.name == "bin":
            return exe_dir.parent
        return exe_dir

    # Not frozen: don't assume a fixed number of parents above this file.
    # That only holds when `atlas` is physically at <project_root>/src/atlas/
    # (an unpacked source tree or an editable install) -- a regular
    # `pip install .` copies the package into site-packages, far away from
    # the project's actual data/config/plugins directories. Atlas is a
    # standalone app meant to be run from its own project directory, so walk
    # up from the current working directory for the project root instead.
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


BASE_DIR: Path = _get_base_dir()

DATA_DIR = BASE_DIR / "data"
PLUGINS_DIR = BASE_DIR / "plugins"
CONFIG_DIR = BASE_DIR / "config"

DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.toml"


class Config:
    """Low-level file loader. See module docstring for when to use this
    directly versus the `@config` decorator."""

    EXTENSIONS = Literal[".toml", ".json", ".txt", ".cfg"]  # cfg is equivalent of txt

    @classmethod
    def _load_file(cls, file: Path) -> dict:
        if file.name == "null":
            return {}

        if file.suffix not in get_args(Config.EXTENSIONS):
            log.warning("Unsupported extension for config %s", file.name)
            return {}

        match file.suffix:
            case ".toml":
                with file.open("rb") as f:
                    return tomllib.load(f)
            case ".json":
                with file.open(mode="rb") as f:
                    return json.load(f)
            case ".txt" | ".cfg":
                with file.open(mode="r") as f:
                    return {"content": f.read()}
            case _:
                raise RuntimeError(f"Unsupported config extension: {file.suffix}.")

        return {}

    @classmethod
    def _write_file(cls, file: Path, origin: str = "") -> None:
        if file.name == "null":
            return

        file.write_text(origin, encoding="utf-8")

    @classmethod
    def load_raw(cls, name: str = "null", origin: str = "") -> dict:
        """Read config file by **id** and returns its content.

        If file is missing, generates it from **origin**.
        """

        file = CONFIG_DIR / name

        if name != "null" and not file.exists():
            cls._write_file(file, origin)

        if file.exists() and file.is_file():
            return cls._load_file(file)

        return {}

    @classmethod
    def write_from_example(cls, path: Path, example: Path) -> str:
        origin: str = example.read_text("utf-8")
        path.write_text(origin)
        return origin


# --- Typed config sections -------------------------------------------------


def _toml_literal(value: object) -> str:
    """Serializes a Python scalar as a TOML literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    raise TypeError(f"Unsupported TOML default value type: {type(value)!r}")


def _dump_section(cls: type, table: str | None) -> str:
    """Renders a dataclass's field defaults as a TOML table -- this is what
    `@config` writes to disk the first time a config file is missing, so the
    generated example can never drift from what the code actually reads."""
    lines = [f"[{table}]"] if table else []
    for f in fields(cls):
        if f.default is not MISSING:
            default = f.default
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            default = f.default_factory()  # type: ignore[misc]
        else:
            raise TypeError(
                f"Config field '{f.name}' on {cls.__name__} needs a default"
            )
        lines.append(f"{f.name} = {_toml_literal(default)}")
    return "\n".join(lines) + "\n"


def _known_fields(cls: type, data: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


def config(*, file: str, table: str | None = None):
    """Class decorator: turns a class into a self-loading, self-documenting
    TOML config section, backed by `Config.load_raw`.

    Usage:
        @config(file="tts.toml", table="tts")
        class TtsConfig:
            volume: float = 1.0

        cfg = TtsConfig.load()          # from config/tts.toml, generated if missing
        cfg = TtsConfig(volume=0.1)     # plain dataclass -- for tests, no I/O

    If more than one class needs fields from the same TOML table (e.g.
    `stt.toml`'s `[stt]` table is read by both the state machine and the
    Whisper wrapper), define ONE dataclass covering the union of fields and
    have every consumer take an instance of it as a constructor parameter --
    don't decorate multiple partial dataclasses against the same table, or
    whichever one is constructed first will generate an incomplete file.
    """

    def wrap(cls: type) -> type:
        cls = dataclass(cls)

        def load(overrides: dict[str, object] | None = None):
            example = _dump_section(cls, table)
            raw = Config.load_raw(file, example)
            data = raw.get(table, {}) if table else raw
            values = _known_fields(cls, data)
            if overrides:
                values.update(overrides)
            return cls(**values)

        cls.load = staticmethod(load)
        return cls

    return wrap


# --- Shared/general config --------------------------------------------------
#
# Settings more than one module needs, so they don't get copy-pasted into
# every module's own TOML file (assistant identity used to be duplicated in
# both op.toml and tts.toml; audio capture format lived only in stt.toml even
# though it describes the whole audio pipeline, not just STT).


@dataclass
class IdentityConfig:
    name: str = "Atlas"
    username: str = "Sir"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    blocksize: int = 512
    dtype: str = "float32"


@dataclass
class GeneralConfig:
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)

    @staticmethod
    def load() -> GeneralConfig:
        # Both sections live in the same physical file, so the example has to
        # be generated for the file as a whole in one shot -- if IdentityConfig
        # and AudioConfig were each decorated with @config independently,
        # whichever loaded first would write general.toml with only its own
        # section, and the other would silently fall back to its defaults.
        example = (
            _dump_section(IdentityConfig, "identity")
            + "\n"
            + _dump_section(AudioConfig, "audio")
        )
        raw = Config.load_raw("general.toml", example)
        identity = IdentityConfig(
            **_known_fields(IdentityConfig, raw.get("identity", {}))
        )
        audio = AudioConfig(**_known_fields(AudioConfig, raw.get("audio", {})))
        return GeneralConfig(identity=identity, audio=audio)


general_cfg = GeneralConfig.load()
