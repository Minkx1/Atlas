#
# core / config.py
#

from __future__ import annotations

import json
import logging
import os
import sys

# from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

import tomllib

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
    return (
        Path(__file__).resolve().parents[3]
    )  # '/src/core/config.py'.parent.parent.parent is '/'


BASE_DIR: Path = _get_base_dir()

DATA_DIR = BASE_DIR / "data"
PLUGINS_DIR = BASE_DIR / "plugins"
CONFIG_DIR = BASE_DIR / "config"

DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.toml"


class Config:
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
    def load_config(cls, id: str = "null", origin: str = "") -> dict:
        """Read config file by **id** and returns its content.

        If file is missing, generates it from **origin**.
        """

        file = CONFIG_DIR / id

        if id != "null" and not file.exists():
            cls._write_file(file, origin)

        if file.exists() and file.is_file():
            return cls._load_file(file)

        return {}

    @classmethod
    def write_from_eample(cls, path: Path, example: Path) -> str:
        origin: str = example.read_text("utf-8")
        path.write_text(origin)
        return origin
