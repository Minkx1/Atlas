"""Application logging setup and policy."""

from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"


def configure_logging(
    logs_dir: Path,
    *,
    enabled: bool = True,
    level: str = "INFO",
) -> None:
    """Configure process-wide logging once.

    Logging is diagnostic infrastructure: it must never be required for a
    subsystem to function and handlers must not raise into application code.
    """
    root = logging.getLogger()
    if getattr(root, "_atlas_configured", False):
        return

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if enabled:
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            logs_dir / "atlas.log",
            when="midnight",
            backupCount=14,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root._atlas_configured = True  # type: ignore[attr-defined]
