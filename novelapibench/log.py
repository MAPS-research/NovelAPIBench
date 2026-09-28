"""Logging setup (loguru)."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}")
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=level, encoding="utf-8",
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {name}:{line} - {message}")


__all__ = ["logger", "setup_logging"]
