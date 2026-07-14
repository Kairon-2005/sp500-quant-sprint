"""Shared logging configuration: console + rotating file handler."""
from __future__ import annotations

import logging
from pathlib import Path

_CONFIGURED: set[str] = set()


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """Return a logger writing to console and, if given, ``log_dir/name.log``."""
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    _CONFIGURED.add(name)
    return logger
