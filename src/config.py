"""Configuration loading and path resolution.

Centralises access to ``config/config.yaml`` and resolves all relative
paths against the project root so scripts can be run from anywhere.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import yaml

# Project root = two levels up from this file (src/config.py -> src -> root).
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"


class Config:
    """Thin wrapper around the YAML config with helpers for paths/dates."""

    def __init__(self, data: dict[str, Any]):
        self._d = data

    # -- dict-style access -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

    @property
    def raw(self) -> dict[str, Any]:
        return self._d

    # -- path helpers ------------------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve a directory from the ``paths`` section against ROOT."""
        p = ROOT / self._d["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolve(self, relpath: str) -> Path:
        """Resolve an arbitrary config-relative path against ROOT."""
        p = ROOT / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # -- date helpers ------------------------------------------------------
    @property
    def start_date(self) -> str:
        return self._d["data"]["start_date"]

    @property
    def end_date(self) -> str:
        """End date, defaulting to today (inclusive) when unset in config."""
        end = self._d["data"].get("end_date")
        if end:
            return str(end)
        return dt.date.today().isoformat()


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data)
