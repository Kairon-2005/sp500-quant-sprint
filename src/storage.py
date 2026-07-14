"""Parquet storage repository — one IO surface for every pipeline stage.

Centralises the ``to_parquet(engine=..., compression=..., index=...)`` calls and
the filesystem-safe ticker naming that were previously duplicated across the
download, clean, panel and indicator stages.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd


def safe_name(ticker: str) -> str:
    """Filesystem-safe file stem for a ticker ('^GSPC' -> '_GSPC')."""
    return ticker.replace("^", "_").replace("/", "_")


class ParquetStore:
    """A directory of Parquet files addressed by ticker/name."""

    def __init__(self, root: Path | str, compression: str = "zstd"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.compression = compression

    def path(self, name: str) -> Path:
        return self.root / f"{safe_name(name)}.parquet"

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def write(self, df: pd.DataFrame, name: str, index: bool = False) -> Path:
        p = self.path(name)
        df.to_parquet(p, engine="pyarrow", compression=self.compression, index=index)
        return p

    def read(self, name: str, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(self.path(name), columns=columns)

    def files(self) -> list[Path]:
        return sorted(self.root.glob("*.parquet"))

    def iter_frames(self, columns: list[str] | None = None) -> Iterator[pd.DataFrame]:
        for p in self.files():
            yield pd.read_parquet(p, columns=columns)
