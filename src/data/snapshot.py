"""Write a reproducibility snapshot of the raw dataset.

Records, per raw file: row count, sha256 content hash, and file mtime (UTC),
plus the library versions used. Because adjusted prices are non-stationary
(future dividends rewrite history), this pins exactly what was fetched and
lets a later run detect drift.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json

import pandas as pd


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_snapshot(cfg) -> dict:
    import numpy
    import pyarrow
    import yfinance
    files = sorted(cfg.path("raw").glob("*.parquet"))
    entries = {}
    total_rows = 0
    for f in files:
        n = len(pd.read_parquet(f, columns=["date"]))
        total_rows += n
        entries[f.stem] = {
            "rows": int(n),
            "sha256": _sha256(f),
            "mtime_utc": dt.datetime.fromtimestamp(
                f.stat().st_mtime, dt.UTC
            ).isoformat(),
        }
    snap = {
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "n_files": len(files),
        "total_rows": total_rows,
        "date_range": {"start": cfg.start_date, "end": cfg.end_date},
        "versions": {
            "pandas": pd.__version__,
            "numpy": numpy.__version__,
            "pyarrow": pyarrow.__version__,
            "yfinance": yfinance.__version__,
        },
        "auto_adjust": cfg["data"]["auto_adjust"],
        "files": entries,
    }
    out = cfg.path("metadata") / "data_snapshot.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=2)
    return snap
