"""Consolidate per-ticker Parquet files into analysis-ready datasets.

Outputs (data/processed/):
* ``sp500_ohlcv_panel.parquet``  — long format (date, ticker, OHLCV+adj+actions).
* ``adj_close_prices.parquet``   — wide matrix of adjusted close (date x ticker).
"""
from __future__ import annotations

import pandas as pd

from ..storage import ParquetStore
from ..utils import get_logger


def build_panel(cfg) -> pd.DataFrame:
    log = get_logger("panel", cfg.path("logs"))
    raw = ParquetStore(cfg.path("raw"))
    if not raw.files():
        raise RuntimeError(f"No raw parquet files in {raw.root}")

    panel = (pd.concat(raw.iter_frames(), ignore_index=True)
             .dropna(subset=["adj_close"])
             .sort_values(["ticker", "date"])
             .reset_index(drop=True))

    processed = ParquetStore(cfg.path("processed"), cfg["storage"]["compression"])
    panel_path = processed.write(panel, "sp500_ohlcv_panel")
    wide = panel.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
    processed.write(wide, "adj_close_prices", index=True)

    log.info("Panel: %d rows x %d cols | %d tickers | %.1f MB -> %s",
             len(panel), panel.shape[1], panel["ticker"].nunique(),
             panel_path.stat().st_size / 1e6, panel_path)
    log.info("Wide adj-close: %d dates x %d tickers", wide.shape[0], wide.shape[1])
    return panel
