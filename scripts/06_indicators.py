#!/usr/bin/env python
"""Step 6 — compute technical indicators + forward returns on the clean panel."""
import _bootstrap  # noqa: F401
import pandas as pd

from src.config import load_config
from src.features.indicators import compute_indicators
from src.features.signals import add_forward_returns
from src.utils import get_logger


def main() -> None:
    cfg = load_config()
    log = get_logger("indicators", cfg.path("logs"))

    clean = cfg.path("processed") / "sp500_clean_panel.parquet"
    panel = pd.read_parquet(clean)
    log.info("Loaded clean panel: %d rows x %d tickers", len(panel), panel["ticker"].nunique())

    ind = compute_indicators(panel)
    ind = add_forward_returns(ind)

    out = cfg.path("processed") / "sp500_indicators.parquet"
    ind.to_parquet(out, engine="pyarrow", compression=cfg["storage"]["compression"], index=False)
    log.info("Indicators panel: %d rows x %d cols -> %s (%.0f MB)",
             len(ind), ind.shape[1], out, out.stat().st_size / 1e6)

    print("\n=== Indicators computed ===")
    print(f"rows x cols : {len(ind):,} x {ind.shape[1]}")
    added = [c for c in ind.columns if c.startswith(("ma", "ema", "macd", "rsi", "bb_", "atr", "mom", "fwd"))]
    print(f"new columns : {added}")
    aapl = ind[ind.ticker == "AAPL"].iloc[-1]
    print("\nAAPL latest snapshot:")
    for c in ["date", "adj_close", "ma20", "ma50", "rsi14", "macd", "macd_signal", "bb_pctb"]:
        print(f"  {c:12s}: {aapl[c]}")


if __name__ == "__main__":
    main()
