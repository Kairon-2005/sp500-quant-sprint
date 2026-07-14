#!/usr/bin/env python
"""Step 4 — consolidate per-ticker files into the analysis panel."""
import _bootstrap  # noqa: F401

from src.config import load_config
from src.data.panel import build_panel


def main() -> None:
    cfg = load_config()
    panel = build_panel(cfg)
    print("\n=== Consolidated panel ===")
    print(f"rows       : {len(panel):,}")
    print(f"tickers    : {panel['ticker'].nunique()}")
    print(f"date range : {panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(f"columns    : {list(panel.columns)}")


if __name__ == "__main__":
    main()
