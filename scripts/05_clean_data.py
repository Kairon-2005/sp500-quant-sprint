#!/usr/bin/env python
"""Step 5 — clean the raw OHLCV: missing values, outliers, standardisation.

Writes cleaned per-ticker files to data/interim/clean/, a consolidated
data/processed/sp500_clean_panel.parquet, an adjustments log, and a report.
Raw data is never modified.
"""
import _bootstrap  # noqa: F401

from src.config import load_config
from src.data.clean import clean_all


def main() -> None:
    cfg = load_config()
    stats = clean_all(cfg)

    kept = stats[~stats["dropped"]]
    print("\n=== Cleaning summary ===")
    print(f"tickers kept   : {len(kept)} / {len(stats)}")
    print(f"dropped        : {', '.join(stats.loc[stats.dropped, 'ticker']) or 'none'}")
    print(f"leading rows trimmed : {int(stats['leading_trimmed'].sum()):,}")
    print(f"interior halts masked: {int(stats['halt_masked'].sum()):,}")
    print(f"short gaps filled    : {int(stats['gap_filled'].sum()):,}")
    print(f"OHLC-order fixes     : {int(stats['ohlc_fixed'].sum())}")
    print(f"extreme moves flagged: {int(stats['n_extreme'].sum()):,} (kept, not altered)")

    print("\n=== Most-trimmed tickers (fabricated leading history) ===")
    top = stats.sort_values("leading_trimmed", ascending=False).head(6)
    print(top[["ticker", "raw_rows", "leading_trimmed", "clean_rows",
               "first_real_date"]].to_string(index=False))


if __name__ == "__main__":
    main()
