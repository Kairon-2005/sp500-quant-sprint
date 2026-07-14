#!/usr/bin/env python
"""Step 1 — fetch & cache the S&P 500 constituent universe."""
import argparse

import _bootstrap  # noqa: F401

from src.config import load_config
from src.data.universe import get_sp500_constituents, get_ticker_list


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch S&P 500 constituents")
    ap.add_argument("--refresh", action="store_true", help="ignore cache, re-scrape")
    args = ap.parse_args()

    cfg = load_config()
    df = get_sp500_constituents(cfg, force_refresh=args.refresh)
    tickers = get_ticker_list(cfg)

    print(f"S&P 500 constituents: {len(df)} rows")
    print(f"Download universe (incl. extras): {len(tickers)} tickers")
    if "gics_sector" in df.columns:
        print("\nBy GICS sector:")
        print(df["gics_sector"].value_counts().to_string())
    print(f"\nCached -> {cfg.resolve(cfg['universe']['cache'])}")


if __name__ == "__main__":
    main()
