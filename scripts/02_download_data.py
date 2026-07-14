#!/usr/bin/env python
"""Step 2 — download 10y daily OHLCV for the universe (resumable)."""
import argparse

import _bootstrap  # noqa: F401

from src.config import load_config
from src.data.download import download_ohlcv
from src.data.universe import get_ticker_list


def main() -> None:
    ap = argparse.ArgumentParser(description="Download S&P 500 OHLCV")
    ap.add_argument("--force", action="store_true", help="re-download cached tickers")
    ap.add_argument("--limit", type=int, default=None,
                    help="only download the first N tickers (for testing)")
    ap.add_argument("--tickers", nargs="*", help="explicit ticker subset")
    args = ap.parse_args()

    cfg = load_config()
    tickers = args.tickers or get_ticker_list(cfg)
    if args.limit:
        tickers = tickers[: args.limit]

    manifest = download_ohlcv(cfg, tickers, force=args.force)

    print("\n=== Download manifest (status counts) ===")
    print(manifest["status"].value_counts().to_string())
    if (manifest["status"] == "ok").any():
        by_src = manifest.loc[manifest.status == "ok", "source"].value_counts()
        print("\nSource used (ok tickers):")
        print(by_src.to_string())


if __name__ == "__main__":
    main()
