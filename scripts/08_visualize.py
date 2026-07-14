#!/usr/bin/env python
"""Step 8 — visualise indicators and the signal analysis (saves PNGs)."""
import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from src.config import load_config
from src.features.signals import FWD_HORIZONS, ic_summary, rsi_bucket_analysis
from src.features.viz import plot_ic_heatmap, plot_price_panel, plot_rsi_buckets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL", help="ticker for the price panel")
    ap.add_argument("--last", type=int, default=504, help="sessions to show")
    args = ap.parse_args()

    cfg = load_config()
    figdir = cfg.resolve("reports/figures/_.png").parent
    ind = pd.read_parquet(cfg.path("processed") / "sp500_indicators.parquet")

    saved = []
    d = ind[ind.ticker == args.ticker]
    if d.empty:
        raise SystemExit(f"No data for ticker {args.ticker}")
    saved.append(plot_price_panel(d, args.ticker, figdir / f"{args.ticker}_price_panel.png",
                                  last_n=args.last))

    buckets = rsi_bucket_analysis(ind)
    saved.append(plot_rsi_buckets(buckets, FWD_HORIZONS, figdir / "rsi_forward_returns.png"))

    ic = ic_summary(ind)
    saved.append(plot_ic_heatmap(ic, figdir / "ic_heatmap.png"))

    print("Saved figures:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
