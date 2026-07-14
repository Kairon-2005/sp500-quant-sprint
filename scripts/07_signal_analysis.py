#!/usr/bin/env python
"""Step 7 — do the indicators predict future returns? (IC + RSI zones)."""

import _bootstrap  # noqa: F401
import pandas as pd

from src.config import load_config
from src.features.signals import (
    ic_summary,
    rsi_bucket_analysis,
    signal_zone_summary,
)
from src.utils import get_logger


def main() -> None:
    cfg = load_config()
    log = get_logger("signals", cfg.path("logs"))
    ind = pd.read_parquet(cfg.path("processed") / "sp500_indicators.parquet")

    ic = ic_summary(ind)
    buckets = rsi_bucket_analysis(ind)
    zones5 = signal_zone_summary(ind, horizon=5)
    zones20 = signal_zone_summary(ind, horizon=20)

    meta = cfg.path("metadata")
    ic.to_csv(meta / "ic_summary.csv", index=False)
    buckets.to_csv(meta / "rsi_buckets.csv", index=False)

    print("\n=== Information Coefficient (mean daily cross-sectional Spearman) ===")
    print(ic.to_string(index=False))
    print("\n=== RSI zones vs forward 5d return ===")
    print(zones5.to_string(index=False))
    print("\n=== RSI zones vs forward 20d return ===")
    print(zones20.to_string(index=False))
    print("\n=== Forward return (bps) by RSI bucket ===")
    print(buckets.to_string(index=False))

    # Persist a compact markdown summary for the report folder.
    report = cfg.resolve("reports/week1_signal_analysis.md")
    best = ic.reindex(ic["mean_ic"].abs().sort_values(ascending=False).index).head(3)
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("# Week 1.3 — Signal analysis\n\n")
        fh.write("## Information Coefficient (higher |mean_ic| = more predictive)\n\n")
        fh.write(ic.to_markdown(index=False) + "\n\n")
        fh.write("Strongest signals by |IC|:\n\n" + best.to_markdown(index=False) + "\n\n")
        fh.write("## RSI overbought/oversold (forward 5d)\n\n")
        fh.write(zones5.to_markdown(index=False) + "\n\n")
        fh.write("## Mean forward return by RSI bucket\n\n")
        fh.write(buckets.to_markdown(index=False) + "\n")
    log.info("Signal analysis saved -> %s + ic_summary.csv, rsi_buckets.csv", report)


if __name__ == "__main__":
    main()
