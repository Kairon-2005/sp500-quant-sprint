#!/usr/bin/env python
"""Step 10 — cross-source reconciliation (Yahoo vs Stooq) to catch bad quotes."""
import argparse

import _bootstrap  # noqa: F401

from src.config import load_config
from src.data.reconcile import reconcile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60, help="tickers to check (0 = all)")
    ap.add_argument("--tickers", nargs="*", help="explicit ticker subset")
    args = ap.parse_args()

    cfg = load_config()
    df = reconcile(cfg, tickers=args.tickers, sample=(args.sample or None))

    print("\n=== Reconciliation (Yahoo vs Stooq, raw close) ===")
    print(df["status"].value_counts().to_string())
    flagged = df[df["status"].isin(["mismatch", "warn"])]
    if len(flagged):
        cols = ["ticker", "status", "return_corr", "median_abs_pct",
                "n_days_gt_2pct", "worst_date"]
        print("\n=== Flagged for inspection ===")
        print(flagged[[c for c in cols if c in flagged.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
