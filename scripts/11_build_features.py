#!/usr/bin/env python
"""Step 11 — build the ML feature matrix (lag features + macro + labels)."""
import _bootstrap  # noqa: F401

from src.config import load_config
from src.features.dataset import build_dataset, feature_list, label_list
from src.features.lags import RET_WINDOWS
from src.features.macro import sentiment_note
from src.features.signals import ic_summary


def main() -> None:
    cfg = load_config()
    matrix = build_dataset(cfg)

    print("\n=== Feature matrix ===")
    print(f"rows      : {len(matrix):,}")
    print(f"tickers   : {matrix['ticker'].nunique()}")
    print(f"features  : {len(feature_list())}  |  labels: {label_list()}")
    print(f"date range: {matrix['date'].min().date()} -> {matrix['date'].max().date()}")
    print(f"\nNote: {sentiment_note()}")

    # 2.1 analysis — short-term momentum/reversal vs long-term trend, via IC.
    windows = [f"roc_{w}" for w in RET_WINDOWS] + ["mom_12_1"]
    ic = ic_summary(matrix, signals=windows, horizons=(5, 20))
    print("\n=== Return-window predictive power (mean cross-sectional IC) ===")
    print(ic.to_string(index=False))
    print("(negative IC at short windows = reversal; sign flip at long windows = trend)")


if __name__ == "__main__":
    main()
