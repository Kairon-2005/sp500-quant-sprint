"""Week 2 — assemble the ML-ready feature matrix with leakage-safe labels.

Pipeline: Week-1 indicators panel -> add lag features (2.1) -> merge macro (2.2)
-> attach forward-return labels -> drop warm-up rows. Features are point-in-time
at date t; labels look forward from t; the two never overlap.

Train/test uses a **purged, embargoed** time split (Jansen ML4T Ch.6 / Lopez de
Prado): a gap of ``embargo`` sessions separates train from test so a training
row's forward-looking label cannot spill into the test window.
"""
from __future__ import annotations

import pandas as pd

from ..storage import ParquetStore
from ..utils import get_logger
from . import lags, macro
from .signals import FWD_HORIZONS, add_forward_returns

# Curated Week-1 indicators reused as features (scale-free / cross-sectional).
_W1_FEATURES = ["rsi14", "macd_hist_norm", "bb_pctb", "bb_width"]


def feature_list() -> list[str]:
    return lags.feature_columns() + macro.macro_columns() + _W1_FEATURES


def label_list() -> list[str]:
    return [f"fwd_ret_{h}" for h in FWD_HORIZONS] + [f"label_up_{h}" for h in FWD_HORIZONS]


def build_dataset(cfg) -> pd.DataFrame:
    log = get_logger("features", cfg.path("logs"))
    processed = ParquetStore(cfg.path("processed"), cfg["storage"]["compression"])

    ind = processed.read("sp500_indicators")
    log.info("Indicators panel: %d rows x %d tickers", len(ind), ind["ticker"].nunique())

    feat = lags.compute_lag_features(ind)
    if "fwd_ret_1" not in feat.columns:               # W1 panel already has these
        feat = add_forward_returns(feat)
    feat = macro.add_macro(feat, macro.fetch_macro(cfg))

    for h in FWD_HORIZONS:                             # binary direction labels
        feat[f"label_up_{h}"] = (feat[f"fwd_ret_{h}"] > 0).astype("int8")

    cols = ["date", "ticker"] + feature_list() + label_list()
    matrix = feat[cols].copy()

    before = len(matrix)
    matrix = matrix.dropna(subset=feature_list()).reset_index(drop=True)
    log.info("Feature matrix: %d rows (dropped %d warm-up/NaN) x %d features",
             len(matrix), before - len(matrix), len(feature_list()))

    processed.write(matrix, "features")
    return matrix


def purged_split(df: pd.DataFrame, train_frac: float = 0.7, embargo: int = 20):
    """Time-ordered split with an `embargo`-session purge gap between train/test."""
    dates = sorted(df["date"].unique())
    cut = int(len(dates) * train_frac)
    test_start = dates[cut]
    train_end = dates[max(cut - embargo, 0)]
    train = df[df["date"] <= train_end].reset_index(drop=True)
    test = df[df["date"] >= test_start].reset_index(drop=True)
    return train, test
