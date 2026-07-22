"""Week 2 — assemble the ML-ready feature matrix with leakage-safe labels.

Pipeline: Week-1 indicators panel -> add lag features (2.1) -> merge macro (2.2)
-> attach labels -> drop warm-up rows. Features are point-in-time at date t;
labels look forward from t; the two never overlap.

Labels are honest about missing futures: the last h sessions of every ticker
have no h-day forward return, so ``fwd_ret_h`` is NaN and ``label_up_h`` is
pandas ``NA`` (nullable Int8) — never a fabricated 0. Rows are kept (shorter
horizons may still be labelled); a model must drop NA rows for the label it
trains on.

Train/test uses a **purged, embargoed** time split (Jansen ML4T Ch.6 / Lopez de
Prado): a gap of ``embargo`` sessions separates train from test so a training
row's forward-looking label cannot spill into the test window. The embargo must
therefore *exceed* the longest label horizon — the default is
``max(FWD_HORIZONS) + 1``.
"""
from __future__ import annotations

import pandas as pd

from ..storage import ParquetStore
from ..utils import get_logger
from . import lags, macro
from .signals import FWD_HORIZONS

# Week-1 indicators curated as ML features. A deliberate modelling choice owned
# here (not by indicators.py); a rename upstream fails loudly in build_dataset.
_W1_FEATURES = ["rsi14", "macd_hist_norm", "bb_pctb", "bb_width"]


def feature_list() -> list[str]:
    return lags.feature_columns() + macro.macro_columns() + _W1_FEATURES


def label_list() -> list[str]:
    return [f"fwd_ret_{h}" for h in FWD_HORIZONS] + [f"label_up_{h}" for h in FWD_HORIZONS]


def attach_labels(feat: pd.DataFrame) -> pd.DataFrame:
    """Binary direction labels as nullable Int8: NA (never a fabricated 0)
    where the forward return does not exist yet (each ticker's last h rows)."""
    for h in FWD_HORIZONS:
        fwd = feat[f"fwd_ret_{h}"]
        feat[f"label_up_{h}"] = (fwd > 0).astype("Int8").where(fwd.notna())
    return feat


def build_dataset(cfg) -> pd.DataFrame:
    log = get_logger("features", cfg.path("logs"))
    processed = ParquetStore(cfg.path("processed"), cfg["storage"]["compression"])

    fwd_cols = [f"fwd_ret_{h}" for h in FWD_HORIZONS]
    ind = processed.read("sp500_indicators",
                         columns=lags.INPUT_COLS + _W1_FEATURES + fwd_cols)
    log.info("Indicators panel: %d rows x %d tickers", len(ind), ind["ticker"].nunique())

    feat = lags.compute_lag_features(ind)
    feat = macro.add_macro(feat, macro.fetch_macro(cfg))
    feat = attach_labels(feat)

    matrix = feat[["date", "ticker"] + feature_list() + label_list()]
    before = len(matrix)
    matrix = matrix.dropna(subset=feature_list()).reset_index(drop=True)
    log.info("Feature matrix: %d rows (dropped %d warm-up/NaN) x %d features",
             len(matrix), before - len(matrix), len(feature_list()))

    processed.write(matrix, "features")
    return matrix


def purged_split(df: pd.DataFrame, train_frac: float = 0.7,
                 embargo: int | None = None):
    """Time-ordered split with an ``embargo``-session purge gap between train
    and test. Defaults to one session more than the longest label horizon, the
    minimum that fully prevents label spillover."""
    if embargo is None:
        embargo = max(FWD_HORIZONS) + 1
    dates = sorted(df["date"].unique())
    cut = int(len(dates) * train_frac)
    test_start = dates[cut]
    train_end = dates[max(cut - embargo, 0)]
    train = df[df["date"] <= train_end].reset_index(drop=True)
    test = df[df["date"] >= test_start].reset_index(drop=True)
    return train, test
