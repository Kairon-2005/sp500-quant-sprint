"""Week 2.1 — lag / time-series features (Alpha158-style + Jansen's 5 families).

Every feature at date t uses only information available at t's close, so the
matrix is free of look-ahead; forward-return *labels* (built separately in
``dataset.py``) are the only thing that peeks forward.

Feature families (Jansen ML4T Ch.8) grounded in Qlib's Alpha158 formulas:
    momentum/reversal : ROC returns over 1/5/10/20/60d, 12-1 momentum
    volatility        : realized return std over 5/10/20/60d, vol regime, skew/kurt
    microstructure    : candlestick K-bars (KMID/KLEN/KUP/KLOW/KSFT)
    price position    : MA-gap, stochastic RSV, distance to rolling hi/lo, up/down counts
    liquidity/volume  : volume-vs-MA ratio, log dollar volume, Amihud illiquidity, price-vol corr

Returns are ALWAYS recomputed from ``adj_close`` (never taken from the panel's
``ret`` column, which is deliberately NaN on halt/filled sessions): every
feature family shares one guaranteed basis, and a lone halt-day NaN cannot null
entire rolling windows. Feature names come from the same dict that builds the
columns, so ``feature_columns()`` can never drift from ``add_lag_features``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12
RET_WINDOWS = (1, 5, 10, 20, 60)
VOL_WINDOWS = (5, 10, 20, 60)
MA_GAP_WINDOWS = (5, 20, 60)
POS_WINDOWS = (20, 60)

# Input columns add_lag_features needs from the panel.
INPUT_COLS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


def _build_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """All lag features for one date-sorted ticker frame, as {name: series}."""
    px = df["adj_close"]
    ret = px.pct_change(fill_method=None)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    v = df["volume"].astype("float64")
    cv = c * v                                        # dollar volume, reused below
    f: dict[str, pd.Series] = {}

    # -- momentum / reversal (ROC) --
    for w in RET_WINDOWS:
        f[f"roc_{w}"] = px / px.shift(w) - 1.0
    f["mom_12_1"] = px.shift(21) / px.shift(252) - 1.0   # 12m return excl. last month

    # -- realized volatility & shape of the return distribution --
    for w in VOL_WINDOWS:
        f[f"vol_{w}"] = ret.rolling(w).std()
    f["vol_regime"] = f["vol_5"] / (f["vol_20"] + EPS)   # short/long vol ratio
    f["skew_20"] = ret.rolling(20).skew()
    f["kurt_20"] = ret.rolling(20).kurt()

    # -- candlestick K-bars (intraday shape; ratios are scale-free) --
    f["kmid"] = (c - o) / (o + EPS)
    f["klen"] = (h - l) / (o + EPS)
    f["kmid2"] = (c - o) / (h - l + EPS)
    f["kup"] = (h - np.maximum(o, c)) / (o + EPS)
    f["klow"] = (np.minimum(o, c) - l) / (o + EPS)
    f["ksft"] = (2 * c - h - l) / (o + EPS)

    # -- price position / trend --
    for w in MA_GAP_WINDOWS:
        f[f"ma_gap_{w}"] = px / px.rolling(w).mean() - 1.0
    for w in POS_WINDOWS:
        lo, hi = px.rolling(w).min(), px.rolling(w).max()
        f[f"rsv_{w}"] = (px - lo) / (hi - lo + EPS)      # stochastic %K
        f[f"higap_{w}"] = hi / px - 1.0                  # distance below rolling high
        f[f"logap_{w}"] = lo / px - 1.0                  # distance above rolling low

    # -- up/down day statistics (CNTP/CNTD) --
    cntp = (ret > 0).astype("float64").rolling(20).mean()
    f["cntp_20"] = cntp
    f["cntd_20"] = cntp - (ret < 0).astype("float64").rolling(20).mean()

    # -- liquidity / volume --
    for w in (5, 20):
        f[f"vratio_{w}"] = v / (v.rolling(w).mean() + EPS)
    f["dollar_vol"] = np.log(cv + 1.0)
    f["amihud_20"] = (ret.abs() / (cv + EPS)).rolling(20).mean() * 1e9
    f["corr_pv_20"] = px.rolling(20).corr(np.log(v + 1.0))  # price-volume corr
    return f


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach lag features to one ticker's frame in a single batch (no
    per-column insertion, so no frame fragmentation)."""
    df = df.sort_values("date").reset_index(drop=True)
    return df.assign(**_build_features(df))


def feature_columns() -> list[str]:
    """Feature names, derived from the same builder that creates the columns."""
    stub = pd.DataFrame({
        "date": pd.bdate_range("2000-01-03", periods=2),
        "ticker": "STUB",
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
        "adj_close": 1.0, "volume": 1.0,
    })
    return list(_build_features(stub))


def compute_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply ``add_lag_features`` to every ticker in the panel."""
    frames = [add_lag_features(g) for _, g in panel.groupby("ticker", sort=False)]
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
