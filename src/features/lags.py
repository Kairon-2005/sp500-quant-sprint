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
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12
RET_WINDOWS = (1, 5, 10, 20, 60)
VOL_WINDOWS = (5, 10, 20, 60)
POS_WINDOWS = (20, 60)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach lag/time-series features to one ticker's (date-sorted) frame."""
    df = df.sort_values("date").copy()
    px = df["adj_close"]
    ret = df["ret"] if "ret" in df.columns else px.pct_change(fill_method=None)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    v = df["volume"].astype("float64")

    # -- momentum / reversal (ROC) --
    for w in RET_WINDOWS:
        df[f"roc_{w}"] = px / px.shift(w) - 1.0
    df["mom_12_1"] = px.shift(21) / px.shift(252) - 1.0      # 12m return excl. last month

    # -- realized volatility & shape of the return distribution --
    for w in VOL_WINDOWS:
        df[f"vol_{w}"] = ret.rolling(w).std()
    df["vol_regime"] = df["vol_5"] / (df["vol_20"] + EPS)    # short/long vol ratio
    df["skew_20"] = ret.rolling(20).skew()
    df["kurt_20"] = ret.rolling(20).kurt()

    # -- candlestick K-bars (intraday shape; ratios are scale-free) --
    df["kmid"] = (c - o) / (o + EPS)
    df["klen"] = (h - l) / (o + EPS)
    df["kmid2"] = (c - o) / (h - l + EPS)
    df["kup"] = (h - np.maximum(o, c)) / (o + EPS)
    df["klow"] = (np.minimum(o, c) - l) / (o + EPS)
    df["ksft"] = (2 * c - h - l) / (o + EPS)

    # -- price position / trend (all on adjusted close for consistency) --
    for w in (5, 20, 60):
        df[f"ma_gap_{w}"] = px / px.rolling(w).mean() - 1.0
    for w in POS_WINDOWS:
        lo, hi = px.rolling(w).min(), px.rolling(w).max()
        df[f"rsv_{w}"] = (px - lo) / (hi - lo + EPS)         # stochastic %K
        df[f"higap_{w}"] = hi / px - 1.0                     # distance below rolling high
        df[f"logap_{w}"] = lo / px - 1.0                     # distance above rolling low

    # -- up/down day statistics (CNTP/CNTD) --
    up, dn = (ret > 0).astype("float64"), (ret < 0).astype("float64")
    df["cntp_20"] = up.rolling(20).mean()
    df["cntd_20"] = up.rolling(20).mean() - dn.rolling(20).mean()

    # -- liquidity / volume --
    for w in (5, 20):
        df[f"vratio_{w}"] = v / (v.rolling(w).mean() + EPS)
    df["dollar_vol"] = np.log(c * v + 1.0)
    df["amihud_20"] = (ret.abs() / (c * v + EPS)).rolling(20).mean() * 1e9
    df["corr_pv_20"] = px.rolling(20).corr(np.log(v + 1.0))  # price-volume corr

    return df


# Names of the features this module adds (for downstream selection/reporting).
def feature_columns() -> list[str]:
    cols = [f"roc_{w}" for w in RET_WINDOWS] + ["mom_12_1"]
    cols += [f"vol_{w}" for w in VOL_WINDOWS] + ["vol_regime", "skew_20", "kurt_20"]
    cols += ["kmid", "klen", "kmid2", "kup", "klow", "ksft"]
    cols += [f"ma_gap_{w}" for w in (5, 20, 60)]
    cols += [f"rsv_{w}" for w in POS_WINDOWS]
    cols += [f"higap_{w}" for w in POS_WINDOWS] + [f"logap_{w}" for w in POS_WINDOWS]
    cols += ["cntp_20", "cntd_20", "vratio_5", "vratio_20",
             "dollar_vol", "amihud_20", "corr_pv_20"]
    return cols


def compute_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply ``add_lag_features`` to every ticker in the panel."""
    frames = [add_lag_features(g) for _, g in panel.groupby("ticker", sort=False)]
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
