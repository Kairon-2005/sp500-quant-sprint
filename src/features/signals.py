"""Signal analysis: do the indicators relate to *future* returns?

Two complementary views:

* **Information Coefficient (IC).** For each trading day we rank all stocks by a
  signal and by their subsequent H-day return, then take the Spearman (rank)
  correlation across the cross-section. The time-series of daily ICs is
  summarised by mean IC, IC volatility, information ratio (mean/std) and a
  t-stat. This is the standard quant measure of predictive value and is robust
  to scale/outliers. (Positive mean IC => higher signal predicts higher return.)

* **RSI overbought/oversold buckets.** Pooling all (stock, day) observations, we
  bucket RSI and report the mean forward return per bucket — the classic test of
  whether oversold (RSI<30) is followed by bounce-backs (mean reversion) or
  further losses (momentum).

Caveat: forward returns overlap, so daily ICs are autocorrelated; the t-stats
are indicative, not a substitute for a proper backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FWD_HORIZONS = (1, 5, 20)
# Signals evaluated for predictive value (all cross-sectionally comparable).
SIGNALS = ("rsi14", "bb_pctb", "ma20_gap", "mom_20", "macd_hist_norm")


def add_forward_returns(panel: pd.DataFrame,
                        horizons=FWD_HORIZONS) -> pd.DataFrame:
    """Per-ticker forward simple returns over each horizon (uses future data —
    for evaluation only, never as a live feature)."""
    panel = panel.sort_values(["ticker", "date"]).copy()
    g = panel.groupby("ticker", sort=False)["adj_close"]
    for h in horizons:
        panel[f"fwd_ret_{h}"] = g.shift(-h) / panel["adj_close"] - 1.0
    return panel


def daily_ic(df: pd.DataFrame, signal: str, fwd: str) -> pd.Series:
    """Vectorised cross-sectional Spearman IC per date (rank-then-Pearson)."""
    d = df[["date", signal, fwd]].dropna()
    # Need at least a few names per day for a meaningful correlation.
    counts = d.groupby("date")[signal].transform("size")
    d = d[counts >= 20]
    if d.empty:
        return pd.Series(dtype="float64")
    g = d.groupby("date")
    rs = g[signal].rank()
    rf = g[fwd].rank()
    rs_d = rs - rs.groupby(d["date"]).transform("mean")
    rf_d = rf - rf.groupby(d["date"]).transform("mean")
    num = (rs_d * rf_d).groupby(d["date"]).sum()
    den = np.sqrt((rs_d**2).groupby(d["date"]).sum() *
                  (rf_d**2).groupby(d["date"]).sum())
    return (num / den).replace([np.inf, -np.inf], np.nan).dropna()


def ic_summary(df: pd.DataFrame, signals=SIGNALS,
               horizons=FWD_HORIZONS) -> pd.DataFrame:
    """Mean IC, IC std, information ratio and t-stat for each signal x horizon."""
    rows = []
    for sig in signals:
        for h in horizons:
            ic = daily_ic(df, sig, f"fwd_ret_{h}")
            if len(ic) == 0:
                continue
            mean, std, n = ic.mean(), ic.std(), len(ic)
            rows.append({
                "signal": sig,
                "horizon": h,
                "mean_ic": round(mean, 4),
                "ic_std": round(std, 4),
                "info_ratio": round(mean / std, 3) if std else np.nan,
                "t_stat": round(mean / std * np.sqrt(n), 2) if std else np.nan,
                "n_days": int(n),
            })
    return pd.DataFrame(rows)


def rsi_bucket_analysis(df: pd.DataFrame, horizons=FWD_HORIZONS) -> pd.DataFrame:
    """Mean forward return by RSI bucket (pooled across all stocks/days)."""
    d = df[["rsi14"] + [f"fwd_ret_{h}" for h in horizons]].dropna(subset=["rsi14"])
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    d = d.assign(rsi_bucket=pd.cut(d["rsi14"], bins))
    agg = {f"fwd_ret_{h}": "mean" for h in horizons}
    out = d.groupby("rsi_bucket", observed=True).agg(agg)
    out["count"] = d.groupby("rsi_bucket", observed=True).size()
    # Convert mean returns to basis points for readability.
    for h in horizons:
        out[f"fwd_ret_{h}"] = (out[f"fwd_ret_{h}"] * 1e4).round(1)  # bps
    out = out.rename(columns={f"fwd_ret_{h}": f"fwd_{h}d_bps" for h in horizons})
    return out.reset_index()


def signal_zone_summary(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Compare classic RSI zones: oversold (<30) vs neutral vs overbought (>70)."""
    d = df[["rsi14", f"fwd_ret_{horizon}"]].dropna()
    zone = pd.cut(d["rsi14"], [0, 30, 70, 100],
                  labels=["oversold(<30)", "neutral(30-70)", "overbought(>70)"])
    g = d.groupby(zone, observed=True)[f"fwd_ret_{horizon}"]
    return pd.DataFrame({
        "n": g.size(),
        f"mean_fwd_{horizon}d_bps": (g.mean() * 1e4).round(1),
        "hit_rate_up": (g.apply(lambda x: (x > 0).mean())).round(3),
    }).reset_index()
