"""Week 2.2 — market-sentiment & macro features.

Free, reliably-fetchable series via the existing YahooSource:
    ^VIX  implied-volatility "fear" index    ^GSPC  S&P 500 index level
    ^TNX  10-year Treasury yield (%)         ^IRX   13-week T-bill yield (%)

Merged onto the stock panel with a **backward as-of join**, so each (ticker,
date) gets the most recent macro value at or before that date — no future macro
leaks in.

Social sentiment (the spec's "Twitter sentiment") — honest verdict: the X/Twitter
API has been paid/closed since 2023, so there is no free, reliable *historical*
tweet-sentiment feed. We use VIX as the market fear/sentiment proxy and expose a
pluggable hook; a real social feed (X API, or a vendor like RavenPack) can be
added behind the same interface. See ``sentiment_note``.
"""
from __future__ import annotations

import pandas as pd

from ..data.sources import YahooSource

_SERIES = {"^VIX": "vix", "^GSPC": "spx", "^TNX": "tnx10y", "^IRX": "irx13w"}


def fetch_macro(cfg) -> pd.DataFrame:
    """Fetch macro series and derive point-in-time features (indexed by date)."""
    raw = YahooSource().fetch(list(_SERIES), cfg.start_date, cfg.end_date,
                              interval=cfg["data"]["interval"], auto_adjust=False)
    closes = {}
    for tkr, name in _SERIES.items():
        df = raw.get(tkr)
        if df is not None and not df.empty:
            closes[name] = df.set_index("date")["close"]
    m = pd.DataFrame(closes).sort_index()
    m = m.ffill()  # carry the last quote across any missing macro day (past info only)

    feat = pd.DataFrame(index=m.index)
    # Fear: VIX level, daily change, and its rolling 1y z-score.
    feat["vix"] = m["vix"]
    feat["vix_chg"] = m["vix"].diff()
    feat["vix_z"] = (m["vix"] - m["vix"].rolling(252).mean()) / m["vix"].rolling(252).std()
    # Rates: 10y & 13w yields, 10y daily change, and the term spread.
    feat["tnx10y"] = m["tnx10y"]
    feat["tnx_chg"] = m["tnx10y"].diff()
    feat["term_spread"] = m["tnx10y"] - m["irx13w"]
    # Market: index return, 20d realized vol, and trend vs its 50d average.
    spx_ret = m["spx"].pct_change(fill_method=None)
    feat["spx_ret_1"] = spx_ret
    feat["spx_vol_20"] = spx_ret.rolling(20).std()
    feat["spx_ma_gap"] = m["spx"] / m["spx"].rolling(50).mean() - 1.0
    return feat.reset_index().rename(columns={"index": "date"})


def macro_columns() -> list[str]:
    return ["vix", "vix_chg", "vix_z", "tnx10y", "tnx_chg", "term_spread",
            "spx_ret_1", "spx_vol_20", "spx_ma_gap"]


def add_macro(panel: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Backward as-of merge of macro features onto the (ticker, date) panel."""
    panel = panel.copy()
    macro = macro.copy()
    # merge_asof requires identical datetime resolution on both keys.
    panel["date"] = panel["date"].astype("datetime64[ns]")
    macro["date"] = macro["date"].astype("datetime64[ns]")
    merged = pd.merge_asof(panel.sort_values("date"), macro.sort_values("date"),
                           on="date", direction="backward")
    return merged.sort_values(["ticker", "date"]).reset_index(drop=True)


def sentiment_note() -> str:
    return ("Social/Twitter sentiment omitted: X API is paid/closed, no free reliable "
            "historical feed. VIX is used as the fear proxy; plug a vendor feed behind "
            "add_macro() to add true social sentiment.")
