"""Offline unit tests for the Week-1.2 cleaning transforms (no network)."""
import numpy as np
import pandas as pd

from src.config import load_config
from src.data.clean import (
    CleanStats,
    _align_calendar,
    _fix_mechanical,
    _flag_outliers,
    _trim_leading_placeholder,
    clean_ticker,
    nyse_sessions,
)


def _raw(n=250, start="2022-01-03", seed=1, lead_flat=0):
    """Synthetic raw frame in the download schema, optional leading flat block."""
    dates = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    df = pd.DataFrame({
        "date": dates,
        "open": close * 0.995, "high": close * 1.01,
        "low": close * 0.99, "close": close, "adj_close": close,
        "volume": pd.array(rng.integers(1e6, 5e6, n), dtype="Int64"),
        "dividends": 0.0, "splits": 0.0, "ticker": "TST", "source": "test",
    })
    if lead_flat:  # prepend flat, zero-volume placeholder rows
        pre = df.iloc[:lead_flat].copy()
        for c in ["open", "high", "low", "close", "adj_close"]:
            pre[c] = 50.0
        pre["volume"] = pd.array([0] * lead_flat, dtype="Int64")
        pre["date"] = pd.bdate_range(end=dates[0] - pd.Timedelta(days=1), periods=lead_flat)
        df = pd.concat([pre, df], ignore_index=True)
    return df


def test_trim_leading_placeholder():
    df = _raw(lead_flat=40)
    st = CleanStats(ticker="TST", raw_rows=len(df))
    out = _trim_leading_placeholder(df, st)
    assert st.leading_trimmed == 40
    assert len(out) == 250
    # No flat zero-volume rows survive at the head.
    assert not ((out.open == out.close) & (out.volume.fillna(0) == 0)).any()


def test_align_calendar_ffill_is_masked():
    df = _raw(n=60)
    # Remove a single interior session -> should be ffilled (<=5) and flagged.
    df = df.drop(index=30).reset_index(drop=True)
    st = CleanStats(ticker="TST", raw_rows=len(df))
    cal = nyse_sessions("2022-01-01", "2022-06-01")
    if len(cal) == 0:  # calendar lib unavailable -> skip alignment assertions
        return
    out = _align_calendar(df, cal, st, max_ffill=5)
    assert out["is_filled"].sum() >= 1
    # A filled row carries a price but is explicitly marked.
    filled = out[out["is_filled"]]
    assert filled["adj_close"].notna().all()


def test_flag_outliers_preserves_price_and_winsorises():
    df = _raw(n=250)
    df["is_halt"] = False
    df["is_filled"] = False
    df.loc[100, ["close", "adj_close"]] = df.loc[99, "adj_close"] * 1.5  # +50% real move
    st = CleanStats(ticker="TST", raw_rows=len(df))
    out = _flag_outliers(df, st, max_ret=0.20)
    assert out.loc[100, "is_extreme"]
    # Price is NOT mutated; the winsorised feature is tamer than the raw return.
    assert out.loc[100, "adj_close"] == df.loc[100, "adj_close"]
    assert abs(out.loc[100, "ret_winsor"]) < abs(out.loc[100, "ret"])


def test_fix_mechanical_logs_ohlc_fix():
    df = _raw(n=60)
    df.loc[10, "low"] = df.loc[10, "high"] * 1.05  # low above high (impossible)
    st = CleanStats(ticker="TST", raw_rows=len(df))
    logs: list = []
    out = _fix_mechanical(df, st, logs, min_price=0.01)
    assert st.ohlc_fixed == 1 and len(logs) == 1
    assert out.loc[10, "low"] <= out.loc[10, "high"]


def test_clean_ticker_end_to_end():
    df = _raw(n=300, lead_flat=30)
    cal = nyse_sessions("2021-01-01", "2023-06-01")
    out, st = clean_ticker(df, cal, load_config(), [])
    assert not st.dropped
    assert st.leading_trimmed == 30
    # No fabricated returns on non-trading rows.
    assert not (out["ret"].notna() & (out["is_halt"] | out["is_filled"])).any()
