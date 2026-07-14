"""Offline unit tests for core data-pipeline logic (no network)."""
import numpy as np
import pandas as pd

from src.config import load_config
from src.data.integrity import build_reference_calendar, check_ticker
from src.data.sources import STD_COLS, standardise
from src.data.universe import to_yahoo_symbol


def test_to_yahoo_symbol():
    assert to_yahoo_symbol("BRK.B") == "BRK-B"
    assert to_yahoo_symbol("bf.b") == "BF-B"
    assert to_yahoo_symbol("AAPL") == "AAPL"


def teststandardise_schema_and_cleaning():
    raw = pd.DataFrame(
        {"Open": [1.0, 2.0], "High": [1.5, 2.5], "Low": [0.9, 1.9],
         "Close": [1.2, 2.2], "Volume": [100, 200]},
        index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
    )
    raw.index.name = "Date"
    out = standardise(raw, "TST", "yfinance")
    assert list(out.columns) == STD_COLS
    assert len(out) == 2
    assert out["ticker"].unique().tolist() == ["TST"]
    assert str(out["date"].dtype).startswith("datetime64")


def teststandardise_empty():
    out = standardise(pd.DataFrame(), "TST", "yfinance")
    assert list(out.columns) == STD_COLS and out.empty


def _synthetic(ticker="TST", n=250, seed=0):
    dates = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame({
        "date": dates,
        "open": close * 0.99, "high": close * 1.01,
        "low": close * 0.98, "close": close,
        "volume": rng.integers(1e6, 5e6, n),
        "ticker": ticker, "source": "test",
    })


def test_check_ticker_clean():
    df = _synthetic()
    cal = build_reference_calendar({"TST": df})
    rec = check_ticker(df, cal, load_config())
    assert rec["status"] == "clean"
    assert rec["coverage"] == 1.0


def test_check_ticker_detects_halt_and_extremes():
    df = _synthetic()
    # Inject a halt-like row (flat OHLC, zero volume) and an extreme jump.
    df.loc[10, ["open", "high", "low", "close", "volume"]] = [50, 50, 50, 50, 0]
    df.loc[11, "close"] = df.loc[10, "close"] * 1.5   # +50% jump
    cal = build_reference_calendar({"TST": df})
    rec = check_ticker(df, cal, load_config())
    assert rec["n_halt_like"] >= 1
    assert rec["n_extreme_return"] >= 1
    assert "halt_like_rows" in rec["flags"]
    assert "extreme_returns" in rec["flags"]
