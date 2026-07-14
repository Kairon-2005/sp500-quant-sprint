"""Offline unit tests for technical indicators & signal helpers."""
import numpy as np
import pandas as pd

from src.features.indicators import add_indicators, bollinger, ema, macd, rsi, sma
from src.features.signals import add_forward_returns


def _series(n=300, seed=3):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)))


def test_sma_matches_manual():
    s = _series(50)
    assert np.isclose(sma(s, 10).iloc[20], s.iloc[11:21].mean())


def test_rsi_bounds_and_monotone():
    # Strictly rising series -> RSI should pin near 100.
    up = pd.Series(np.arange(1, 200, dtype=float))
    r = rsi(up, 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()
    assert r.iloc[-1] > 99


def test_macd_definition():
    s = _series()
    m = macd(s)
    assert np.allclose((ema(s, 12) - ema(s, 26)).dropna(),
                       m["macd"].dropna(), equal_nan=True)
    assert np.allclose((m["macd"] - m["macd_signal"]).dropna(),
                       m["macd_hist"].dropna())


def test_bollinger_ordering():
    s = _series()
    b = bollinger(s).dropna()
    assert (b["bb_upper"] >= b["bb_mid"]).all()
    assert (b["bb_mid"] >= b["bb_lower"]).all()
    # %B == 1 at the upper band, 0 at the lower band.
    mid_pctb = ((b["bb_mid"] - b["bb_lower"]) / (b["bb_upper"] - b["bb_lower"]))
    assert np.allclose(mid_pctb, 0.5, atol=1e-6)


def test_add_indicators_and_forward_returns():
    df = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=300),
        "ticker": "TST",
        "open": _series().values, "high": _series(seed=4).values,
        "low": _series(seed=5).values, "close": _series().values,
        "adj_close": _series().values, "volume": 1_000_000,
    })
    out = add_indicators(df)
    for col in ["ma20", "rsi14", "macd", "bb_pctb", "atr14", "mom_20"]:
        assert col in out.columns
    fwd = add_forward_returns(out, horizons=(1, 5))
    # fwd_ret_1 at row i equals adj_close[i+1]/adj_close[i]-1.
    px = fwd["adj_close"].to_numpy()
    assert np.isclose(fwd["fwd_ret_1"].iloc[0], px[1] / px[0] - 1)
    # last row has no future -> NaN.
    assert pd.isna(fwd["fwd_ret_1"].iloc[-1])
