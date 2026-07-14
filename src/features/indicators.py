"""Technical indicators (MA, EMA, MACD, RSI, Bollinger, ATR).

All indicators are computed on the **adjusted** close (``adj_close``) so that
splits/dividends don't create artificial jumps. Each function takes a price
Series (time-ordered) and returns aligned Series/DataFrame; ``add_indicators``
attaches the standard set to one ticker's frame, and ``compute_indicators``
applies it across the whole panel, per ticker.

Parameter choices follow the common conventions (and are configurable):
    MA        : 5, 10, 20, 50, 200
    EMA/MACD  : 12, 26, signal 9
    RSI       : 14 (Wilder's smoothing)
    Bollinger : 20-day, 2 sigma
    ATR       : 14
"""
from __future__ import annotations

import pandas as pd

MA_WINDOWS = (5, 10, 20, 50, 200)
BB_WINDOW = 20
BB_K = 2.0
RSI_WINDOW = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_WINDOW = 14


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(s: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """Wilder's RSI (EWM with alpha=1/window, adjust=False)."""
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    # If avg_loss == 0 over the window, RSI is 100 (pure gains).
    out = out.where(avg_loss != 0, 100.0)
    return out


def macd(s: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW,
         signal: int = MACD_SIGNAL) -> pd.DataFrame:
    macd_line = ema(s, fast) - ema(s, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def bollinger(s: pd.Series, window: int = BB_WINDOW, k: float = BB_K) -> pd.DataFrame:
    mid = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    width = (upper - lower) / mid
    pctb = (s - lower) / (upper - lower)
    return pd.DataFrame({
        "bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
        "bb_width": width, "bb_pctb": pctb,
    })


def atr(df: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    """Average True Range on raw OHLC (Wilder smoothing)."""
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the standard indicator set to one ticker's (sorted) frame."""
    df = df.sort_values("date").copy()
    px = df["adj_close"]

    for w in MA_WINDOWS:
        df[f"ma{w}"] = sma(px, w)
    df["ema12"] = ema(px, MACD_FAST)
    df["ema26"] = ema(px, MACD_SLOW)

    df = pd.concat([df, macd(px)], axis=1)
    df["rsi14"] = rsi(px, RSI_WINDOW)
    df = pd.concat([df, bollinger(px)], axis=1)
    df["atr14"] = atr(df, ATR_WINDOW)

    # A few normalised, cross-sectionally comparable signals for analysis.
    df["ma20_gap"] = px / df["ma20"] - 1.0          # distance from MA20
    df["mom_20"] = px / px.shift(20) - 1.0          # 20-day momentum
    df["macd_hist_norm"] = df["macd_hist"] / px     # scale-free MACD histogram
    df["vol_ma20"] = df["volume"].astype("float64").rolling(20, min_periods=20).mean()
    return df


def compute_indicators(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply ``add_indicators`` to every ticker in the consolidated panel."""
    frames = [add_indicators(g) for _, g in panel.groupby("ticker", sort=False)]
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)
