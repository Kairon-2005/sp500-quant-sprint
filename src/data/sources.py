"""Price data sources (Strategy pattern).

A ``PriceSource`` fetches OHLCV for a set of tickers and returns each one in the
canonical schema below. The downloader tries a primary source and falls back to
others per-ticker, so adding a new vendor is just another subclass.

Canonical per-ticker schema (long, one row per trading day):
    date, open, high, low, close, adj_close, volume, dividends, splits,
    ticker, source
open/high/low/close are RAW (unadjusted); adj_close is split+dividend adjusted.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

STD_COLS = [
    "date", "open", "high", "low", "close", "adj_close",
    "volume", "dividends", "splits", "ticker", "source",
]
_PRICE_INPUT = ["open", "high", "low", "close", "volume"]


def standardise(df: pd.DataFrame, ticker: str, source: str) -> pd.DataFrame:
    """Coerce any source frame into ``STD_COLS``; drop empty rows.

    Missing adjusted/action columns default sensibly (adj_close := close for
    sources like Stooq that only provide raw prices).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=STD_COLS)

    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df = df.reset_index()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    date_col = "date" if "date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "date", "stock_splits": "splits"})
    if [c for c in _PRICE_INPUT if c not in df.columns]:
        return pd.DataFrame(columns=STD_COLS)

    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    for opt, default in (("dividends", 0.0), ("splits", 0.0)):
        df[opt] = df.get(opt, default)

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    for c in ["open", "high", "low", "close", "adj_close", "dividends", "splits"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").round().astype("Int64")

    df["ticker"], df["source"] = ticker, source
    df = df[STD_COLS].dropna(subset=["open", "high", "low", "close"], how="all")
    return df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


class PriceSource(ABC):
    name: str

    @abstractmethod
    def fetch(self, tickers: list[str], start: str, end: str, *,
              interval: str, auto_adjust: bool) -> dict[str, pd.DataFrame]:
        """Return {ticker: standardised frame}; an empty frame means 'unavailable'."""


class YahooSource(PriceSource):
    """Yahoo Finance via yfinance (batched, splits+dividends retained)."""
    name = "yfinance"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def fetch(self, tickers, start, end, *, interval, auto_adjust):
        import yfinance as yf
        end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
        raw = yf.download(
            tickers=tickers, start=start, end=end_excl, interval=interval,
            auto_adjust=auto_adjust, actions=True, group_by="ticker",
            threads=True, progress=False, timeout=self.timeout,
        )
        out: dict[str, pd.DataFrame] = {}
        multi = isinstance(raw.columns, pd.MultiIndex)
        lvl0 = set(raw.columns.get_level_values(0)) if multi else set()
        for t in tickers:
            sub = raw[t] if (multi and t in lvl0) else (None if multi else raw)
            out[t] = standardise(sub, t, self.name)
        return out


class StooqSource(PriceSource):
    """Stooq fallback via its direct CSV endpoint (unadjusted EOD; adj_close :=
    close). pandas_datareader dropped Stooq support, so we hit the CSV API."""
    name = "stooq"
    BASE = "https://stooq.com/q/d/l/"

    def fetch(self, tickers, start, end, *, interval, auto_adjust):
        import io
        import time

        import requests
        d1, d2 = pd.Timestamp(start).strftime("%Y%m%d"), pd.Timestamp(end).strftime("%Y%m%d")
        out: dict[str, pd.DataFrame] = {}
        for t in tickers:
            frame = pd.DataFrame(columns=STD_COLS)
            for sym in (f"{t}.US", f"{t.replace('-', '.')}.US"):
                try:
                    url = f"{self.BASE}?s={sym.lower()}&d1={d1}&d2={d2}&i=d"
                    r = requests.get(url, timeout=30, headers=_HEADERS)
                    txt = r.text.strip()
                    if r.ok and txt and not txt.startswith("<") and "Close" in txt[:200]:
                        df = pd.read_csv(io.StringIO(txt))
                        if not df.empty and "Close" in df.columns:
                            frame = standardise(df, t, self.name)
                            break
                except Exception:
                    continue
            out[t] = frame
            time.sleep(0.15)  # be polite to the free endpoint
        return out


_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


class TiingoSource(PriceSource):
    """Independent EOD source via Tiingo's free API. Reads the token from the
    ``TIINGO_API_KEY`` env var (or the constructor); returns empty frames if no
    key is set, so reconciliation degrades gracefully. Tiingo returns both raw
    and adjusted fields, so raw-close reconciliation is apples-to-apples.

    NOTE: written to Tiingo's documented endpoint but not exercised in this
    environment (no key here); validate with a real token.
    """
    name = "tiingo"
    BASE = "https://api.tiingo.com/tiingo/daily"

    def __init__(self, api_key: str | None = None):
        import os
        self.api_key = api_key or os.environ.get("TIINGO_API_KEY")

    def fetch(self, tickers, start, end, *, interval, auto_adjust):
        import requests
        out: dict[str, pd.DataFrame] = {}
        for t in tickers:
            frame = pd.DataFrame(columns=STD_COLS)
            if self.api_key:
                try:
                    r = requests.get(
                        f"{self.BASE}/{t.lower()}/prices",
                        params={"startDate": str(start), "endDate": str(end),
                                "token": self.api_key, "format": "json"},
                        timeout=30)
                    rows = r.json() if r.ok else []
                    if isinstance(rows, list) and rows:
                        df = pd.DataFrame(rows).rename(columns={"adjClose": "adj_close"})
                        frame = standardise(df, t, self.name)
                except Exception:
                    pass
            out[t] = frame
        return out


def alt_source() -> PriceSource:
    """The reconciliation source: Tiingo if a key is configured, else Stooq."""
    tiingo = TiingoSource()
    return tiingo if tiingo.api_key else StooqSource()
