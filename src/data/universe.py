"""S&P 500 universe: fetch, normalise and cache the constituent list.

Primary source is the Wikipedia "List of S&P 500 companies" table. The
result is cached to ``data/metadata/sp500_constituents.csv`` so the rest of
the pipeline is reproducible offline.

Survivorship-bias caveat
-------------------------
Wikipedia lists the *current* constituents. Backtesting a 10-year window on
today's members over-represents survivors (companies that were dropped are
missing; companies added recently didn't exist in the index for most of the
window). For research this is acceptable if acknowledged; a fully unbiased
study needs a point-in-time membership dataset (e.g. CRSP / paid vendor).
"""
from __future__ import annotations

import io

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def to_yahoo_symbol(symbol: str) -> str:
    """Wikipedia uses e.g. 'BRK.B'; Yahoo Finance expects 'BRK-B'."""
    return symbol.strip().upper().replace(".", "-")


def _fetch_from_wikipedia() -> pd.DataFrame:
    resp = requests.get(WIKI_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    # First table (id="constituents") holds the current members.
    tables = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})
    df = tables[0]

    # Normalise column names to snake_case.
    rename = {
        "Symbol": "symbol",
        "Security": "security",
        "GICS Sector": "gics_sector",
        "GICS Sub-Industry": "gics_sub_industry",
        "Headquarters Location": "headquarters",
        "Date added": "date_added",
        "CIK": "cik",
        "Founded": "founded",
    }
    df = df.rename(columns=rename)
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["yahoo_symbol"] = df["symbol"].map(to_yahoo_symbol)
    keep = [c for c in rename.values() if c in df.columns] + ["yahoo_symbol"]
    return df[keep].drop_duplicates(subset="symbol").reset_index(drop=True)


def get_sp500_constituents(cfg, force_refresh: bool = False) -> pd.DataFrame:
    """Return the S&P 500 constituent table, using the cache when possible."""
    cache = cfg.resolve(cfg["universe"]["cache"])

    if cache.exists() and not force_refresh:
        return pd.read_csv(cache)

    try:
        df = _fetch_from_wikipedia()
    except Exception as exc:  # network / parse failure -> fall back to cache
        if cache.exists():
            return pd.read_csv(cache)
        raise RuntimeError(
            f"Failed to fetch S&P 500 constituents and no cache at {cache}: {exc}"
        ) from exc

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def get_ticker_list(cfg, force_refresh: bool = False) -> list[str]:
    """Yahoo-formatted ticker list including any configured extra tickers."""
    df = get_sp500_constituents(cfg, force_refresh=force_refresh)
    tickers = df["yahoo_symbol"].tolist()
    tickers += list(cfg["universe"].get("extra_tickers", []))
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    out: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
