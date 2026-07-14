"""Week 1.2 — data cleaning (missing values, outliers, standardisation).

Design philosophy (informed by the Week-1 review, reports/week1_data_review.md):

* **``data/raw/`` is immutable.** Cleaning reads raw, writes cleaned per-ticker
  files to ``data/interim/clean/`` and a consolidated panel to
  ``data/processed/``. Every value changed or flagged is recorded in an
  auditable adjustments log.
* **Flag, don't silently mutate.** A >20% daily move is almost always a *real*
  event (earnings, biotech, bankruptcy, COVID crash) — verified median annual
  extreme ≈ 0.19 across the universe. We flag/winsorise *return features*; we do
  **not** overwrite real price moves. Only mechanical errors (OHLC-order
  violations, non-positive prices) are corrected, and each is logged.
* **Missing values are structural, not per-cell.** Raw has 0 NaN cells; the real
  problems are (a) fabricated leading placeholder rows (SW ~41%, AMCR ~22%) and
  (b) short histories. We truncate each series to its first *sustained* real
  trading day, mask interior non-trading (halt) bars as NaN, and forward-fill
  only short gaps (≤ N days) with an ``is_filled`` flag — never across a
  delisting or before the first real bar (avoids look-ahead / fabricated quotes).

The three spec sub-tasks map to:
    1. 缺失值 (missing)      -> _trim_leading_placeholder, _align_calendar
    2. 异常值 (outliers)     -> _fix_mechanical (correct), _flag_outliers (flag+winsorise)
    3. 标准化 (standardise)  -> _standardise_types
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..storage import ParquetStore
from ..utils import get_logger

# Columns carried through from raw (see download._STD_COLS).
_PRICE_COLS = ["open", "high", "low", "close", "adj_close"]


@dataclass
class CleanStats:
    ticker: str
    raw_rows: int = 0
    clean_rows: int = 0
    leading_trimmed: int = 0
    halt_masked: int = 0
    gap_filled: int = 0
    ohlc_fixed: int = 0
    nonpos_fixed: int = 0
    n_extreme: int = 0
    n_suspect: int = 0
    first_real_date: str = ""
    dropped: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        return self.__dict__


# --------------------------------------------------------------------------
# Reference NYSE calendar (authoritative, independent of the data)
# --------------------------------------------------------------------------
def nyse_sessions(start: str, end: str) -> pd.DatetimeIndex:
    try:
        import pandas_market_calendars as mcal
        sched = mcal.get_calendar("XNYS").schedule(start_date=start, end_date=end)
        return pd.DatetimeIndex(sched.index).tz_localize(None).normalize()
    except Exception:
        return pd.DatetimeIndex([])  # caller falls back to per-ticker dates


# --------------------------------------------------------------------------
# Step 1 — missing values / structural
# --------------------------------------------------------------------------
def _is_placeholder(df: pd.DataFrame) -> pd.Series:
    """Flat carried-forward bar with no volume — a non-trading placeholder."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    v = df["volume"].fillna(0)
    return ((o == h) & (h == l) & (l == c) & (v == 0)).fillna(False)


def _first_sustained_date(df: pd.DataFrame, window: int = 21):
    """First date whose forward `window` bars are all real trades — i.e. where
    genuine trading begins (skips thinly-traded pre-listing placeholder blocks)."""
    real = (~_is_placeholder(df)) & (df["volume"].fillna(0) > 0)
    real = real.to_numpy()
    n = len(real)
    if real.all():
        return df["date"].iloc[0]
    for i in range(n):
        if real[i] and real[i : min(i + window, n)].all():
            return df["date"].iloc[i]
    # No sustained window found -> first real bar, if any.
    idx = np.flatnonzero(real)
    return df["date"].iloc[idx[0]] if len(idx) else None


def _trim_leading_placeholder(df: pd.DataFrame, stats: CleanStats,
                              window: int = 21) -> pd.DataFrame:
    start = _first_sustained_date(df, window)
    if start is None:
        stats.dropped = True
        stats.reason = "no sustained real trading window"
        return df.iloc[0:0]
    before = len(df)
    df = df[df["date"] >= start].reset_index(drop=True)
    stats.leading_trimmed = before - len(df)
    stats.first_real_date = str(pd.Timestamp(start).date())
    return df


def _align_calendar(df: pd.DataFrame, ref_cal: pd.DatetimeIndex,
                    stats: CleanStats, max_ffill: int) -> pd.DataFrame:
    """Reindex onto the NYSE calendar within [first,last]; mask interior halts as
    NaN; forward-fill only short gaps (<= max_ffill), flagging filled rows."""
    first, last = df["date"].min(), df["date"].max()
    if len(ref_cal):
        cal = ref_cal[(ref_cal >= first) & (ref_cal <= last)]
    else:
        cal = pd.DatetimeIndex(sorted(df["date"].unique()))

    df = df.set_index("date")
    # Interior non-trading placeholder bars -> NaN prices (not tradable quotes).
    halt = _is_placeholder(df.reset_index()).to_numpy()
    stats.halt_masked = int(halt.sum())
    df.loc[halt, _PRICE_COLS] = np.nan
    df.loc[halt, "volume"] = pd.NA

    df = df.reindex(cal)                       # inserts NaN rows for missing days
    df.index.name = "date"
    df["is_halt"] = df["adj_close"].isna()     # any non-observed session

    # Constrained forward-fill of prices for short gaps only.
    fill_cols = _PRICE_COLS
    before_na = df[fill_cols].isna().any(axis=1)
    df[fill_cols] = df[fill_cols].ffill(limit=max_ffill)
    df["is_filled"] = before_na & df[fill_cols].notna().all(axis=1)
    stats.gap_filled = int(df["is_filled"].sum())

    # Volume on filled/halt rows is 0 (no trading occurred).
    df["volume"] = df["volume"].fillna(0).astype("Int64")
    for col in ("dividends", "splits"):
        df[col] = df[col].fillna(0.0)
    df["ticker"] = df["ticker"].ffill().bfill()
    df["source"] = df["source"].ffill().bfill()
    return df.reset_index()


# --------------------------------------------------------------------------
# Step 2 — outliers
# --------------------------------------------------------------------------
def _fix_mechanical(df: pd.DataFrame, stats: CleanStats, log_rows: list,
                    min_price: float) -> pd.DataFrame:
    """Correct unambiguous mechanical errors (OHLC order, non-positive price).
    These are the only value edits cleaning makes to prices — each is logged."""
    o, h, l, c = (df[x] for x in ["open", "high", "low", "close"])
    tol = (c.abs() * 1e-6).clip(lower=1e-6)
    bad = ((l - h > tol) | (o - h > tol) | (c - h > tol) |
           (l - o > tol) | (l - c > tol)).fillna(False)
    for i in df.index[bad]:
        row = df.loc[i, ["open", "high", "low", "close"]]
        new_h, new_l = float(row.max()), float(row.min())
        if abs(new_h - df.at[i, "high"]) > 0 or abs(new_l - df.at[i, "low"]) > 0:
            log_rows.append(dict(ticker=stats.ticker, date=str(df.at[i, "date"].date()),
                                 field="high/low", old=f"{df.at[i,'high']:.4f}/{df.at[i,'low']:.4f}",
                                 new=f"{new_h:.4f}/{new_l:.4f}", reason="ohlc_order_fix"))
            df.at[i, "high"], df.at[i, "low"] = new_h, new_l
            stats.ohlc_fixed += 1

    nonpos = (df[_PRICE_COLS] <= min_price).any(axis=1) & df["adj_close"].notna()
    stats.nonpos_fixed = int(nonpos.sum())
    for i in df.index[nonpos]:
        log_rows.append(dict(ticker=stats.ticker, date=str(df.at[i, "date"].date()),
                             field="price", old="<=min", new="NaN", reason="nonpositive_price"))
        df.loc[i, _PRICE_COLS] = np.nan
    return df


def _flag_outliers(df: pd.DataFrame, stats: CleanStats, max_ret: float) -> pd.DataFrame:
    """Compute returns and FLAG extremes (never overwrite prices). Adds a
    separately winsorised return feature for downstream modelling."""
    # Returns on ADJUSTED close; no return on filled/halt rows (avoid fake 0s).
    ret = df["adj_close"].pct_change(fill_method=None)
    ret[df["is_filled"] | df["is_halt"]] = np.nan
    df["ret"] = ret
    df["log_ret"] = np.log1p(ret)

    df["is_extreme"] = ret.abs() > max_ret
    # "Suspect" = extreme AND next-day reverses >50% (possible bad tick, still
    # only flagged — most survive as real crash-period volatility).
    nxt = ret.shift(-1)
    reversal = (np.sign(nxt) == -np.sign(ret)) & (nxt.abs() > 0.5 * ret.abs())
    df["is_suspect"] = (df["is_extreme"] & reversal).fillna(False)
    stats.n_extreme = int(df["is_extreme"].fillna(False).sum())
    stats.n_suspect = int(df["is_suspect"].sum())

    # Winsorise the return FEATURE at median +/- 3 robust-MAD (separate column).
    r = df["ret"].dropna()
    if len(r) > 20:
        med = r.median()
        mad = (r - med).abs().median() * 1.4826
        lo, hi = med - 3 * mad, med + 3 * mad
        df["ret_winsor"] = df["ret"].clip(lo, hi)
    else:
        df["ret_winsor"] = df["ret"]
    return df


# --------------------------------------------------------------------------
# Step 3 — standardisation
# --------------------------------------------------------------------------
_CLEAN_SCHEMA = [
    "date", "ticker", "open", "high", "low", "close", "adj_close",
    "volume", "dividends", "splits", "ret", "log_ret", "ret_winsor",
    "is_halt", "is_filled", "is_extreme", "is_suspect", "source",
]


def _standardise_types(df: pd.DataFrame) -> pd.DataFrame:
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    for c in _PRICE_COLS + ["dividends", "splits", "ret", "log_ret", "ret_winsor"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["volume"] = df["volume"].astype("Int64")
    for c in ["is_halt", "is_filled", "is_extreme", "is_suspect"]:
        df[c] = df[c].fillna(False).astype(bool)
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    return df[_CLEAN_SCHEMA].reset_index(drop=True)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def clean_ticker(df: pd.DataFrame, ref_cal: pd.DatetimeIndex, cfg,
                 log_rows: list) -> tuple[pd.DataFrame, CleanStats]:
    ic = cfg["integrity"]
    st = CleanStats(ticker=str(df["ticker"].iloc[0]) if len(df) else "?",
                    raw_rows=len(df))
    min_hist = int(ic.get("min_history_days", 60))
    max_ffill = int(cfg.get("cleaning", {}).get("max_ffill_days", 5))

    df = _trim_leading_placeholder(df, st)
    if st.dropped or len(df) < min_hist:
        st.dropped = True
        st.reason = st.reason or f"real history < {min_hist} days"
        return df.iloc[0:0], st

    df = _align_calendar(df, ref_cal, st, max_ffill)
    df = _fix_mechanical(df, st, log_rows, ic["min_price"])
    df = _flag_outliers(df, st, ic["max_daily_return"])
    df = _standardise_types(df)
    st.clean_rows = len(df)
    return df, st


def clean_all(cfg) -> pd.DataFrame:
    """Clean every raw ticker, write per-ticker + consolidated outputs, and a
    cleaning report. Returns the per-ticker stats table."""
    import json

    log = get_logger("clean", cfg.path("logs"))
    comp = cfg["storage"]["compression"]
    raw_store = ParquetStore(cfg.path("raw"))
    clean_store = ParquetStore(cfg.path("interim") / "clean", comp)
    processed = ParquetStore(cfg.path("processed"), comp)
    raw_files = raw_store.files()
    if not raw_files:
        raise RuntimeError(f"No raw parquet files in {raw_store.root}")

    ref_cal = nyse_sessions(cfg.start_date, cfg.end_date)
    log.info("Cleaning %d tickers | NYSE calendar: %d sessions%s",
             len(raw_files), len(ref_cal),
             "" if len(ref_cal) else " (unavailable -> per-ticker dates)")

    log_rows: list[dict] = []
    stats_rows: list[dict] = []
    kept, dropped = [], []

    for raw in raw_store.iter_frames():
        cleaned, st = clean_ticker(raw, ref_cal, cfg, log_rows)
        stats_rows.append(st.as_dict())
        if st.dropped or cleaned.empty:
            dropped.append(st.ticker)
            continue
        clean_store.write(cleaned, st.ticker)
        kept.append(cleaned)

    # Consolidated cleaned panel + adjusted-close wide matrix.
    panel = pd.concat(kept, ignore_index=True).sort_values(["ticker", "date"])
    panel_path = processed.write(panel, "sp500_clean_panel")
    wide = panel.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
    processed.write(wide, "adj_close_clean", index=True)

    stats = pd.DataFrame(stats_rows)
    stats.to_csv(cfg.path("metadata") / "cleaning_stats.csv", index=False)
    adj_log = pd.DataFrame(log_rows)
    adj_log.to_csv(cfg.path("metadata") / "cleaning_adjustments.csv", index=False)

    report = {
        "generated_utc": pd.Timestamp.now("UTC").isoformat(),
        "tickers_in": len(raw_files),
        "tickers_kept": len(kept),
        "tickers_dropped": dropped,
        "rows_raw": int(stats["raw_rows"].sum()),
        "rows_clean": int(panel.shape[0]),
        "leading_placeholder_rows_trimmed": int(stats["leading_trimmed"].sum()),
        "interior_halt_rows_masked": int(stats["halt_masked"].sum()),
        "gap_rows_forward_filled": int(stats["gap_filled"].sum()),
        "ohlc_order_fixes": int(stats["ohlc_fixed"].sum()),
        "nonpositive_price_fixes": int(stats["nonpos_fixed"].sum()),
        "extreme_moves_flagged": int(stats["n_extreme"].sum()),
        "suspect_reversal_flagged": int(stats["n_suspect"].sum()),
        "value_edits_logged": int(len(adj_log)),
        "outputs": {
            "clean_panel": str(panel_path.relative_to(_root(cfg))),
            "wide_adj_close": "data/processed/adj_close_clean.parquet",
            "per_ticker_dir": "data/interim/clean/",
            "adjustments_log": "data/metadata/cleaning_adjustments.csv",
            "stats": "data/metadata/cleaning_stats.csv",
        },
    }
    with open(cfg.path("metadata") / "cleaning_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    log.info("Cleaned: kept=%d dropped=%d | rows %d -> %d | trimmed=%d halts=%d "
             "filled=%d ohlc_fix=%d extreme_flag=%d",
             len(kept), len(dropped), report["rows_raw"], report["rows_clean"],
             report["leading_placeholder_rows_trimmed"], report["interior_halt_rows_masked"],
             report["gap_rows_forward_filled"], report["ohlc_order_fixes"],
             report["extreme_moves_flagged"])
    if dropped:
        log.info("Dropped (too short / no real history): %s", ", ".join(dropped))
    return stats


def _root(cfg):
    from ..config import ROOT
    return ROOT
