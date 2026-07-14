"""Data-integrity checks for the downloaded OHLCV panel.

Checks performed per ticker (against a reference NYSE trading calendar built
from the data itself, preferring the ^GSPC benchmark):

* **Coverage**         rows present / expected trading days in [first, last].
* **Internal gaps**    longest run of consecutive expected trading days that
                       are missing (delisting/long halt signature).
* **Duplicates**       repeated dates.
* **NaN / non-positive prices**  missing or <= 0 OHLC values.
* **OHLC consistency** high >= max(open,close,low), low <= min(...), high>=low.
* **Halt-like rows**   open==high==low==close with zero volume — the classic
                       "carried-forward price during a suspension" artifact.
* **Zero-volume days** possible halts / illiquidity.
* **Extreme returns**  |close-to-close return| > threshold (default 20%).

The result is a per-ticker table plus a JSON summary. Nothing is mutated —
this stage only *reports*; cleaning happens in week 1.2.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..utils import get_logger


def _iter_raw_files(cfg) -> list[Path]:
    return sorted(cfg.path("raw").glob("*.parquet"))


def build_reference_calendar(
    frames: dict[str, pd.DataFrame], benchmark: str = "^GSPC"
) -> pd.DatetimeIndex:
    """Reference trading days: benchmark dates if available, else the dates
    that appear for a majority (>=50%) of tickers."""
    if benchmark in frames and not frames[benchmark].empty:
        return pd.DatetimeIndex(sorted(frames[benchmark]["date"].unique()))

    counts: dict[pd.Timestamp, int] = {}
    for df in frames.values():
        for dte in df["date"].unique():
            counts[dte] = counts.get(dte, 0) + 1
    if not counts:
        return pd.DatetimeIndex([])
    n = len(frames)
    keep = [d for d, c in counts.items() if c >= 0.5 * n]
    return pd.DatetimeIndex(sorted(keep))


def check_ticker(df: pd.DataFrame, ref_cal: pd.DatetimeIndex, cfg) -> dict:
    ic = cfg["integrity"]
    rec: dict = {
        "ticker": df["ticker"].iloc[0] if len(df) else "?",
        "source": df["source"].iloc[0] if len(df) else "",
        "rows": int(len(df)),
    }
    if df.empty:
        rec.update(status="fail", flags=["empty"])
        return rec

    dates = pd.DatetimeIndex(df["date"])
    first, last = dates.min(), dates.max()
    rec["first_date"] = str(first.date())
    rec["last_date"] = str(last.date())

    # Expected trading days within the ticker's own listed window.
    exp = ref_cal[(ref_cal >= first) & (ref_cal <= last)]
    rec["expected_days"] = int(len(exp))
    present = set(dates)
    missing = [d for d in exp if d not in present]
    rec["missing_days"] = int(len(missing))
    rec["coverage"] = round(len(present) / len(exp), 4) if len(exp) else 0.0

    # Longest run of consecutive missing expected trading days.
    max_gap = 0
    run = 0
    for d in exp:
        if d not in present:
            run += 1
            max_gap = max(max_gap, run)
        else:
            run = 0
    rec["max_internal_gap"] = int(max_gap)

    rec["n_dupe_dates"] = int(df["date"].duplicated().sum())

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    v = df["volume"]
    rec["n_nan_close"] = int(c.isna().sum())
    rec["n_nan_ohlcv"] = int(df[["open", "high", "low", "close", "volume"]].isna().any(axis=1).sum())

    pos = pd.concat([o, h, l, c], axis=1)
    rec["n_nonpos_price"] = int((pos <= ic["min_price"]).any(axis=1).sum())

    # Use a small relative tolerance: split/dividend-adjusted prices carry
    # floating-point noise (~1e-14), so exact '<' comparisons produce spurious
    # violations. Only flag a bar when it breaks OHLC order by > tol.
    tol = (c.abs() * 1e-6).clip(lower=1e-6)
    ohlc_bad = (
        (l - h > tol)
        | (o - h > tol) | (c - h > tol)
        | (l - o > tol) | (l - c > tol)
    )
    rec["n_ohlc_violation"] = int(ohlc_bad.fillna(False).sum())

    rec["n_zero_volume"] = int((v.fillna(0) == 0).sum())
    halt_like = ((o == h) & (h == l) & (l == c) & (v.fillna(0) == 0)).fillna(False)
    rec["n_halt_like"] = int(halt_like.sum())

    # Fraction of rows that are REAL quotes (not flat carried-forward placeholders).
    # Coverage counts date presence; this exposes fabricated fill (SW ~41%, AMCR ~22%).
    rec["real_row_ratio"] = round(1.0 - rec["n_halt_like"] / max(len(df), 1), 4)
    real_dates = df.loc[~halt_like, "date"]
    rec["first_real_date"] = str(real_dates.min().date()) if len(real_dates) else None
    rec["real_rows"] = int((~halt_like).sum())

    ret = c.pct_change()
    rec["max_abs_return"] = round(float(ret.abs().max(skipna=True) or 0.0), 4)
    rec["n_extreme_return"] = int((ret.abs() > ic["max_daily_return"]).sum())

    # ---- classify ----
    flags: list[str] = []
    if rec["coverage"] < ic["min_coverage"]:
        flags.append("low_coverage")
    if rec["max_internal_gap"] > ic["max_gap_trading_days"]:
        flags.append("long_gap")
    if rec["n_dupe_dates"]:
        flags.append("dupe_dates")
    if rec["n_nonpos_price"]:
        flags.append("nonpos_price")
    if rec["n_ohlc_violation"]:
        flags.append("ohlc_violation")
    if rec["n_halt_like"]:
        flags.append("halt_like_rows")
    if rec["n_extreme_return"]:
        flags.append("extreme_returns")
    if rec["real_row_ratio"] < ic.get("min_real_row_ratio", 0.8):
        flags.append("fabricated_rows")      # mostly placeholder/flat data
    if rec["real_rows"] < ic.get("min_history_days", 60):
        flags.append("short_history")        # too little real data for indicators

    # Severity: "fail" is reserved for structural / widespread problems that
    # make the series unusable as-is. Isolated glitches (a handful of bars in
    # thousands) are "warn" — flagged for the week 1.2 cleaning step, not fatal.
    n = max(rec["rows"], 1)
    fatal = (
        rec["coverage"] < 0.5
        or rec["n_dupe_dates"] > 0
        or rec["n_nonpos_price"] > 0
        or rec["n_ohlc_violation"] > 0.005 * n         # >0.5% of bars inconsistent
        or rec["real_row_ratio"] < ic.get("min_real_row_ratio", 0.8)
        or rec["real_rows"] < ic.get("min_history_days", 60)
    )
    if fatal:
        rec["status"] = "fail"
    elif flags:
        rec["status"] = "warn"
    else:
        rec["status"] = "clean"
    rec["flags"] = flags
    return rec


def run_integrity(cfg) -> pd.DataFrame:
    log = get_logger("integrity", cfg.path("logs"))
    files = _iter_raw_files(cfg)
    if not files:
        raise RuntimeError(f"No raw parquet files in {cfg.path('raw')}")

    frames: dict[str, pd.DataFrame] = {}
    for f in files:
        df = pd.read_parquet(f)
        if len(df):
            frames[str(df["ticker"].iloc[0])] = df

    ref_cal = build_reference_calendar(frames)
    log.info("Loaded %d tickers | reference calendar: %d trading days (%s -> %s)",
             len(frames), len(ref_cal),
             ref_cal.min().date() if len(ref_cal) else "-",
             ref_cal.max().date() if len(ref_cal) else "-")

    records = [check_ticker(df, ref_cal, cfg) for df in frames.values()]
    report_df = pd.DataFrame(records).sort_values(
        ["status", "coverage"], ascending=[True, True]
    ).reset_index(drop=True)

    per_ticker_csv = cfg.path("metadata") / "integrity_per_ticker.csv"
    report_df.to_csv(per_ticker_csv, index=False)

    # ---- aggregate summary ----
    status_counts = report_df["status"].value_counts().to_dict()
    all_flags: dict[str, int] = {}
    for flist in report_df["flags"]:
        for fl in flist:
            all_flags[fl] = all_flags.get(fl, 0) + 1

    summary = {
        "generated_utc": pd.Timestamp.now("UTC").isoformat(),
        "n_tickers": int(len(report_df)),
        "reference_calendar": {
            "n_trading_days": int(len(ref_cal)),
            "start": str(ref_cal.min().date()) if len(ref_cal) else None,
            "end": str(ref_cal.max().date()) if len(ref_cal) else None,
        },
        "status_counts": {k: int(v) for k, v in status_counts.items()},
        "flag_counts": all_flags,
        "thresholds": dict(cfg["integrity"]),
        "total_rows": int(report_df["rows"].sum()),
        "tickers_failed": report_df.loc[report_df.status == "fail", "ticker"].tolist(),
        "tickers_warn": report_df.loc[report_df.status == "warn", "ticker"].tolist(),
        "per_ticker_csv": str(per_ticker_csv.relative_to(_root(cfg))),
    }
    report_path = cfg.resolve(cfg["integrity"]["report"])
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    log.info("Integrity: %s", summary["status_counts"])
    log.info("Flags: %s", all_flags)
    log.info("Report -> %s | per-ticker -> %s", report_path, per_ticker_csv)
    return report_df


def _root(cfg) -> Path:
    from ..config import ROOT
    return ROOT
