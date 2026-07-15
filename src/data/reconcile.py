"""Cross-source reconciliation — catch erroneous quotes by comparing vendors.

A single vendor can return bad bars (a wrong tick, a stale carry-forward, or a
ticker that now points at a different company — e.g. SATS -> ECHO). Having a
*fallback* source is not the same as *validating* against it. This module pulls
the same ticker from the alternate source (Stooq) and compares it, day by day,
against the primary (Yahoo) data already on disk.

Comparison is on the **raw** close (both vendors unadjusted -> apples to apples):
* ``return_corr``      — correlation of daily returns; low => likely a different
                         security or badly misaligned data (ticker reuse).
* ``median_abs_pct``   — typical |price disagreement| across overlapping days.
* ``n_days_gt_2pct``   — count of days the two sources disagree by >2% (candidate
                         bad bars in one source), with the worst day recorded.
"""
from __future__ import annotations

import pandas as pd

from ..storage import ParquetStore
from ..utils import get_logger
from .sources import alt_source


def compare_frames(ticker: str, yahoo: pd.DataFrame, alt: pd.DataFrame,
                   cfg) -> dict:
    """Pure comparison of two OHLCV frames on their overlapping raw closes.
    Network-free, so unit-testable in isolation."""
    rec = {"ticker": ticker, "overlap_days": 0, "status": "no_alt_source"}
    if alt is None or alt.empty:
        return rec

    m = yahoo[["date", "close"]].merge(
        alt[["date", "close"]], on="date", suffixes=("_y", "_s")).dropna()
    if len(m) < 30:
        rec.update(overlap_days=len(m), status="insufficient_overlap")
        return rec

    pct = (m["close_y"] / m["close_s"] - 1.0).abs()
    ry = m["close_y"].pct_change()
    rs = m["close_s"].pct_change()
    corr = float(ry.corr(rs))
    worst = m.loc[pct.idxmax()]

    rec.update(
        overlap_days=int(len(m)),
        return_corr=round(corr, 4),
        median_abs_pct=round(float(pct.median()), 5),
        max_abs_pct=round(float(pct.max()), 4),
        n_days_gt_2pct=int((pct > 0.02).sum()),
        worst_date=str(pd.Timestamp(worst["date"]).date()),
    )
    # Classify. Persistent price disagreement (high median) => the two series are
    # different securities (ticker reuse / wrong data). Sparse big-diff days or a
    # merely reduced correlation => inspect. Otherwise the sources agree.
    if rec["median_abs_pct"] > 0.05:
        rec["status"] = "mismatch"        # likely ticker reuse / wrong series
    elif corr < 0.9 or rec["n_days_gt_2pct"] > 0.02 * len(m) or rec["median_abs_pct"] > 0.01:
        rec["status"] = "warn"            # some disagreement -> inspect
    else:
        rec["status"] = "ok"
    return rec


def reconcile_ticker(ticker: str, yahoo: pd.DataFrame, cfg, source) -> dict:
    """Fetch the alternate source for one ticker and compare it to `yahoo`."""
    alt = source.fetch([ticker], cfg.start_date, cfg.end_date,
                        interval=cfg["data"]["interval"], auto_adjust=False).get(ticker)
    return compare_frames(ticker, yahoo, alt, cfg)


def reconcile(cfg, tickers: list[str] | None = None, sample: int | None = 60) -> pd.DataFrame:
    """Reconcile a sample (default) or an explicit ticker list against Stooq."""
    log = get_logger("reconcile", cfg.path("logs"))
    store = ParquetStore(cfg.path("raw"))
    available = [p.stem for p in store.files()]
    if tickers is None:
        tickers = available
        if sample and sample < len(tickers):
            # Deterministic evenly-spaced sample (no RNG needed for reproducibility).
            step = len(tickers) / sample
            tickers = [tickers[int(i * step)] for i in range(sample)]

    src = alt_source()
    log.info("Reconciliation source: %s", src.name)
    rows = []
    for i, stem in enumerate(tickers, 1):
        yahoo = pd.read_parquet(store.path(stem))
        tkr = str(yahoo["ticker"].iloc[0]) if len(yahoo) else stem
        rows.append(reconcile_ticker(tkr, yahoo, cfg, src))
        if i % 20 == 0:
            log.info("  reconciled %d/%d", i, len(tickers))

    df = pd.DataFrame(rows)
    out = cfg.path("metadata") / "reconciliation.csv"
    df.to_csv(out, index=False)
    counts = df["status"].value_counts().to_dict()
    log.info("Reconciliation: %s -> %s", counts, out)
    return df
