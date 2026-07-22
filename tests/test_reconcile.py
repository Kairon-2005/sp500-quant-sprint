"""Offline tests for the cross-source reconciliation engine (no network)."""
import numpy as np
import pandas as pd

from src.data.reconcile import compare_frames


def _frame(close, start="2022-01-03"):
    dates = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame({"date": dates, "close": close})


def _walk(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return 100 * np.cumprod(1 + rng.normal(0, 0.01, n))


def test_identical_sources_ok():
    px = _walk()
    rec = compare_frames("TST", _frame(px), _frame(px.copy()))
    assert rec["status"] == "ok"
    assert rec["return_corr"] > 0.99 and rec["median_abs_pct"] == 0.0


def test_scattered_bad_ticks_warn():
    px = _walk()
    alt = px.copy()
    alt[::15] *= 1.05                       # ~7% of days off by 5%
    rec = compare_frames("TST", _frame(px), _frame(alt))
    assert rec["status"] == "warn"
    assert rec["n_days_gt_2pct"] >= 1


def test_uncorrelated_series_mismatch():
    # Two independent walks -> low return correlation -> ticker-reuse signature.
    rec = compare_frames("TST", _frame(_walk(seed=1)), _frame(_walk(seed=2)))
    assert rec["status"] == "mismatch"
    assert rec["return_corr"] < 0.9


def test_no_overlap_reports_insufficient():
    a = _frame(_walk(50), start="2022-01-03")
    b = _frame(_walk(50), start="2024-01-03")   # disjoint dates
    rec = compare_frames("TST", a, b)
    assert rec["status"] in ("insufficient_overlap", "no_alt_source")
