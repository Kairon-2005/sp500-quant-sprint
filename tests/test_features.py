"""Offline tests for Week-2 feature engineering (no network)."""
import numpy as np
import pandas as pd

from src.features.dataset import attach_labels, purged_split
from src.features.lags import add_lag_features, feature_columns
from src.features.macro import add_macro, derive_macro_features, macro_columns
from src.features.reduce import correlation_prune, pca_fit
from src.features.signals import FWD_HORIZONS


def _ticker(n=400, seed=1, name="TST"):
    dates = pd.bdate_range("2021-01-04", periods=n)
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.012, n))
    return pd.DataFrame({
        "date": dates, "ticker": name,
        "open": close * 0.997, "high": close * 1.012, "low": close * 0.988,
        "close": close, "adj_close": close,
        "volume": rng.integers(1e6, 5e6, n).astype("float64"),
    })


def test_lag_features_are_point_in_time():
    """The definitive no-look-ahead check: a feature at row k must be identical
    whether computed on the full series or only rows [0..k]."""
    df = _ticker()
    full = add_lag_features(df)
    k = 320
    partial = add_lag_features(df.iloc[: k + 1])
    for col in ["roc_5", "vol_20", "rsv_20", "kmid", "corr_pv_20", "ma_gap_20", "cntp_20"]:
        assert np.isclose(full[col].iloc[k], partial[col].iloc[k], equal_nan=True), col


def test_roc_formula():
    df = add_lag_features(_ticker())
    px = df["adj_close"].to_numpy()
    assert np.isclose(df["roc_5"].iloc[100], px[100] / px[95] - 1)


def test_feature_columns_match_builder():
    """feature_columns() is derived from the builder, so every listed feature
    must exist in add_lag_features output (and drift is impossible)."""
    out = add_lag_features(_ticker(n=300))
    missing = [c for c in feature_columns() if c not in out.columns]
    assert missing == []


def test_labels_are_na_not_zero_when_future_missing():
    """Regression: NaN forward returns must yield NA labels, never a fake 0."""
    n = 50
    feat = pd.DataFrame({f"fwd_ret_{h}": np.random.default_rng(0).normal(size=n)
                         for h in FWD_HORIZONS})
    for h in FWD_HORIZONS:            # simulate each ticker's tail
        feat.loc[n - h:, f"fwd_ret_{h}"] = np.nan
    out = attach_labels(feat)
    for h in FWD_HORIZONS:
        tail = out[f"label_up_{h}"].iloc[n - h:]
        assert tail.isna().all(), f"label_up_{h} fabricated values at the tail"
        assert str(out[f"label_up_{h}"].dtype) == "Int8"


def test_correlation_prune_drops_redundant_but_not_transitive():
    """Regression: prune must compare against KEPT features only — a feature
    correlated with an already-dropped one (but not with any kept one) stays."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    y = x + 0.33 * rng.normal(size=5000)        # corr(x,y) ~ .95 -> y dropped
    z = y + 0.38 * rng.normal(size=5000)        # corr(y,z) ~ .94, corr(x,z) ~ .89
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    kept, dropped, corr = correlation_prune(df, ["x", "y", "z"], threshold=0.90)
    assert dropped["dropped"].tolist() == ["y"]
    assert kept == ["x", "z"]                    # z survives: only corr with kept x matters
    assert corr.shape == (3, 3)


def test_purged_split_embargo_exceeds_label_horizon():
    """Regression: the default embargo must be STRICTLY greater than the longest
    label horizon, else the last train row's forward label touches test_start."""
    df = pd.concat([_ticker(n=500, seed=i, name=f"T{i}") for i in range(3)])
    train, test = purged_split(df, train_frac=0.7)      # default embargo
    dates = sorted(df["date"].unique())
    gap = dates.index(test["date"].min()) - dates.index(train["date"].max())
    assert gap > max(FWD_HORIZONS)                       # 21 > 20: no label spillover


def test_macro_asof_uses_past_only():
    panel = pd.DataFrame({"ticker": "T", "date": pd.bdate_range("2022-01-03", periods=5)})
    macro = pd.DataFrame({"date": pd.to_datetime(["2022-01-03", "2022-01-06"]),
                          "vix": [20.0, 25.0]})
    merged = add_macro(panel, macro)
    # 2022-01-05 (a Wed) should carry the 01-03 value, not the future 01-06 one.
    row = merged[merged["date"] == "2022-01-05"]
    assert float(row["vix"].iloc[0]) == 20.0


def test_macro_missing_series_degrades_to_nan():
    """Regression: a macro series that failed to download must yield NaN
    features (with a warning), not a KeyError crash."""
    idx = pd.bdate_range("2022-01-03", periods=30)
    closes = pd.DataFrame({"vix": 20.0, "spx": 4000.0, "tnx10y": 3.0}, index=idx)
    import pytest
    with pytest.warns(UserWarning, match="irx13w"):
        feat = derive_macro_features(closes)             # no ^IRX column
    assert feat["term_spread"].isna().all()              # degrades, not crashes
    assert set(macro_columns()) <= set(feat.columns)
    assert feat["vix"].notna().all()                     # present series unaffected


def test_pca_fits_on_given_split():
    df = pd.concat([_ticker(n=500, seed=i, name=f"T{i}") for i in range(3)])
    feat = add_lag_features(df).dropna(subset=["roc_5", "vol_20", "rsv_20", "kmid"])
    cols = ["roc_5", "vol_20", "rsv_20", "kmid"]
    scaler, pca = pca_fit(feat, cols, var_target=0.95)
    assert pca.n_components_ <= len(cols)
    # scaler was fit on the passed frame (mean matches that frame, not something else).
    assert np.allclose(scaler.mean_, feat[cols].mean().to_numpy(), rtol=1e-6)
