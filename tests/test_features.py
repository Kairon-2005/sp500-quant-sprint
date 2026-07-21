"""Offline tests for Week-2 feature engineering (no network)."""
import numpy as np
import pandas as pd

from src.features.dataset import purged_split
from src.features.lags import add_lag_features
from src.features.macro import add_macro
from src.features.reduce import correlation_prune, pca_fit


def _ticker(n=400, seed=1):
    dates = pd.bdate_range("2021-01-04", periods=n)
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.012, n))
    return pd.DataFrame({
        "date": dates, "ticker": "TST",
        "open": close * 0.997, "high": close * 1.012, "low": close * 0.988,
        "close": close, "adj_close": close,
        "volume": rng.integers(1e6, 5e6, n).astype("float64"),
        "ret": pd.Series(close).pct_change().values,
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


def test_correlation_prune_drops_redundant():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    df = pd.DataFrame({"a": a, "b": a + rng.normal(0, 1e-3, 500), "c": rng.normal(size=500)})
    kept, dropped = correlation_prune(df, ["a", "b", "c"], threshold=0.95)
    assert "b" in dropped["dropped"].values      # b ~ a -> dropped
    assert "a" in kept and "c" in kept


def test_purged_split_has_embargo_gap():
    df = pd.concat([_ticker(n=500, seed=i).assign(ticker=f"T{i}") for i in range(3)])
    train, test = purged_split(df, train_frac=0.7, embargo=20)
    assert train["date"].max() < test["date"].min()
    dates = sorted(df["date"].unique())
    gap = dates.index(test["date"].min()) - dates.index(train["date"].max())
    assert gap >= 20                              # purge gap enforced


def test_macro_asof_uses_past_only():
    panel = pd.DataFrame({"ticker": "T", "date": pd.bdate_range("2022-01-03", periods=5)})
    macro = pd.DataFrame({"date": pd.to_datetime(["2022-01-03", "2022-01-06"]),
                          "vix": [20.0, 25.0]})
    merged = add_macro(panel, macro)
    # 2022-01-05 (a Wed) should carry the 01-03 value, not the future 01-06 one.
    row = merged[merged["date"] == "2022-01-05"]
    assert float(row["vix"].iloc[0]) == 20.0


def test_pca_fits_on_given_split():
    df = pd.concat([_ticker(n=500, seed=i).assign(ticker=f"T{i}") for i in range(3)])
    feat = add_lag_features(df).dropna(subset=["roc_5", "vol_20", "rsv_20", "kmid"])
    cols = ["roc_5", "vol_20", "rsv_20", "kmid"]
    scaler, pca = pca_fit(feat, cols, var_target=0.95)
    assert pca.n_components_ <= len(cols)
    # scaler was fit on the passed frame (mean matches that frame, not something else).
    assert np.allclose(scaler.mean_, feat[cols].mean().to_numpy(), rtol=1e-6)
