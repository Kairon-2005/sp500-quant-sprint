"""Week 2.3 — correlation pruning & dimensionality reduction.

* **Correlation pruning** removes redundant features: walking the columns in
  order, a feature is dropped only if it correlates above the threshold with an
  already-**kept** feature (comparing against dropped ones would transitively
  discard non-redundant signal).
* **PCA** compresses the pruned set to the components explaining a variance
  target. The scaler and PCA are **fit on the training split only** and then
  applied to test — fitting on the full sample would leak test-period structure.
* **t-SNE** is for 2-D *visualisation* only (it can't transform new points), on a
  sample of rows.

``run_reduction`` orchestrates the full stage (split -> prune -> PCA -> t-SNE ->
figures + report) with every knob read from the ``features`` config section, so
the reported parameters can never diverge from the applied ones.
"""
from __future__ import annotations

import json

import pandas as pd

from ..storage import ParquetStore
from ..utils import get_logger
from .dataset import feature_list, purged_split


def _standardise(train_df: pd.DataFrame, cols: list[str]):
    """One shared scaler definition for PCA and t-SNE (fit on what's given)."""
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit(train_df[cols])


def correlation_prune(df: pd.DataFrame, cols: list[str], threshold: float = 0.95):
    """Greedy prune: keep a feature unless it correlates > threshold with an
    earlier KEPT feature. Returns (kept, dropped_table, corr_matrix)."""
    corr = df[cols].corr()
    abs_corr = corr.abs()
    kept: list[str] = []
    dropped = []
    for c in cols:
        partners = [k for k in kept if abs_corr.loc[c, k] > threshold]
        if partners:
            worst = max(partners, key=lambda k: abs_corr.loc[c, k])
            dropped.append({"dropped": c, "correlated_with": worst,
                            "corr": round(float(abs_corr.loc[c, worst]), 3)})
        else:
            kept.append(c)
    return kept, pd.DataFrame(dropped), corr


def pca_fit(train: pd.DataFrame, cols: list[str], var_target: float = 0.95):
    """Fit StandardScaler + PCA on the TRAIN split only."""
    from sklearn.decomposition import PCA
    scaler = _standardise(train, cols)
    pca = PCA(n_components=var_target, svd_solver="full").fit(scaler.transform(train[cols]))
    return scaler, pca


def pca_transform(df: pd.DataFrame, cols: list[str], scaler, pca) -> pd.DataFrame:
    comps = pca.transform(scaler.transform(df[cols]))
    return pd.DataFrame(comps, columns=[f"pc{i+1}" for i in range(comps.shape[1])],
                        index=df.index)


def top_loadings(pca, cols: list[str], n_components: int = 3, n_top: int = 6) -> pd.DataFrame:
    """The features that dominate the first few principal components."""
    rows = []
    for i in range(min(n_components, pca.n_components_)):
        load = pd.Series(pca.components_[i], index=cols)
        for name, val in load.reindex(load.abs().sort_values(ascending=False).index).head(n_top).items():
            rows.append({"pc": f"pc{i+1}", "feature": name, "loading": round(float(val), 3)})
    return pd.DataFrame(rows)


def tsne_embedding(df: pd.DataFrame, cols: list[str], n_sample: int = 2500,
                   seed: int = 42) -> pd.DataFrame:
    """2-D t-SNE embedding of a random row sample (viz only)."""
    from sklearn.manifold import TSNE
    s = df.dropna(subset=cols)
    if len(s) > n_sample:
        s = s.sample(n_sample, random_state=seed)
    xs = _standardise(s, cols).transform(s[cols])
    emb = TSNE(n_components=2, init="pca", perplexity=30,
               random_state=seed).fit_transform(xs)
    return pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1]}, index=s.index)


def run_reduction(cfg) -> dict:
    """Full 2.3 stage: purged split -> prune -> PCA -> t-SNE -> figures + report."""
    from .viz import plot_corr_heatmap, plot_pca_scree, plot_tsne

    log = get_logger("reduce", cfg.path("logs"))
    fc = cfg.get("features", {})
    train_frac = float(fc.get("train_frac", 0.7))
    embargo = fc.get("embargo_days")            # None -> purged_split's safe default
    corr_threshold = float(fc.get("corr_threshold", 0.95))
    var_target = float(fc.get("pca_var_target", 0.95))
    n_sample = int(fc.get("tsne_sample", 2500))

    meta = cfg.path("metadata")
    figs = cfg.resolve("reports/figures/_.png").parent
    matrix = ParquetStore(cfg.path("processed")).read("features")
    cols = feature_list()

    train, test = purged_split(matrix, train_frac=train_frac, embargo=embargo)
    applied_embargo = embargo if embargo is not None else "max_horizon+1"
    log.info("purged split: train=%d (<= %s) | test=%d (>= %s) | embargo=%s",
             len(train), train["date"].max().date(),
             len(test), test["date"].min().date(), applied_embargo)

    kept, dropped, corr = correlation_prune(train, cols, threshold=corr_threshold)
    dropped.to_csv(meta / "feature_correlation_dropped.csv", index=False)
    log.info("correlation prune (|r|>%.2f): %d -> %d features",
             corr_threshold, len(cols), len(kept))

    scaler, pca = pca_fit(train, kept, var_target=var_target)
    loadings = top_loadings(pca, kept)
    loadings.to_csv(meta / "pca_loadings.csv", index=False)
    log.info("PCA: %d features -> %d components for %.0f%% variance",
             len(kept), pca.n_components_, var_target * 100)

    plot_corr_heatmap(corr, figs / "feature_corr_heatmap.png")
    plot_pca_scree(pca, figs / "pca_scree.png")
    emb = tsne_embedding(test, kept, n_sample=n_sample)
    color = test.loc[emb.index, "fwd_ret_5"]
    valid = color.notna()                       # tail rows have no forward return
    plot_tsne(emb[valid.to_numpy()], color[valid].clip(-0.1, 0.1),
              figs / "tsne_features.png")

    report = {
        "n_features": len(cols),
        "n_after_prune": len(kept),
        "dropped_features": dropped["dropped"].tolist() if len(dropped) else [],
        "pca_components": int(pca.n_components_),
        "pca_explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "params": {"train_frac": train_frac, "embargo_days": applied_embargo,
                   "corr_threshold": corr_threshold, "pca_var_target": var_target,
                   "tsne_sample": n_sample},
        "dropped_table": dropped.to_dict("records"),
        "top_loadings": loadings.to_dict("records"),
    }
    with open(meta / "feature_reduction_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info("report -> %s | figures -> %s", meta / "feature_reduction_report.json", figs)
    return report
