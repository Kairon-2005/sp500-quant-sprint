"""Week 2.3 — correlation pruning & dimensionality reduction.

* **Correlation pruning** removes redundant features (|Pearson ρ| > threshold),
  keeping the earlier one in a fixed order (deterministic).
* **PCA** compresses the pruned set to the components explaining a variance
  target. The scaler and PCA are **fit on the training split only** and then
  applied to test — fitting on the full sample would leak test-period structure
  (the exact mistake flagged in the mentor questions).
* **t-SNE** is for 2-D *visualisation* only (it can't transform new points), on a
  sample of rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_prune(df: pd.DataFrame, cols: list[str], threshold: float = 0.95):
    """Drop features correlated > threshold with an earlier-kept feature."""
    corr = df[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = []
    for c in upper.columns:
        if (upper[c] > threshold).any():
            partner = upper[c].idxmax()
            dropped.append({"dropped": c, "correlated_with": partner,
                            "corr": round(float(upper[c].max()), 3)})
    drop_names = {d["dropped"] for d in dropped}
    kept = [c for c in cols if c not in drop_names]
    return kept, pd.DataFrame(dropped)


def pca_fit(train: pd.DataFrame, cols: list[str], var_target: float = 0.95):
    """Fit StandardScaler + PCA on the TRAIN split only."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train[cols])
    pca = PCA(n_components=var_target, svd_solver="full").fit(scaler.transform(train[cols]))
    return scaler, pca


def pca_transform(df: pd.DataFrame, cols: list[str], scaler, pca) -> pd.DataFrame:
    comps = pca.transform(scaler.transform(df[cols]))
    out = pd.DataFrame(comps, columns=[f"pc{i+1}" for i in range(comps.shape[1])],
                       index=df.index)
    return out


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
    from sklearn.preprocessing import StandardScaler

    s = df.dropna(subset=cols)
    if len(s) > n_sample:
        s = s.sample(n_sample, random_state=seed)
    xs = StandardScaler().fit_transform(s[cols])
    emb = TSNE(n_components=2, init="pca", perplexity=30,
               random_state=seed).fit_transform(xs)
    return pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1]}, index=s.index)
