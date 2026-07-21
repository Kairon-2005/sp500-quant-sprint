#!/usr/bin/env python
"""Step 12 — correlation pruning, PCA and t-SNE on the feature matrix.

Everything is fit on the purged TRAIN split only (no test leakage). Writes a
report + figures to reports/, and the pruning/PCA tables to data/metadata/.
"""
import json

import _bootstrap  # noqa: F401
import pandas as pd

from src.config import load_config
from src.features.dataset import feature_list, purged_split
from src.features.reduce import correlation_prune, pca_fit, top_loadings, tsne_embedding
from src.features.viz import plot_corr_heatmap, plot_pca_scree, plot_tsne


def main() -> None:
    cfg = load_config()
    meta, figs = cfg.path("metadata"), cfg.resolve("reports/figures/_.png").parent
    matrix = pd.read_parquet(cfg.path("processed") / "features.parquet")
    cols = feature_list()

    train, test = purged_split(matrix, train_frac=0.7, embargo=20)
    print(f"purged split: train={len(train):,} rows (<= {train['date'].max().date()}) | "
          f"test={len(test):,} rows (>= {test['date'].min().date()})")

    # -- correlation pruning (fit on train) --
    kept, dropped = correlation_prune(train, cols, threshold=0.95)
    dropped.to_csv(meta / "feature_correlation_dropped.csv", index=False)
    print(f"\ncorrelation prune (|r|>0.95): {len(cols)} -> {len(kept)} features "
          f"({len(dropped)} dropped)")
    if len(dropped):
        print(dropped.to_string(index=False))

    # -- PCA (fit on train, pruned features) --
    scaler, pca = pca_fit(train, kept, var_target=0.95)
    loadings = top_loadings(pca, kept)
    loadings.to_csv(meta / "pca_loadings.csv", index=False)
    print(f"\nPCA: {len(kept)} features -> {pca.n_components_} components for 95% variance")
    print("top loadings (PC1-3):")
    print(loadings.to_string(index=False))

    # -- figures --
    plot_corr_heatmap(train, cols, figs / "feature_corr_heatmap.png")
    plot_pca_scree(pca, figs / "pca_scree.png")
    emb = tsne_embedding(test, kept, n_sample=2500)
    plot_tsne(emb, test.loc[emb.index, "fwd_ret_5"].clip(-0.1, 0.1),
              figs / "tsne_features.png")
    print(f"\nfigures -> {figs}/(feature_corr_heatmap, pca_scree, tsne_features).png")

    report = {
        "n_features": len(cols),
        "n_after_prune": len(kept),
        "dropped_features": dropped["dropped"].tolist() if len(dropped) else [],
        "pca_components_95pct": int(pca.n_components_),
        "pca_explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "embargo_days": 20,
    }
    with open(meta / "feature_reduction_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
