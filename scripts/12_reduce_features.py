#!/usr/bin/env python
"""Step 12 — correlation pruning, PCA and t-SNE on the feature matrix.

Thin wrapper: the stage logic lives in src/features/reduce.run_reduction, with
every knob read from the `features` section of config.yaml.
"""
import _bootstrap  # noqa: F401

from src.config import load_config
from src.features.reduce import run_reduction


def main() -> None:
    report = run_reduction(load_config())

    print("\n=== Feature reduction ===")
    print(f"features        : {report['n_features']} -> {report['n_after_prune']} after "
          f"|r|>{report['params']['corr_threshold']} prune "
          f"(dropped: {', '.join(report['dropped_features']) or 'none'})")
    print(f"PCA             : {report['n_after_prune']} -> {report['pca_components']} "
          f"components for {report['params']['pca_var_target']:.0%} variance")
    print(f"split           : train={report['train_rows']:,} test={report['test_rows']:,} "
          f"(embargo={report['params']['embargo_days']})")


if __name__ == "__main__":
    main()
