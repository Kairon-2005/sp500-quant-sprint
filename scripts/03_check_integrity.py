#!/usr/bin/env python
"""Step 3 — run data-integrity checks and write reports."""
import _bootstrap  # noqa: F401

from src.config import load_config
from src.data.integrity import run_integrity


def main() -> None:
    cfg = load_config()
    report = run_integrity(cfg)

    print("\n=== Integrity summary (by status) ===")
    print(report["status"].value_counts().to_string())

    warn_fail = report[report["status"].isin(["warn", "fail"])]
    if len(warn_fail):
        cols = ["ticker", "status", "coverage", "max_internal_gap",
                "n_halt_like", "n_extreme_return", "flags"]
        print("\n=== Tickers needing attention (top 25) ===")
        print(warn_fail[cols].head(25).to_string(index=False))
    else:
        print("\nAll tickers clean.")


if __name__ == "__main__":
    main()
