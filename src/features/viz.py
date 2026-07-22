"""Matplotlib/Seaborn charts for Week 1.3.

Uses the non-interactive Agg backend and saves PNGs under reports/figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")


def plot_price_panel(df: pd.DataFrame, ticker: str, out: Path,
                     last_n: int = 504) -> Path:
    """3-panel chart: price + MA20/MA50 + Bollinger, then RSI, then MACD."""
    d = df.sort_values("date").tail(last_n).copy()
    x = pd.to_datetime(d["date"])

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(13, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1.2]})

    # --- Panel 1: price + MA + Bollinger band ---
    ax1.plot(x, d["adj_close"], color="#111", lw=1.3, label="Adj Close")
    ax1.plot(x, d["ma20"], color="#1f77b4", lw=1.0, label="MA20")
    ax1.plot(x, d["ma50"], color="#ff7f0e", lw=1.0, label="MA50")
    ax1.plot(x, d["bb_upper"], color="#888", lw=0.8, ls="--", label="Bollinger ±2σ")
    ax1.plot(x, d["bb_lower"], color="#888", lw=0.8, ls="--")
    ax1.fill_between(x, d["bb_lower"], d["bb_upper"], color="#1f77b4", alpha=0.06)
    ax1.set_title(f"{ticker} — price, MA20/50 & Bollinger Bands "
                  f"(last {last_n} sessions)", fontsize=13, weight="bold")
    ax1.set_ylabel("Price ($, adj)")
    ax1.legend(loc="upper left", ncols=2, fontsize=9)

    # --- Panel 2: RSI ---
    ax2.plot(x, d["rsi14"], color="#6a3d9a", lw=1.1)
    ax2.axhline(70, color="#d62728", lw=0.8, ls="--")
    ax2.axhline(30, color="#2ca02c", lw=0.8, ls="--")
    ax2.axhspan(70, 100, color="#d62728", alpha=0.06)
    ax2.axhspan(0, 30, color="#2ca02c", alpha=0.06)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI(14)")

    # --- Panel 3: MACD ---
    colors = np.where(d["macd_hist"] >= 0, "#2ca02c", "#d62728")
    ax3.bar(x, d["macd_hist"], color=colors, width=1.0, alpha=0.5, label="Histogram")
    ax3.plot(x, d["macd"], color="#1f77b4", lw=1.0, label="MACD")
    ax3.plot(x, d["macd_signal"], color="#ff7f0e", lw=1.0, label="Signal")
    ax3.axhline(0, color="#444", lw=0.6)
    ax3.set_ylabel("MACD")
    ax3.legend(loc="upper left", ncols=3, fontsize=8)
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax3.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_rsi_buckets(bucket_df, horizons, out: Path) -> Path:
    """Bar chart: mean forward return (bps) by RSI bucket, per horizon."""
    d = bucket_df.copy()
    d["rsi_bucket"] = d["rsi_bucket"].astype(str)
    cols = [f"fwd_{h}d_bps" for h in horizons]
    m = d.melt(id_vars="rsi_bucket", value_vars=cols,
               var_name="horizon", value_name="bps")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=m, x="rsi_bucket", y="bps", hue="horizon", ax=ax)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_title("Mean forward return by RSI(14) bucket "
                 "(pooled, S&P 500, ~10y)", fontsize=13, weight="bold")
    ax.set_xlabel("RSI(14) bucket")
    ax.set_ylabel("Mean forward return (bps)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="horizon")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_ic_heatmap(ic_df, out: Path) -> Path:
    """Heatmap of mean IC across signal x horizon."""
    piv = ic_df.pivot(index="signal", columns="horizon", values="mean_ic")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(piv, annot=True, fmt=".3f", center=0, cmap="RdBu_r",
                cbar_kws={"label": "mean daily IC (Spearman)"}, ax=ax)
    ax.set_title("Predictive power: mean IC by signal & horizon",
                 fontsize=12, weight="bold")
    ax.set_xlabel("forward horizon (days)")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Week 2 — feature-engineering charts
# --------------------------------------------------------------------------
def plot_corr_heatmap(corr: pd.DataFrame, out: Path) -> Path:
    """Feature Pearson-correlation heatmap from a precomputed corr matrix."""
    n = len(corr)
    size = min(0.4 * n + 2, 20)
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={"shrink": 0.6}, ax=ax)
    ax.set_title(f"Feature correlation ({n} features)", fontsize=13, weight="bold")
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pca_scree(pca, out: Path) -> Path:
    """PCA explained-variance scree + cumulative curve."""
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    x = np.arange(1, len(evr) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, evr, color="#1f77b4", alpha=0.6, label="per component")
    ax.plot(x, cum, color="#d62728", marker="o", ms=3, label="cumulative")
    ax.axhline(0.95, color="#333", ls="--", lw=0.8)
    ax.set_title(f"PCA scree — {len(evr)} components reach 95% variance",
                 fontsize=13, weight="bold")
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_tsne(emb: pd.DataFrame, color: pd.Series, out: Path,
              title: str = "t-SNE of feature space") -> Path:
    """2-D t-SNE scatter coloured by a series (e.g. forward-return sign)."""
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(emb["x"], emb["y"], c=color.to_numpy(), cmap="coolwarm",
                    s=6, alpha=0.6)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="forward 5d return")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
