# Week 2 — Feature Engineering

Grounded in **Qlib Alpha158** (feature formulas) and **Stefan Jansen's ML for
Trading** (feature families, label engineering, purged CV). Pipeline:
`scripts/11_build_features.py` → `scripts/12_reduce_features.py`.

## What was built

**Feature matrix** (`data/processed/features.parquet`): 601 tickers, ~1.30M rows,
**48 features**, 2017-07 → 2026-07 (first year dropped as indicator/lag warm-up).

| Group (Jansen family) | Features |
|---|---|
| Momentum / reversal | `roc_{1,5,10,20,60}`, `mom_12_1` (12-1 month) |
| Volatility | `vol_{5,10,20,60}`, `vol_regime`, `skew_20`, `kurt_20` |
| Microstructure (Alpha158 K-bars) | `kmid`, `klen`, `kmid2`, `kup`, `klow`, `ksft` |
| Price position / trend | `ma_gap_{5,20,60}`, `rsv_{20,60}`, `higap`, `logap`, `cntp_20`, `cntd_20` |
| Liquidity / volume | `vratio_{5,20}`, `dollar_vol`, `amihud_20`, `corr_pv_20` |
| Week-1 indicators reused | `rsi14`, `macd_hist_norm`, `bb_pctb`, `bb_width` |
| Macro / sentiment | `vix`, `vix_chg`, `vix_z`, `tnx10y`, `tnx_chg`, `term_spread`, `spx_ret_1`, `spx_vol_20`, `spx_ma_gap` |

**Labels**: forward returns `fwd_ret_{1,5,20}` + binary `label_up_{h}` (strictly
future — never overlaps features).

## 2.1 — Short-term reversal vs long-term momentum

Mean cross-sectional IC of return-window features (the spec's "short momentum vs
long trend" question):

| signal | fwd 5d IC | fwd 20d IC | reading |
|---|---|---|---|
| roc_1 / roc_5 / roc_10 | **−0.010 to −0.013** (t≈−3) | negative | short-term **reversal** |
| roc_20 | −0.008 | −0.010 | fading reversal |
| roc_60 | ~0 | ~0 | neutral |
| **mom_12_1** | **+0.021 (t=+4.3)** | **+0.021 (t=+4.8)** | long-term **momentum** |

→ The classic pattern: recent returns mean-revert, but 12-month momentum
(excluding the last month) is the strongest single predictor — the Jegadeesh-Titman
momentum anomaly, clearly present in our survivorship-free universe.

## 2.2 — Macro / sentiment

VIX, 10y & 13w Treasury yields, term spread, and market return/vol, **as-of merged**
(backward join — no future macro leaks). **Twitter sentiment: honest omission** —
the X API is paid/closed with no free historical feed; VIX serves as the fear proxy,
and a real social feed can be plugged behind `add_macro()`.

## 2.3 — Correlation pruning + PCA

* **Pearson pruning (|ρ|>0.95)**: 48 → 46 (dropped `cntd_20`≈`cntp_20` ρ=0.997,
  `bb_pctb`≈`rsv_20` ρ=0.956).
* **PCA (fit on train only)**: 46 features → **26 components for 95% variance**.
  The leading components map to recognisable risk factors:
  * **PC1 (~24%)** — trend/momentum (`ma_gap`, `rsi14`, `roc_20`, `rsv`)
  * **PC2 (~15%)** — volatility (`vol_5..60`, `bb_width`)
  * **PC3 (~8%)** — intraday + market (`ksft`, `kmid`, `roc_1`, `spx_ret_1`, `vix_chg`)
* **t-SNE** 2-D embedding of a test sample (viz only).

Figures: `reports/figures/{feature_corr_heatmap, pca_scree, tsne_features}.png`.

## Leakage discipline (enforced)

* Features are point-in-time at t; labels look forward — proven by a unit test
  that recomputes features on a truncated series and asserts identical values.
* **Correlation pruning, StandardScaler and PCA are fit on the TRAIN split only.**
* Train/test use a **purged, embargoed** time split (20-session gap) so a training
  row's forward label cannot spill into the test window.
