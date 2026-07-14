# JPM Quant Sprint

Data-driven stock research: data acquisition → feature engineering → modeling
→ backtesting → risk management. This repo currently covers **Week 1: data
acquisition & exploration**.

## Project layout

```
jpm/
├── config/config.yaml        # single source of truth: dates, source, thresholds, paths
├── src/
│   ├── config.py             # config loader + path/date resolution
│   ├── data/
│   │   ├── universe.py       # S&P 500 constituents (Wikipedia) + ticker normalisation
│   │   ├── download.py       # OHLCV downloader (yfinance primary, Stooq fallback)
│   │   ├── integrity.py      # data-integrity checks (coverage, gaps, halts, outliers)
│   │   └── panel.py          # consolidate per-ticker files -> analysis panel
│   └── utils/                # logging
├── scripts/                  # numbered, runnable pipeline steps
│   ├── 01_fetch_universe.py
│   ├── 02_download_data.py
│   ├── 03_check_integrity.py
│   └── 04_build_panel.py
├── data/
│   ├── raw/                  # one Parquet per ticker (zstd) — resumable, git-ignored
│   ├── interim/  processed/  # cleaned data + consolidated panel — git-ignored
│   └── metadata/             # constituents list, download manifest, integrity report (tracked)
├── notebooks/  reports/      # exploration & figures (weeks 1.2 / 1.3)
├── logs/                     # run logs
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run the Week-1 pipeline

```bash
.venv/bin/python scripts/01_fetch_universe.py        # cache S&P 500 members
.venv/bin/python scripts/02_download_data.py         # 10y daily OHLCV (resumable)
.venv/bin/python scripts/03_check_integrity.py       # integrity report
.venv/bin/python scripts/04_build_panel.py           # consolidated raw panel + adj-close matrix
.venv/bin/python scripts/05_clean_data.py            # Week 1.2: clean (missing/outliers/standardise)
.venv/bin/python scripts/06_indicators.py            # Week 1.3: MA/EMA/MACD/RSI/Bollinger/ATR
.venv/bin/python scripts/07_signal_analysis.py       # Week 1.3: IC + RSI overbought/oversold
.venv/bin/python scripts/08_visualize.py --ticker AAPL   # Week 1.3: charts -> reports/figures/
```

`02` is **resumable**: re-running skips tickers already saved in `data/raw/`.
Use `--force` to re-download, `--limit N` to fetch only the first N tickers,
or `--tickers AAPL MSFT ...` for a subset.

## Data design notes

* **Source.** Primary = Yahoo Finance via `yfinance`; per-ticker fallback =
  Stooq (`pandas_datareader`) when Yahoo returns nothing (e.g. HTTP 429).
  Configurable in `config.yaml`.
* **Both raw and adjusted prices are stored** (`auto_adjust=false`): `open/high/low/close`
  are the raw traded prices, `adj_close` is split/dividend adjusted (use it for
  returns/indicators), and `dividends`/`splits` carry the corporate-action events.
  A `data/metadata/data_snapshot.json` pins per-file row counts + sha256 + fetch
  time + library versions, since adjusted history is non-stationary.
* **Storage = Parquet + zstd.** Columnar & compressed: the full S&P 500 × 10y
  panel is ~tens of MB vs. hundreds as CSV. Per-ticker files make the download
  resumable and allow re-fetching one name without rewriting everything.
* **Survivorship bias.** The universe is *today's* constituents (Wikipedia),
  so a 10-year backtest over-represents survivors. Acceptable for a learning
  sprint if acknowledged; a point-in-time membership dataset is needed to
  remove the bias. See `data/metadata/`.

## Integrity checks (`03`)

Per ticker, against a reference NYSE calendar derived from the data:
coverage vs. expected trading days, longest internal gap, duplicate dates,
NaN / non-positive prices, OHLC consistency, **halt-like rows**
(`open==high==low==close` with zero volume), zero-volume days, and extreme
(>20%) daily returns. Output: `data/metadata/integrity_report.json` +
`integrity_per_ticker.csv`. This stage only *reports*; cleaning is week 1.2.

## Cleaning (`05`, Week 1.2)

Reads `data/raw/` (never mutated) → writes `data/interim/clean/` + a consolidated
`data/processed/sp500_clean_panel.parquet`, an adjustments log, and a report.

* **Missing values.** Raw has 0 NaN cells; the real issues are structural.
  Each series is truncated to its first *sustained* real trading day (drops
  fabricated placeholder blocks — SW's pre-2024, AMCR's pre-2019), interior
  non-trading (halt) bars are masked to NaN, and only short gaps (≤5 days) are
  forward-filled with an `is_filled` flag. Pre-listing NaNs are left as NaN
  (never back-filled into fake prices); ultra-short names (FDXF, HONA) are dropped.
* **Outliers — flag, don't mutate.** A >20% day is usually a *real* event, so
  prices are never overwritten. We add `ret`/`log_ret` (on adjusted close, NaN on
  non-trading rows so no fake zero-returns), `is_extreme`/`is_suspect` flags, and a
  separate winsorised `ret_winsor` feature (median ± 3·MAD) for modeling. Only
  mechanical errors (OHLC-order, non-positive price) are corrected — each logged
  to `data/metadata/cleaning_adjustments.csv`.
* **Standardisation.** Upper-case tickers, tz-naive normalised dates, consistent
  dtypes (float64 prices, `Int64` volume, bool flags), sorted & de-duplicated.

## Indicators & signal analysis (`06`–`08`, Week 1.3)

Computed on `adj_close` (`src/features/`): MA(5/10/20/50/200), EMA(12/26),
MACD(12/26/9), RSI(14, Wilder), Bollinger(20, 2σ) with %B & bandwidth, ATR(14),
plus normalised signals (`ma20_gap`, `mom_20`, `macd_hist_norm`). Output:
`data/processed/sp500_indicators.parquet`.

**Predictive analysis** (`07`) measures each signal's **Information Coefficient**
— the mean daily cross-sectional Spearman rank-correlation between the signal and
forward 1/5/20-day returns — plus the classic RSI overbought/oversold buckets.
Finding on this data: all raw indicators are weakly **mean-reverting** (negative
IC ≈ −0.01, t ≈ −3 at 5d; oversold RSI<30 → higher forward returns). Small but
consistent — a starting point for features, not a standalone strategy. Figures in
`reports/figures/`; tables in `reports/week1_signal_analysis.md`.

> ⚠️ Forward returns overlap, so the daily ICs are autocorrelated and the t-stats
> are optimistic; treat this as exploratory, not a backtest.
