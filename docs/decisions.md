# Critical Decisions Log — Week 1

Every non-obvious choice made in the data pipeline, with the reasoning and the
trade-off. Newest sections build on earlier ones. See `reports/week1_data_review.md`
for the independent review that drove several of these.

---

## Data source & fidelity

**D1. Yahoo Finance (`yfinance`) as primary, Stooq as fallback.**
Free, no API key, 10y+ daily coverage, reproducible. A raw HTTP call to Yahoo
returns HTTP 429, but `yfinance` + `curl_cffi` browser impersonation gets through.
*Trade-off:* unofficial API, no point-in-time index membership. Alternatives with
survivorship-free data (CRSP, Norgate, EODHD) cost money / access — deferred.

**D2. Store BOTH raw and adjusted prices (`auto_adjust=False`).**
`open/high/low/close` are raw traded prices; `adj_close` is split+dividend
adjusted; `dividends`/`splits` keep the corporate-action events. *Why:* adjusted
prices are the right basis for returns/indicators, but adjusted history is
non-stationary (future dividends rewrite it) and can't give a true fill price or
dollar volume. Keeping both covers returns, realistic backtests, and split
detection. (Originally stored adjusted-only — the review caught that the "raw
retained" claim was false; fixed.)

**D3. Universe = current Wikipedia S&P 500 → survivorship bias, documented not fixed.**
A 10y backtest on today's members over-weights survivors. Acceptable for a
learning sprint if acknowledged; removing it needs point-in-time membership
(a paid/CRSP dataset). Flagged in README + this log; open question for the mentor.

**D4. Reproducibility via `data_snapshot.json`.**
Per-file row count + sha256 + fetch time + library versions, because adjusted
history drifts. Deps pinned to the real environment (pandas 3.0, yfinance 1.5).

## Storage & layout

**D5. Per-ticker Parquet + zstd, consolidated into a long panel.**
Per-ticker files make downloads resumable and let one name be re-fetched in
isolation; Parquet+zstd is ~3.6× smaller than CSV and column-selective. A wide
adjusted-close matrix is derived for correlation/return work.

**D6. `raw/` is immutable; cleaning writes to `interim/` + `processed/`.**
Cleaning never edits raw. Every value change is recorded in
`cleaning_adjustments.csv`. Keeps the pipeline auditable and re-runnable.

## Integrity & cleaning

**D7. Integrity only reports; cleaning is a separate, deliberate stage.**
Separation of concerns — `03` never mutates.

**D8. Coverage alone is not "completeness"; added `real_row_ratio`.**
Date-presence coverage read 1.0 for SW (41% flat placeholder rows) and AMCR
(22%). Added `real_row_ratio` / `first_real_date` / `min_history_days` so
fabricated and ultra-short series `fail` instead of passing silently.

**D9. Outliers: FLAG, don't mutate. Only mechanical errors are corrected.**
A >20% daily move is usually real (earnings, biotech, bankruptcy, COVID crash) —
verified median annual extreme ≈ 0.19. So prices are never overwritten for size;
we add `is_extreme`/`is_suspect` flags and a separate winsorised `ret_winsor`
feature. Only OHLC-order violations and non-positive prices are corrected (each
logged). *This reverses the literal Week-1.2 spec ("correct >20% moves"), which
would delete real signal.*

**D10. Missing values are structural, handled by trim + constrained fill.**
Raw has 0 NaN cells. Real issues: (a) fabricated leading blocks → truncate to the
first *sustained* real trading day; (b) interior halt bars → mask to NaN;
(c) short gaps (≤5 days) → forward-fill with an `is_filled` flag. Never ffill
before the first bar or across a delisting (avoids look-ahead / fake quotes);
returns on filled/halt rows are NaN, not 0.

## Indicators & analysis

**D11. Indicators on `adj_close`; RSI uses Wilder smoothing; conventional periods.**
Adjusted close avoids split/dividend jumps. Kept default periods (RSI14,
MACD 12/26/9, BB 20/2σ) to avoid overfitting parameters on one sample.

**D12. Predictive power measured by cross-sectional IC (Spearman), not time-series correlation.**
Daily rank-correlation of a signal vs forward return across the universe — the
standard, scale/outlier-robust quant measure. Reported with IR and t-stat.
*Caveat logged:* forward windows overlap → ICs autocorrelated → t-stats optimistic;
this is exploratory, not a backtest.

## Code design (this refactor)

**D13. OOP where there is real reuse; pure functions for stateless math.**
- `ParquetStore` (repository pattern) — the one place that knows Parquet IO +
  filesystem-safe ticker naming; used by every stage.
- `PriceSource` (strategy pattern) with `YahooSource`/`StooqSource` — a new vendor
  is one subclass; the `Downloader` tries primary then fallbacks uniformly.
- Indicators stay **pure functions** — they are stateless transforms; wrapping
  them in classes would add ceremony without reuse. Deliberate non-OOP choice.

**D14. `pyproject.toml` for tooling; tests run via `pytest` (no `sys.path` hacks).**
Standard project metadata + `pythonpath=["."]` so tests import `src` cleanly.

**D15. Partial current-day bar is clamped + logged, not specially dropped (known artifact).**
Downloading intraday pulls a still-forming bar for *today* whose High/Low lag the
last trade, so open/close can momentarily exceed them — this showed up as ~22
tiny `ohlc_order_fix` edits all dated the run day. The cleaner clamps and logs
them harmlessly. A stricter pipeline would drop any incomplete session; deferred
(needs market-close awareness). Re-running after the close removes them.

## Mentor-feedback round (survivorship, lifecycle, reconciliation)

**D16. Point-in-time membership from Wikipedia's changes table (survivorship fix).**
`membership.py` reconstructs historical index membership by walking backwards from
today's members and reversing each add/remove event. Gives a survivorship-free
universe (503 -> 699) and recovers delisted names. *Honest limit:* Wikipedia is
not point-in-time authoritative (CRSP/Norgate are); dates/renames can be imperfect.
It's a free, documented approximation — far better than "today's snapshot only".

**D17. Delisted-from-index ≠ delisted-from-market.** Of ~196 removed names, ~107
are fetchable from Yahoo and only ~10 actually stopped trading (`lifecycle=delisted`);
most removals are market-cap demotions where the company still trades. The other
~89 (acquired/gone) are absent from Yahoo entirely — a hard limit of the free source,
reported in the download manifest, not silently hidden.

**D18. Lifecycle classification** (`integrity.py`): every ticker tagged
active / delisted (data stops >N sessions before market end) / suspended (long
internal gap but resumes) / short_history — so long-dead and long-halted names are
separated, not lumped with active ones.

**D19. Pre-listing invalid data is split out, not silently trimmed.** Cleaning
still truncates fabricated leading rows, but now writes `prelisting_trimmed.csv`
(ticker, original vs first-real date, rows trimmed) and a `quarantine.csv` of names
auto-filtered by the min-history rule — so the removals are auditable.

**D20. Reconciliation is a real comparison engine, but the free second feed is gated.**
`reconcile.py` compares two vendors' raw close per ticker and flags mismatch
(persistent price disagreement => ticker reuse) / warn / ok. The engine is
unit-tested on synthetic data. *Live limit:* Stooq now sits behind a JavaScript
proof-of-work bot-check (and `pandas_datareader` dropped Stooq), so it can't be hit
from a plain request — and we do **not** bypass bot detection. To run it live, use a
network where Stooq is reachable or plug a keyed feed (e.g. Tiingo) into the same
`PriceSource` interface. Framework done; live execution needs a reachable source.
