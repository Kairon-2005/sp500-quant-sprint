# Week 1 Data Pipeline — Adversarial Design & Code Review

**Reviewer:** independent senior quant data engineer
**Date:** 2026-07-09
**Scope:** data acquisition + integrity (Week 1.1). Cleaning (1.2) and indicators (1.3) not yet built.
**Verdict basis:** all findings below were confirmed against the actual code paths and the
504 downloaded parquet files, not inferred from the README.

---

## Summary verdict

The pipeline is **well-engineered as software** (clean structure, resumable, tested, honest
survivorship caveat) but has **material data-quality blind spots that make it *not yet* fit
for return/indicator work without a cleaning pass that is more surgical than the one the
Week-1.2 spec describes.** Two concrete, verified problems dominate: (a) the integrity
**coverage metric reports 1.0 for series that are up to 41% fabricated flat placeholder rows**
(SW, AMCR), so the report's "0 fail / 217 warn" headline understates the real state of the
data; and (b) the config/README claim that **raw unadjusted OHLC is retained is false** — only
adjusted OHLC is stored, with no `adj_close`, dividends, splits, or true volume-price. The
Week-1.2 task as written ("detect and *correct* >20% daily moves") is actively dangerous and
must be reframed before any code is written.

---

## Strengths (honest, brief)

- **Resumable per-ticker parquet + zstd** is the right storage design; re-runs skip cached
  tickers; single names can be re-fetched. Manifest is written every run.
- **Survivorship bias is documented** in three places (config, README, universe.py) rather than
  hidden. Good intellectual honesty.
- **Integrity stage only *reports*, never mutates.** Correct separation — cleaning is a
  deliberate downstream decision.
- **OHLC-consistency check uses a relative tolerance** (`integrity.py:102`) to avoid spurious
  violations from float noise in adjusted prices — a subtle, correct touch.
- **Offline unit tests exist and pass** (5/5), covering symbol mapping, `_standardise`, and the
  halt/extreme detectors on synthetic data.
- Batch retry logic retries only the *still-empty* subset (`_retry_batch`), which is efficient
  and polite.

---

## Weaknesses & risks — by severity

### CRITICAL

**C1. Coverage metric is blind to fabricated/placeholder rows — the headline integrity number is misleading.**
*File:* `integrity.py:71-76`. Coverage = `present_dates / expected_trading_days` within
`[first, last]`. It counts a date as "present" regardless of whether the row is a real quote or a
carried-forward flat placeholder. Verified consequence:

| ticker | total rows | real rows | flat/zero-vol rows | % fabricated | reported coverage | status |
|--------|-----------:|----------:|-------------------:|-------------:|------------------:|--------|
| SW     | 2512       | 1476      | 1036               | **41.2%**    | 1.0               | warn   |
| AMCR   | 2512       | 1959      | 553                | **22.0%**    | 1.0               | warn   |

SW (Smurfit WestRock) did not exist as the current entity for most of the window; the pre-merger
series is thinly traded with zero-volume days that carry the prior close forward. Coverage=1.0
tells the reader "this series is complete" when nearly half of it is non-informative. *Why it
matters:* these flat rows inject artificial **zero-return days**, understate volatility, corrupt
MA/RSI/Bollinger, and will silently poison any model that trains on the panel. *Fix:* add a
`real_row_ratio` = (rows that are not flat-zero-volume) / total, and treat a low ratio as a
first-class flag (or `fail`). Report `first_real_trading_date` distinct from `first_date`.

**C2. "Raw unadjusted OHLC is also retained" is false — only adjusted OHLC is stored.**
*Files:* `config.yaml:23`, `README.md:59-60` claim raw is kept; `download.py:28,60,74`
contradict it. `_STD_COLS` has no `adj_close`; the rename `{"adj_close": "adj_close"}` is a
no-op; `auto_adjust=True` means yfinance never even returns a separate Adj Close. Verified: no
parquet file has an `adj_close` column. *Why it matters:* (1) adjusted history is **non-stationary
by construction** — every future dividend retroactively rewrites the entire back-history, so the
dataset is not reproducible and any stored feature/label goes stale; (2) you **cannot recover the
true traded price** for realistic backtest fills or VWAP; (3) volume is split-adjusted but price
is div+split-adjusted, so **`price × volume` is not dollar volume**; (4) no dividend/split events
are retained, so you can't detect an *unadjusted* split (a real data error) versus a legitimate
adjustment. *Fix:* re-download with `auto_adjust=False` and store BOTH raw OHLC and `adj_close`
(plus the `actions`/dividends+splits frame), or at minimum store the raw close alongside adjusted.
Correct the README/config to match reality now, regardless.

**C3. Week-1.2 as specified ("correct outliers / >20% moves") will destroy real signal.**
See the dedicated cleaning section below — this is the single biggest risk in the whole week and
it hasn't been built yet, so it's the cheapest to fix.

### MAJOR

**M1. Stooq fallback is dead code — never exercised, never validated, no cross-source reconciliation.**
*File:* `download.py:191-193`. The manifest shows **499/504 ok, all `source=yfinance`, 0 from
stooq**. The fallback only fires when yfinance returns *empty*, never when it returns *suspect*
data, so the "cross-source reconciliation" implied by having two sources is not actually
performed. Worse, Stooq returns **unadjusted** prices while yfinance here is adjusted — if the
fallback ever *did* fire, that ticker's file would be on a totally different price basis than its
neighbours, silently. *Fix:* either (a) remove the pretense and document single-source, or (b)
make Stooq a genuine reconciliation check on a sample and normalise its adjustment basis before
mixing.

**M2. Reproducibility: no data snapshot, hash, or dependency pinning that matches the real env.**
`requirements.txt` pins `yfinance>=0.2.40` and `pandas>=2.2`, but the actual environment is
**yfinance 1.5 / pandas 3.0** (per MEMORY). Nothing records the yfinance version, the fetch
timestamp per ticker, or a content hash of each file. Combined with C2's non-stationarity, the
dataset **cannot be reproduced** and two runs a month apart will differ. *Fix:* pin exact
versions; write a `data_snapshot.json` with per-file row count + sha256 + fetch UTC + yfinance
version; treat `data/raw` as an immutable dated snapshot.

**M3. Near-empty "ticker" artifacts admitted without question.**
Verified: **HONA = 16 rows** (starts 2026-06-15), **FDXF = 29 rows**, both coverage=1.0 because
their `[first,last]` window is tiny. These are brand-new spinoff/placeholder symbols scraped from
today's Wikipedia table. A 16-row "stock" has no business in a 10-year panel and will break any
per-ticker rolling indicator. *Fix:* add a `min_history_days` gate (e.g. 60–252) that flags or
quarantines ultra-short series explicitly, rather than letting coverage=1.0 wave them through.

**M4. `fail` severity is effectively unreachable for the real failure modes here.**
*File:* `integrity.py:139-144`. `fatal` triggers on coverage<0.5, dupes, non-positive price, or
>0.5% OHLC violations. But the actual bad series (SW, AMCR, HONA) have **coverage=1.0, 0 dupes, 0
non-pos, 0 OHLC violations** — so they can never be `fail`. The classifier measures the wrong
things. The "0 fail" result is a property of the thresholds, not of the data being clean.
*Fix:* fold C1's `real_row_ratio` and M3's history gate into the fatal condition.

### MINOR

**M5. `pd.Timestamp.utcnow()` is deprecated in pandas 3.0.**
*File:* `integrity.py:189`. Verified it raises `Pandas4Warning` ("use `Timestamp.now('UTC')`").
Harmless today, will break on the next major. One-line fix.

**M6. Volume dtype is inconsistent across the panel (int64 vs float64).**
Verified: **467 tickers int64, 37 float64**; the consolidated panel silently upcasts *all* volume
to float64. `_standardise` coerces prices with `to_numeric` but doesn't enforce an integer volume
dtype, so any ticker that ever had a NaN-volume row during the batch fetch is now float. *Why it
matters:* float volume compares oddly (`== 0` on a float is fragile) and wastes space. *Fix:*
after dropping bad rows, cast `volume` to a nullable `Int64` consistently.

**M7. Duplicate-date resolution is arbitrary.**
*File:* `download.py:77`. `drop_duplicates(subset="date")` keeps the *first* occurrence with no
rule. If a batch ever returns a corrected + stale bar for the same date, the wrong one may win
silently. Low probability with yfinance, but undocumented. *Fix:* keep last, or assert
uniqueness and log.

**M8. `end_date` uses local `date.today()`, not the project's `America/New_York` tz.**
*File:* `config.py:60`. On a machine west of NY, "today" can resolve to a date the US market
hasn't reached, requesting a not-yet-existent bar. Minor here, worth a `tz`-aware `now`.

**M9. Reference calendar is derived from the data, so it can't reveal a market-wide missing day.**
*File:* `integrity.py:35-51`. Using `^GSPC`'s own dates as the "expected" calendar means a day
missing from *every* ticker (including the benchmark) is invisible. Acceptable for now, but note
it: coverage is measured against the data, not an independent NYSE calendar (e.g.
`pandas_market_calendars`).

---

## Correctness bugs found (with reproduction)

1. **C2 (adj_close silently dropped)** — reproduction: `pd.read_parquet('data/raw/AAPL.parquet').columns`
   → no `adj_close`; the rename dict entry `{"adj_close": "adj_close"}` in `download.py:60` is a
   no-op and `_STD_COLS` never lists it. This is a functional bug against the stated schema.

2. **M5 (deprecated API)** — reproduction: under `warnings.simplefilter('error')`,
   `pd.Timestamp.utcnow()` raises `Pandas4Warning`.

3. **M6 (dtype drift)** — reproduction: iterate `data/raw/*.parquet` reading only `volume`;
   dtype counts = `{int64: 467, float64: 37}`. The panel's volume column is `float64`.

4. **Coverage-metric semantic bug (C1/M4)** — reproduction: recompute non-flat rows per file;
   SW = 1476/2512 real (41% flat), AMCR = 1959/2512 (22% flat), yet both `coverage == 1.0,
   status == warn`. The metric's definition, not a typo, is the bug.

No crash-level bugs were found in the happy path; the download/standardise/integrity code runs
correctly and the tests pass (5/5). The bugs above are **semantic/data-fidelity** bugs, which are
the ones that hurt a quant pipeline most.

---

## Recommendations for the cleaning step (Week 1.2) — MOST IMPORTANT

The spec says to *detect and correct* >20% daily moves. **Do not
implement that literally.** Verified reality of this dataset: `max_abs_return` has a median of
**0.19** and mean **0.21** across 504 names — i.e. a ~20% single-day move is the *typical annual
extreme*, not an anomaly. CVNA has 34 such days, PCG's max is +75% (post-bankruptcy), SMCI 20
days. **These are almost all real** (earnings gaps, biotech readouts, bankruptcy/spin events,
COVID crash). Blindly "correcting" them fabricates data and deletes exactly the signal a quant
strategy trades on.

Adopt this philosophy and priority order:

1. **Flag, never silently mutate raw.** Cleaning writes to `data/interim/` / `data/processed`;
   `data/raw/` stays immutable. Keep an auditable `adjustments_log` of every cell changed and why.

2. **Distinguish *data errors* from *real moves* before touching anything.** A >20% move is a
   data error only if corroborated:
   - **Unadjusted split** — the classic false outlier. Detect by checking whether the move is
     near a clean ratio (2:1, 3:1, 3:2…) and *reverses* the next day, or better, cross-check against
     retained split/dividend actions (which C2 says you must start storing). If it's an
     un-applied split, *re-adjust*, don't clip.
   - **Bad single tick / fat-finger** — an isolated spike that round-trips within a day (high or
     low far outside a robust band, close back near prior). Cross-source check vs Stooq for that
     one bar (this is the *real* use for the fallback source).
   - **Everything else that survives corroboration is a REAL move — keep it.**

3. **For modeling, winsorize/robust-scale returns — don't edit prices.** If a model needs tamer
   tails, clip the *return feature* (e.g. at the 1st/99th pct or ±3 robust-MAD) as a **separate
   feature column**, leaving the price series intact. Never overwrite a real 40% biotech pop in
   the price panel.

4. **Handle the flat/zero-volume placeholder rows (C1) explicitly — this is the biggest cleaning
   task, bigger than outliers.** For SW/AMCR/VRT and the halt-like set: do **not** forward-fill
   them into "prices." The correct move is to **truncate each series to its first real trading
   day** (`first_real_trading_date`) and drop interior zero-volume carried-forward bars, marking
   them `NaN` (halted) rather than as tradable prices. AMCR should start ~2019-06, SW ~2024, per
   the data.

5. **Forward-fill is dangerous — constrain it.** For a genuinely halted/suspended name, ffilling
   the last price **fabricates a tradable quote and creates look-ahead** (you "knew" a price on a
   day the stock couldn't trade). Rules: (a) never ffill across a delisting or before the first
   real bar; (b) ffill at most N consecutive days for a short halt, and mark those rows
   `is_filled=True` so downstream code can exclude them; (c) for return/label computation, treat
   filled bars as `NaN` returns, not zero returns.

6. **Missing-value strategy must respect the wide matrix's NaN structure.** Verified: the wide
   `close_prices.parquet` is **2.93% NaN**, all in **37 short-history columns**, and a naive
   `dropna(how='any')` keeps only **16 of 2512 rows**. So: **never** `dropna` rows on the wide
   panel, and **never** back-fill a pre-IPO NaN region into a fake price. Leave pre-listing NaNs
   as NaN; align on a per-ticker basis.

7. **Standardisation (the benign part of 1.2).** Symbol normalisation (`to_yahoo_symbol`) and date
   normalisation are already done in `_standardise`. The remaining real work is dtype consistency
   (M6) and a documented, timezone-safe date index.

---

## Prioritized action list

**Before writing any cleaning code (do first):**
1. Fix the coverage/integrity metric to expose fabricated rows: add `real_row_ratio`,
   `first_real_trading_date`, `min_history_days` gate; promote SW/AMCR/HONA/FDXF to a
   `fail`/quarantine tier (C1, M3, M4). Re-run `03`. The current "0 fail" is not trustworthy.
2. Decide the adjusted-vs-raw question and **re-download with `auto_adjust=False`** to retain raw
   OHLC + `adj_close` + split/dividend actions (C2). Everything in cleaning (split detection,
   VWAP, realistic fills) depends on this. Fix README/config to stop claiming raw is retained.
3. Reframe the Week-1.2 outlier task from "correct >20% moves" to "flag & corroborate; winsorize
   features, not prices" (cleaning section). Write it down before coding.

**Cheap fixes to batch in now:**
4. `Timestamp.utcnow()` → `Timestamp.now('UTC')` (M5).
5. Enforce `Int64` volume in `_standardise` (M6).
6. Pin exact dep versions and write a `data_snapshot.json` with per-file sha256 + fetch UTC +
   yfinance version (M2).

**After cleaning, before indicators (1.3):**
7. Add reconciliation: sample-check yfinance vs Stooq (adjustment-basis aware) for a handful of
   names to validate the primary source (M1).
8. Consider an independent NYSE calendar (`pandas_market_calendars`) so coverage can detect a
   market-wide missing day (M9).
9. Add tests for the new cleaning transforms (split re-adjustment, flat-row truncation,
   ffill-with-mask) on synthetic fixtures — the current suite covers acquisition only.
