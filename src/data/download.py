"""Resumable OHLCV downloader.

``Downloader`` orchestrates a primary ``PriceSource`` with per-ticker fallbacks:
batched fetch with retries on the still-empty subset, then any remaining empties
are retried through the fallback sources one ticker at a time. Each ticker is
stored as its own Parquet file (via :class:`ParquetStore`) so runs are resumable
and individual names can be re-fetched. A manifest of per-ticker outcomes is
written every run.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import pandas as pd

from ..storage import ParquetStore
from ..utils import get_logger
from .sources import STD_COLS, PriceSource, StooqSource, YahooSource


@dataclass
class TickerResult:
    ticker: str
    status: str          # ok | failed | skipped
    source: str
    rows: int
    start: str
    end: str
    error: str = ""


class Downloader:
    def __init__(self, cfg, sources: list[PriceSource] | None = None,
                 store: ParquetStore | None = None):
        self.cfg = cfg
        self.log = get_logger("download", cfg.path("logs"))
        self.store = store or ParquetStore(cfg.path("raw"), cfg["storage"]["compression"])
        self.sources = sources or self._default_sources()

    def _default_sources(self) -> list[PriceSource]:
        by_name = {"yfinance": YahooSource(int(self.cfg["download"]["timeout"])),
                   "stooq": StooqSource()}
        order = [self.cfg["data"]["source"], self.cfg["data"].get("fallback_source")]
        return [by_name[n] for n in order if n in by_name]

    # -- public API --------------------------------------------------------
    def run(self, tickers: list[str], force: bool = False) -> pd.DataFrame:
        start, end = self.cfg.start_date, self.cfg.end_date
        d = self.cfg["download"]

        todo, results = self._resume(tickers, force)
        self.log.info("Universe: %d | to download: %d | cached: %d",
                      len(tickers), len(todo), len(tickers) - len(todo))
        self.log.info("Range: %s -> %s (adjusted=%s)", start, end, self.cfg["data"]["auto_adjust"])

        size = int(d["batch_size"])
        batches = [todo[i:i + size] for i in range(0, len(todo), size)]
        for bi, batch in enumerate(batches, 1):
            self.log.info("Batch %d/%d (%d tickers)...", bi, len(batches), len(batch))
            frames = self._fetch_batch(batch, start, end)
            results.extend(self._save_batch(batch, frames))
            if bi < len(batches):
                time.sleep(float(d["pause_between_batches"]))

        return self._write_manifest(results)

    # -- internals ---------------------------------------------------------
    def _resume(self, tickers, force):
        todo, results = [], []
        for t in tickers:
            if not force and self.store.exists(t):
                ex = self.store.read(t)
                results.append(TickerResult(
                    t, "skipped", str(ex["source"].iloc[0]) if len(ex) else "",
                    len(ex), str(ex["date"].min().date()) if len(ex) else "",
                    str(ex["date"].max().date()) if len(ex) else ""))
            else:
                todo.append(t)
        return todo, results

    def _fetch_batch(self, batch, start, end) -> dict[str, pd.DataFrame]:
        """Primary source with retries, then fallbacks for still-empty tickers."""
        d = self.cfg["download"]
        kw = dict(interval=self.cfg["data"]["interval"],
                  auto_adjust=self.cfg["data"]["auto_adjust"])
        frames: dict[str, pd.DataFrame] = {}
        remaining = list(batch)

        for attempt in range(1, int(d["max_retries"]) + 1):
            try:
                got = self.sources[0].fetch(remaining, start, end, **kw)
            except Exception as exc:  # whole-batch failure (e.g. 429)
                self.log.warning("  %s attempt %d error: %s", self.sources[0].name, attempt, exc)
                got = {}
            frames.update({t: df for t, df in got.items() if not df.empty})
            remaining = [t for t in remaining if t not in frames]
            if not remaining:
                return frames
            if attempt < int(d["max_retries"]):
                wait = float(d["pause_between_retries"]) * attempt
                self.log.info("  %d empty; retry %d in %.1fs", len(remaining), attempt + 1, wait)
                time.sleep(wait)

        for src in self.sources[1:]:
            if not remaining:
                break
            self.log.info("  fallback(%s): %s", src.name, ", ".join(remaining))
            got = src.fetch(remaining, start, end, **kw)
            frames.update({t: df for t, df in got.items() if not df.empty})
            remaining = [t for t in remaining if t not in frames]

        for t in remaining:
            frames.setdefault(t, pd.DataFrame(columns=STD_COLS))
        return frames

    def _save_batch(self, batch, frames) -> list[TickerResult]:
        out = []
        for t in batch:
            df = frames.get(t, pd.DataFrame(columns=STD_COLS))
            if df.empty:
                out.append(TickerResult(t, "failed", "", 0, "", "", "no data from any source"))
                self.log.warning("  FAILED: %s", t)
            else:
                self.store.write(df, t)
                out.append(TickerResult(t, "ok", str(df["source"].iloc[0]), len(df),
                                        str(df["date"].min().date()), str(df["date"].max().date())))
        return out

    def _write_manifest(self, results) -> pd.DataFrame:
        manifest = pd.DataFrame([asdict(r) for r in results])
        path = self.cfg.path("metadata") / "download_manifest.csv"
        manifest.to_csv(path, index=False)
        counts = manifest["status"].value_counts().to_dict()
        self.log.info("Done. %s -> %s", counts, path)
        failed = manifest.loc[manifest.status == "failed", "ticker"].tolist()
        if failed:
            self.log.warning("Failed tickers: %s", ", ".join(failed))
        return manifest


def download_ohlcv(cfg, tickers: list[str], force: bool = False) -> pd.DataFrame:
    """Functional entry point used by the pipeline scripts."""
    return Downloader(cfg).run(tickers, force=force)
