"""Point-in-time S&P 500 membership — the survivorship-bias fix.

The current constituent list only tells you who is in the index *today*. To
study a 10-year window without survivorship bias you need who was in the index
*on each historical date*, including names since delisted or removed.

This reconstructs that from Wikipedia's "Selected changes to the S&P 500" table
(effective date, added ticker, removed ticker) by walking backwards from today's
members and reversing each change. It is a free, best-effort approximation:
Wikipedia is not point-in-time authoritative (CRSP/Norgate are), exact effective
dates and ticker renames can be imperfect, and only changes Wikipedia lists are
captured. Good enough to *include* delisted names and build a far less biased
universe; documented as approximate.
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from .universe import _HEADERS, WIKI_URL, get_sp500_constituents, to_yahoo_symbol


class MembershipHistory:
    def __init__(self, current: set[str], changes: pd.DataFrame):
        self.current = current                      # yahoo-formatted current members
        changes = changes.copy()
        changes["date"] = pd.to_datetime(changes["date"], errors="coerce")
        self.changes = changes.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_wikipedia(cls, cfg, force_refresh: bool = False) -> MembershipHistory:
        cur_df = get_sp500_constituents(cfg, force_refresh=force_refresh)
        current = {to_yahoo_symbol(s) for s in cur_df["yahoo_symbol"]}

        cache = cfg.resolve("data/metadata/sp500_changes.csv")
        if cache.exists() and not force_refresh:
            changes = pd.read_csv(cache, parse_dates=["date"])
        else:
            changes = cls._fetch_changes()
            changes.to_csv(cache, index=False)
        return cls(current, changes)

    @staticmethod
    def _fetch_changes() -> pd.DataFrame:
        resp = requests.get(WIKI_URL, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        raw = pd.read_html(io.StringIO(resp.text))[1]
        raw.columns = ["date", "add_tkr", "add_sec", "rem_tkr", "rem_sec", "reason"]
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        for c in ("add_tkr", "rem_tkr"):
            raw[c] = raw[c].map(lambda x: to_yahoo_symbol(str(x)) if pd.notna(x) and str(x).strip() else None)
        return raw.dropna(subset=["date"])[["date", "add_tkr", "add_sec", "rem_tkr", "rem_sec", "reason"]]

    # -- queries -----------------------------------------------------------
    def members_on(self, date) -> set[str]:
        """Index membership on a past date, by reversing all later changes."""
        date = pd.Timestamp(date)
        members = set(self.current)
        # Reverse changes strictly after `date`, newest first.
        later = self.changes[self.changes["date"] > date].sort_values("date", ascending=False)
        for _, ch in later.iterrows():
            a, r = ch["add_tkr"], ch["rem_tkr"]
            if pd.notna(a) and a:             # added later -> not a member before
                members.discard(a)
            if pd.notna(r) and r:             # removed later -> was a member before
                members.add(r)
        return members

    def universe(self, start, end) -> list[str]:
        """Survivorship-free ticker set: everyone who was a member at any point
        in [start, end] — current members, plus names added or removed within."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        members = self.members_on(start)
        window = self.changes[(self.changes["date"] >= start) & (self.changes["date"] <= end)]
        members |= set(window["add_tkr"].dropna())
        members |= set(window["rem_tkr"].dropna())
        return sorted(members)

    def removed_in(self, start, end) -> pd.DataFrame:
        """Names removed from the index within the window (the survivorship gap)."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        w = self.changes[(self.changes["date"] >= start) & (self.changes["date"] <= end)]
        rem = w[w["rem_tkr"].notna()][["date", "rem_tkr", "rem_sec", "reason"]]
        rem = rem.rename(columns={"rem_tkr": "ticker", "rem_sec": "security"})
        # Keep the latest removal per ticker.
        return rem.sort_values("date").drop_duplicates("ticker", keep="last").reset_index(drop=True)

    def write_reports(self, cfg) -> dict:
        """Persist the survivorship-free universe, removed names, and yearly
        point-in-time snapshots to data/metadata/."""
        meta = cfg.path("metadata")
        start, end = cfg.start_date, cfg.end_date
        uni = self.universe(start, end)
        removed = self.removed_in(start, end)

        pd.DataFrame({"ticker": uni,
                      "status": ["current" if t in self.current else "removed" for t in uni]}
                     ).to_csv(meta / "sp500_membership_universe.csv", index=False)
        removed.to_csv(meta / "sp500_removed.csv", index=False)

        # Membership count at the start of each year — shows index turnover.
        snaps = []
        for yr in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
            d = pd.Timestamp(f"{yr}-01-03")
            if pd.Timestamp(start) <= d <= pd.Timestamp(end):
                mem = self.members_on(d)
                snaps.append({"date": d.date(), "n_members": len(mem),
                              "still_current": len(mem & self.current)})
        pd.DataFrame(snaps).to_csv(meta / "membership_snapshots.csv", index=False)
        return {"universe": len(uni), "current": len(self.current), "removed": len(removed)}


def download_universe(cfg) -> list[str]:
    """Ticker list for the downloader. Survivorship-free (current + delisted in
    window) when ``universe.survivorship_free`` is set, else current snapshot.
    Always appends configured extra tickers (e.g. the ^GSPC benchmark)."""
    from .universe import get_ticker_list

    if not cfg["universe"].get("survivorship_free"):
        return get_ticker_list(cfg)

    hist = MembershipHistory.from_wikipedia(cfg)
    tickers = hist.universe(cfg.start_date, cfg.end_date)
    tickers += list(cfg["universe"].get("extra_tickers", []))
    seen: set[str] = set()
    return [t for t in tickers if not (t in seen or seen.add(t))]
