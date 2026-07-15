#!/usr/bin/env python
"""Step 9 — reconstruct point-in-time S&P 500 membership (survivorship-bias fix).

Builds the survivorship-free universe (current + delisted-in-window), the list
of removed names, and yearly membership snapshots. Run before step 02 when
`universe.survivorship_free` is on. Reports written to data/metadata/.
"""
import _bootstrap  # noqa: F401

from src.config import load_config
from src.data.membership import MembershipHistory


def main() -> None:
    cfg = load_config()
    hist = MembershipHistory.from_wikipedia(cfg)
    summary = hist.write_reports(cfg)

    print("=== Point-in-time membership ===")
    print(f"current constituents      : {summary['current']}")
    print(f"survivorship-free universe: {summary['universe']} "
          f"(+{summary['universe'] - summary['current']} delisted/removed names)")
    print(f"removed within window     : {summary['removed']}")

    rem = hist.removed_in(cfg.start_date, cfg.end_date)
    print("\nMost recent removals (previously missing from our data):")
    print(rem.tail(8)[["date", "ticker", "security", "reason"]].to_string(index=False))
    print("\nReports -> data/metadata/{sp500_membership_universe,sp500_removed,membership_snapshots}.csv")


if __name__ == "__main__":
    main()
