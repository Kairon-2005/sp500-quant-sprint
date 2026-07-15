"""Offline tests for point-in-time membership reconstruction (no network)."""
import pandas as pd

from src.data.membership import MembershipHistory


def _changes(rows):
    cols = ["date", "add_tkr", "add_sec", "rem_tkr", "rem_sec", "reason"]
    return pd.DataFrame([dict(zip(cols, r, strict=True)) for r in rows])


def test_members_on_reverses_changes():
    # Today: {A, B, C}. On 2020-06-01, C was added and X removed.
    hist = MembershipHistory(
        current={"A", "B", "C"},
        changes=_changes([("2020-06-01", "C", "C Corp", "X", "X Corp", "swap")]),
    )
    # Before the change, C was not yet a member and X still was.
    assert hist.members_on("2020-01-01") == {"A", "B", "X"}
    # After the change, membership matches today.
    assert hist.members_on("2021-01-01") == {"A", "B", "C"}


def test_universe_is_survivorship_free():
    hist = MembershipHistory(
        current={"A", "B", "C"},
        changes=_changes([("2020-06-01", "C", "C Corp", "X", "X Corp", "swap")]),
    )
    # The window universe includes the since-removed name X.
    uni = hist.universe("2019-01-01", "2021-01-01")
    assert set(uni) == {"A", "B", "C", "X"}


def test_removed_in_lists_delisted():
    hist = MembershipHistory(
        current={"A", "B"},
        changes=_changes([
            ("2018-03-01", None, None, "Y", "Y Corp", "acquired"),
            ("2020-06-01", "C", "C Corp", "X", "X Corp", "swap"),
        ]),
    )
    rem = hist.removed_in("2019-01-01", "2021-01-01")
    assert set(rem["ticker"]) == {"X"}          # Y removed before window
