"""Packer and fusion behaviour, checked against the official scorer's budget accounting."""

import numpy as np
import pytest

from isr.eval.kit import kit_score
from isr.pack import pack
from isr.pipeline import rrf


class FakeStore:
    """Minimal UnitStore stand-in: units given as (doc, start, end, kind, parent, class)."""

    def __init__(self, units):
        self.doc = np.array([u[0] for u in units], dtype=object)
        self.start = np.array([u[1] for u in units])
        self.end = np.array([u[2] for u in units])
        self.kind = np.array([u[3] for u in units], dtype=object)
        self.parent = np.array([u[4] for u in units])
        self.class_id = np.array([u[5] for u in units])

    def span(self, uid):
        return self.doc[uid], int(self.start[uid]), int(self.end[uid])


UNITS = [
    ("A", 0, 1000, "provision", -1, 0),       # 0
    ("B", 0, 1000, "provision", -1, 0),       # 1 same class as 0 (copy) -> skipped
    ("A", 1000, 6000, "provision", -1, 1),    # 2 long provision
    ("A", 3000, 4500, "subunit", 2, 2),       # 3 its sub-unit
    ("C", 0, 900, "window", -1, 3),           # 4
]


def _used(spans, budget):
    return kit_score.within_budget(spans, budget).used_chars


@pytest.mark.parametrize("budget", [500, 2000, 8000])
def test_never_exceeds_budget_and_matches_scorer(budget):
    store = FakeStore(UNITS)
    spans = pack([0, 1, 2, 3, 4], budget, store)
    assert _used(spans, budget) <= budget
    assert sum(e - s for _, s, e in spans) >= _used(spans, budget)


def test_duplicate_class_skipped():
    store = FakeStore(UNITS)
    spans = pack([0, 1, 4], 8000, store)
    assert ("B", 0, 1000) not in spans and ("A", 0, 1000) in spans


def test_long_provision_window_centres_on_ranked_subunit():
    store = FakeStore(UNITS)
    spans = pack([3, 2], 2000, store)          # sub-unit ranked first, then its parent
    assert spans[0] == ("A", 3000, 4500)
    # parent does not fit: remaining 500 chars must come from inside the parent
    did, s, e = spans[1]
    assert did == "A" and 1000 <= s and e <= 6000 and e - s == 500


def test_window_when_first_unit_too_long():
    store = FakeStore(UNITS)
    spans = pack([2, 3], 2000, store)          # parent first, sub-unit known
    did, s, e = spans[0]
    assert e - s == 2000 and s <= 3000 and e >= 4500   # window covers the best sub-unit


def test_min_piece_stops():
    store = FakeStore(UNITS)
    spans = pack([0, 4], 1100, store, min_piece=200)
    assert spans == [("A", 0, 1000)]          # 100 chars left < min_piece


def test_rrf_weights_and_ties():
    fused = rrf({"a": [1, 2, 3], "b": [3, 2]}, {"a": 1.0, "b": 1.0})
    assert fused[0] in (2, 3) and set(fused) == {1, 2, 3}
    assert rrf({"a": [1, 2], "b": [2, 1]}, {"a": 5.0, "b": 1.0})[0] == 1
