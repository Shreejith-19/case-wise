"""Access to the exported IndiaStatRAG scoring kit (benchmark_kit/, pinned commit b45befa).

The kit is the *official* scorer. We never reimplement metrics; we only call it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from isr import CORPUS_DIR, KIT_DIR

if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

from benchmark import predictions as kit_predictions  # noqa: E402
from benchmark import score as kit_score  # noqa: E402
from benchmark import splits as kit_splits  # noqa: E402
from benchmark.questions import load_questions  # noqa: E402

SPLIT_FILE = KIT_DIR / "benchmark" / "data" / "splits" / "split-s20260923.json"
BUDGETS: tuple[int, ...] = tuple(kit_score.DEFAULT_BUDGETS)
HEADLINE_BUDGET = 8000
TAU = kit_score.DEFAULT_TAU

__all__ = ["kit_predictions", "kit_score", "kit_splits", "questions_for", "BUDGETS",
           "HEADLINE_BUDGET", "TAU", "SPLIT_FILE", "CORPUS_DIR"]


def questions_for(split: str) -> list[dict]:
    """Verified questions of one split: 'dev', 'test' or 'all'."""
    verified = load_questions(verified_only=True)
    if split == "all":
        return verified
    return kit_splits.select(verified, SPLIT_FILE, split)


def load_predictions(path: Path, questions: Sequence[dict], budgets: Sequence[int] = BUDGETS) -> dict:
    """Strictly validated predictions file (the kit's own loader)."""
    return kit_predictions.load(Path(path), questions, budgets)
