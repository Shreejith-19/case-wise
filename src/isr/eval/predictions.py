"""Write predictions files in the exact format the official scorer loads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from isr.eval.kit import BUDGETS

Span = tuple[str, int, int]


def write_predictions(path: Path, *, system: str, split: str,
                      run: Mapping[str, Mapping[int, Sequence[Span]]],
                      answers: Optional[Mapping[str, dict]] = None,
                      budgets: Sequence[int] = BUDGETS, meta: Optional[dict] = None) -> Path:
    """``run`` maps question_id -> budget -> ranked spans (best first). Empty list = abstain."""
    predictions = {}
    for qid, by_budget in run.items():
        row = {}
        for b in budgets:
            spans = by_budget.get(b, by_budget.get(str(b)))  # type: ignore[arg-type]
            if spans is None:
                raise ValueError(f"{qid}: no spans for budget {b}")
            row[str(b)] = [[d, int(s), int(e)] for d, s, e in spans]
        predictions[qid] = row
    payload: dict = {"system": system, "split": split, "budgets": list(budgets),
                     "predictions": predictions}
    if answers is not None:
        payload["answers"] = dict(answers)
    if meta:
        payload["meta"] = meta
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path
