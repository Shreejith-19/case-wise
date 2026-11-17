"""Phase 3.0: fixed, category-stratified split of dev into dev_tune (100) / dev_holdout (50).

    python scripts/make_dev_split.py      -> configs/dev_split.json  (tracked; never regenerated)

From Phase 3 on, every prompt / setting is chosen on dev_tune only. dev_holdout is scored only for
frozen configs (logged to HOLDOUT_RUNS.md), as an overfitting check while test stays sealed (D13).
Note: Phase 2 (Plan A) was tuned on all of dev, so dev_holdout is clean only for Phase 3+ decisions.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict

from isr import ROOT
from isr.eval.kit import questions_for

OUT = ROOT / "configs" / "dev_split.json"
SEED = 20261117
HOLDOUT_FRACTION = 1 / 3


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; the split is fixed once made")
    by_cat = defaultdict(list)
    for q in questions_for("dev"):
        by_cat[q["category"]].append(q["question_id"])
    rng = random.Random(SEED)
    tune, hold = [], []
    for cat in sorted(by_cat):
        ids = sorted(by_cat[cat])
        rng.shuffle(ids)
        k = round(len(ids) * HOLDOUT_FRACTION)
        hold += ids[:k]
        tune += ids[k:]
    OUT.write_text(json.dumps({"seed": SEED, "source_split": "dev", "dev_tune": sorted(tune),
                               "dev_holdout": sorted(hold),
                               "counts": {c: [len(v) - round(len(v) * HOLDOUT_FRACTION),
                                              round(len(v) * HOLDOUT_FRACTION)] for c, v in sorted(by_cat.items())}},
                              indent=1), encoding="utf-8")
    print(f"dev_tune {len(tune)} / dev_holdout {len(hold)} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
