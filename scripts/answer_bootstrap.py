"""Paired bootstrap of per-question answer metrics (official kit score_answer) between two systems.

    python scripts/answer_bootstrap.py runs/B6-Afull-answers runs/B8-answers-v2 [more ...] --split dev_tune

Each later system is compared with the first. Answerable questions only; a missing citation precision (no
citations) counts as 0. 95% percentile CI, 5000 resamples, seed 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from isr.eval.kit import kit_score, questions_for

METRICS = ("fact_coverage", "citation_recall", "citation_precision")


def per_question(run_dir: Path, split: str, qs: list[dict]) -> dict[str, dict]:
    answers = json.loads((run_dir / f"pred_{split}.json").read_text(encoding="utf-8"))["answers"]
    return {q["question_id"]: kit_score.score_answer(q, answers.get(q["question_id"])) for q in qs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--split", default="dev_tune")
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()
    qs = [q for q in questions_for(args.split) if not q["unanswerable"]]
    ids = [q["question_id"] for q in qs]
    base = per_question(args.runs[0], args.split, qs)
    for other in args.runs[1:]:
        s = per_question(other, args.split, qs)
        rng = np.random.default_rng(0)
        print(f"\n{other.name} vs {args.runs[0].name}  (n={len(ids)} answerable, {args.split})")
        for m in METRICS:
            x = np.array([base[i].get(m) or 0.0 for i in ids])
            y = np.array([s[i].get(m) or 0.0 for i in ids])
            d = y - x
            boots = d[rng.integers(0, len(d), (args.n, len(d)))].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            sig = "*" if lo > 0 or hi < 0 else " "
            print(f"  {m:<20} {y.mean():.3f} vs {x.mean():.3f}  d={d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]{sig} "
                  f"wins/losses {int((d > 0).sum())}/{int((d < 0).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
