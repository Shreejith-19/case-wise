"""Phase 2.8 / 3.4: tune the reranker-score abstention threshold on DEV traces only.

    python scripts/tune_abstention.py runs/<system>/trace_dev.json [--lam 2.0]

Objective (fixed in advance, plan.md Phase 3.4): abstention_rate - lam * false_abstention.
Questions already abstained by deterministic rules keep that decision; questions whose
named provision was pinned by an exact lookup are never threshold-abstained (as in the
pipeline). Prints the curve and the chosen threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from isr.eval.kit import questions_for


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("trace", type=Path)
    p.add_argument("--lam", type=float, default=2.0)
    p.add_argument("--max-false", type=float, default=0.05)
    args = p.parse_args()
    traces = json.loads(args.trace.read_text(encoding="utf-8"))
    qs = {q["question_id"]: q for q in questions_for("dev")}
    rows = []
    for qid, t in traces.items():
        pinned = bool(t["lists"].get("exact_ref"))
        rows.append((qs[qid]["unanswerable"], t["abstain_reason"] not in (None, "low_rerank_score"), pinned,
                     t["rerank_top"] if t["rerank_top"] is not None else -1e9))
    n_u = sum(r[0] for r in rows)
    n_a = len(rows) - n_u
    cands = sorted({r[3] for r in rows if r[3] > -1e9})
    grid = [-1e9] + [(a + b) / 2 for a, b in zip(cands, cands[1:])]
    best = None
    print(f"{'threshold':>10} {'abst':>6} {'false':>6} {'objective':>9}")
    for thr in grid:
        ab_u = ab_a = 0
        for unans, ruled, pinned, top in rows:
            abst = ruled or (not pinned and top < thr)
            ab_u += unans and abst
            ab_a += (not unans) and abst
        rate, false = ab_u / n_u, ab_a / n_a
        obj = rate - args.lam * false
        if false <= args.max_false and (best is None or obj > best[0] + 1e-12):
            best = (obj, thr, rate, false)
        if thr == -1e9 or abs(obj - best[0]) < 1e-9 or ab_a <= 3:
            print(f"{thr:10.3f} {rate:6.3f} {false:6.3f} {obj:9.3f}")
    print(f"\nchosen threshold {best[1]:.3f}: abstention {best[2]:.3f}, false abstention {best[3]:.3f} "
          f"(lam={args.lam}; {n_u} unanswerable / {n_a} answerable dev questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
