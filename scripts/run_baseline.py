"""Run a shipped baseline (oracle | bm25-windows | bm25-two-stage) and write a predictions file.

    python scripts/run_baseline.py bm25-windows --split dev
    -> runs/baselines/bm25-windows_dev.json   (then: python -m isr.eval.evaluate <file> --split dev)
"""

from __future__ import annotations

import argparse
import time

from isr import CORPUS_DIR, INDEX_DIR, RUNS_DIR
from isr.eval.kit import BUDGETS, kit_score, questions_for
from isr.eval.predictions import write_predictions

DEFAULT_INDEX = INDEX_DIR / "bm25_baseline.sqlite"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("system", choices=["oracle", "bm25-windows", "bm25-two-stage"])
    p.add_argument("--split", required=True, choices=["dev", "test"])
    p.add_argument("--index", default=str(DEFAULT_INDEX))
    args = p.parse_args()

    from benchmark import baselines  # kit, on sys.path via isr.eval.kit

    questions = questions_for(args.split)
    system = baselines.load(args.system, corpus_dir=CORPUS_DIR, questions=questions,
                            index_path=args.index)
    t0 = time.time()
    run = kit_score.run_system(system, questions, BUDGETS, progress=True)
    secs = time.time() - t0
    out = write_predictions(RUNS_DIR / "baselines" / f"{args.system}_{args.split}.json",
                            system=system.name, split=args.split, run=run,
                            meta={"seconds": round(secs, 1), "questions": len(questions)})
    print(f"{len(questions)} questions in {secs:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
