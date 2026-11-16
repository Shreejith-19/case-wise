"""Run a configured system on a split, write predictions, and score it (dev) with CIs.

    python scripts/run_system.py configs/A0.yaml --split dev [--compare runs/baselines/bm25-windows_dev.json]

Test split: only with --test-note (forwarded to the evaluator, logged to TEST_RUNS.md).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from isr import RUNS_DIR
from isr.eval import evaluate as ev
from isr.eval.kit import questions_for
from isr.eval.predictions import write_predictions
from isr.pipeline import Pipeline


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--split", required=True, choices=["dev", "test"])
    p.add_argument("--compare", type=Path)
    p.add_argument("--test-note")
    args = p.parse_args()
    if args.split == "test" and not args.test_note:
        p.error("test split requires --test-note (plan.md D6)")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pipe = Pipeline(config)
    questions = questions_for(args.split)
    t0 = time.time()
    run = {}
    for n, q in enumerate(questions, 1):
        run[q["question_id"]] = pipe.run(q["question"])
        if n % 50 == 0:
            print(f"  {n}/{len(questions)} ({time.time() - t0:.0f}s)", flush=True)
    secs = time.time() - t0
    out = write_predictions(RUNS_DIR / config["name"] / f"pred_{args.split}.json", system=config["name"],
                            split=args.split, run=run,
                            meta={"config": config, "seconds": round(secs, 1),
                                  "sec_per_question": round(secs / len(questions), 3)})
    print(f"{len(questions)} questions in {secs:.1f}s -> {out}")
    argv = [str(out), "--split", args.split]
    if args.compare:
        argv += ["--compare", str(args.compare)]
    if args.test_note:
        argv += ["--test-note", args.test_note]
    return ev.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
