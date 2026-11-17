"""Run a configured system on a split, write predictions, and score it (dev) with CIs.

    python scripts/run_system.py configs/A0.yaml --split dev [--compare runs/baselines/bm25-windows_dev.json]

test / dev_holdout: only with --test-note (forwarded to the evaluator, logged to TEST_RUNS.md /
HOLDOUT_RUNS.md). Phase 3+ development uses --split dev_tune.
"""

from __future__ import annotations

import argparse
import json
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
    p.add_argument("--split", required=True, choices=["dev", "test", "dev_tune", "dev_holdout"])
    p.add_argument("--compare", type=Path)
    p.add_argument("--test-note")
    args = p.parse_args()
    if args.split in ("test", "dev_holdout") and not args.test_note:
        p.error(f"{args.split} requires --test-note (plan.md D6 / D14)")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pipe = Pipeline(config)
    questions = questions_for(args.split)
    t0 = time.time()
    run, traces = {}, {}
    for n, q in enumerate(questions, 1):
        run[q["question_id"]], tr = pipe.run_traced(q["question"])
        traces[q["question_id"]] = {"abstain_reason": tr.abstain_reason, "rerank_top": tr.rerank_top,
                                    "lists": {k: v[:300] for k, v in tr.lists.items()}, "final": tr.final[:100]}
        if n % 50 == 0:
            print(f"  {n}/{len(questions)} ({time.time() - t0:.0f}s)", flush=True)
    secs = time.time() - t0
    trace_path = RUNS_DIR / config["name"] / f"trace_{args.split}.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(traces), encoding="utf-8")
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
