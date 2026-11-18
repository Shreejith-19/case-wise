"""Phase 6 upper bound E-oracle (DEV ONLY): B-full's answer stage (qwen3:8b, answer-v1) given the GOLD provisions.

    python scripts/oracle_answers.py --split dev_tune [--compare runs/C-full/pred_dev_tune.json]

Context = the first location of every gold evidence group (full provision text, <= 2500 chars each, same
formatting as B-full), so the answer metrics measure the generator's ceiling with perfect retrieval.
Unanswerable questions are abstained (an oracle knows they have no evidence). Retrieval spans = the gold spans.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isr import RUNS_DIR, pause
from isr.eval import evaluate as ev
from isr.eval.kit import questions_for
from isr.eval.predictions import write_predictions
from isr.llm import stages as S
from isr.llm.client import LLM
from isr.store import UnitStore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_b import fmt_passages  # noqa: E402

NAME = "E-oracle-answers"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev_tune", "dev_holdout"])
    ap.add_argument("--compare", type=Path)
    ap.add_argument("--test-note")
    args = ap.parse_args()
    if args.split == "dev_holdout" and not args.test_note:
        ap.error("dev_holdout requires --test-note")
    store = UnitStore()
    llm = LLM("qwen3:8b")
    run, answers = {}, {}
    qs = questions_for(args.split)
    for n, q in enumerate(qs, 1):
        pause.check(f"question {n}/{len(qs)}")
        qid = q["question_id"]
        if q["unanswerable"]:
            run[qid] = {b: [] for b in (2000, 8000, 32000)}
            answers[qid] = {"text": "", "abstained": True, "citations": []}
            continue
        locs = [g["locations"][0] for g in q["gold_evidence"]]
        spans = [(l["document_id"], l["char_start"], l["char_end"]) for l in locs]
        run[qid] = {b: spans for b in (2000, 8000, 32000)}
        ps = [{"document_id": l["document_id"], "key": l["key"], "title": store.title(l["document_id"]),
               "label": l["key"], "text": store.doc_text(l["document_id"])[l["char_start"]:l["char_end"]][:2500]}
              for l in locs]
        o = S.answer(llm, q["question"], fmt_passages(ps))
        cited = [ps[i - 1] for i in o["citations"] if 1 <= i <= len(ps)]
        text = S.ground_quotes(o["answer"], [p["text"] for p in cited])
        answers[qid] = {"text": "" if o["abstain"] else text, "abstained": o["abstain"] or not text.strip(),
                        "citations": [{"document_id": p["document_id"], "key": p["key"]} for p in cited]}
    llm.unload()
    pred = write_predictions(RUNS_DIR / NAME / f"pred_{args.split}.json", system=NAME, split=args.split, run=run,
                             answers=answers, meta={"llm_stats": llm.stats})
    print(f"{len(qs)} questions -> {pred}")
    argv = [str(pred), "--split", args.split]
    if args.compare:
        argv += ["--compare", str(args.compare)]
    if args.test_note:
        argv += ["--test-note", args.test_note]
    return ev.main(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except pause.Paused as e:
        print(e)
        raise SystemExit(pause.EXIT_PAUSED)
