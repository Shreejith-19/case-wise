"""Phase 6 runner: E (agentic upper bound) on a DEV split. Pausable (PAUSE file); resumes via the LLM cache.

    python scripts/run_e.py configs/E.yaml --split dev_tune [--limit 10] [--compare runs/C-full/pred_dev_tune.json]

Stage 2 (GPU): frozen C-full retrieval (rule / threshold abstention, initial results, fallback ranking).
Stage 3 (LLM on GPU, search on CPU): agent episodes (isr.agent). Final ranking for the retrieval metrics =
provisions the agent cited, then the ones it read, then the C-full ranking. test is refused: it is spent (D6/D13).
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import yaml

from isr import RUNS_DIR, pause
from isr.eval import evaluate as ev
from isr.eval.kit import questions_for
from isr.eval.predictions import write_predictions
from isr.pack import pack
from isr.pipeline import Pipeline

BUDGETS = (2000, 8000, 32000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--split", required=True, choices=["dev_tune", "dev_holdout"])
    ap.add_argument("--limit", type=int, help="first N questions only (plumbing check; not evaluated)")
    ap.add_argument("--compare", type=Path)
    ap.add_argument("--test-note")
    args = ap.parse_args()
    if args.split == "dev_holdout" and not args.test_note:
        ap.error("dev_holdout requires --test-note")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    qs = questions_for(args.split)[: args.limit] if args.limit else questions_for(args.split)
    out_dir = RUNS_DIR / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    timing: dict = {}

    t0 = time.time()
    pipe = Pipeline(cfg)
    ranked_all = {q["question_id"]: pipe.rank(q["question"]) for q in qs}
    pipe.close_gpu()
    del pipe.dense, pipe.reranker
    gc.collect()
    timing["stage2_s"] = round(time.time() - t0, 1)

    from isr.agent import Agent, CpuSearch
    from isr.llm.client import LLM
    from isr.llm.stages import ground_quotes
    from isr.xref import XrefGraph
    t0 = time.time()
    store = pipe.store
    llm = LLM(cfg["agent"].get("model", "qwen3:8b"), num_ctx=cfg["agent"].get("num_ctx", 12288))
    agent = Agent(llm, CpuSearch(store), XrefGraph())
    timing["tools_load_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    run, answers, traces = {}, {}, {}
    for n, q in enumerate(qs, 1):
        pause.check(f"question {n}/{len(qs)} (agent LLM replies so far are cached)")
        qid = q["question_id"]
        ranked, tr = ranked_all[qid]
        t = {"abstain_reason": tr.abstain_reason}
        if tr.abstain_reason:
            run[qid] = {b: [] for b in BUDGETS}
            answers[qid] = {"text": "", "abstained": True, "citations": []}
            traces[qid] = t
            continue
        ep = agent.run(q["question"], ranked[:30])
        a = ep.answer or {"text": "", "citations": [], "abstain": True}
        t.update(steps=ep.steps, read=ep.read, cited=a["citations"], agent_abstain=a["abstain"])
        if a["abstain"]:
            t["abstain_reason"] = "agent"
            run[qid] = {b: [] for b in BUDGETS}
            answers[qid] = {"text": "", "abstained": True, "citations": []}
        else:
            order = list(dict.fromkeys([*a["citations"], *ep.read, *ranked]))
            run[qid] = {b: pack(order, b, store, min_piece=cfg.get("pack", {}).get("min_piece", 200)) for b in BUDGETS}
            cited = [u for u in a["citations"] if store.kind[u] == "provision"]
            text = ground_quotes(a["text"], [store.text(u) for u in cited])
            answers[qid] = {"text": text, "abstained": not text.strip(),
                            "citations": [{"document_id": store.doc[u], "key": str(store.units.at[u, "key"])}
                                          for u in dict.fromkeys(cited)]}
        traces[qid] = t
        if n % 10 == 0:
            print(f"  agent {n}/{len(qs)} ({time.time() - t0:.0f}s, {llm.stats['cached']}/{llm.stats['calls']} cached)",
                  flush=True)
    llm.unload()
    timing["agent_s"] = round(time.time() - t0, 1)
    timing["llm_stats"] = llm.stats
    steps = [len(t.get("steps", [])) for t in traces.values() if "steps" in t]
    timing["mean_steps"] = round(sum(steps) / max(1, len(steps)), 2)
    suffix = f"_first{args.limit}" if args.limit else ""
    (out_dir / f"trace_{args.split}{suffix}.json").write_text(json.dumps(traces, ensure_ascii=False, default=int),
                                                             encoding="utf-8")
    pred = write_predictions(out_dir / f"pred_{args.split}{suffix}.json", system=cfg["name"], split=args.split,
                             run=run, answers=answers, meta={"config": cfg, "timing": timing})
    print(f"{len(qs)} questions; timing {json.dumps(timing)} -> {pred}")
    if args.limit:
        return 0
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
