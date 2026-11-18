"""Plan B runner (plan.md Phase 3): Plan A pipeline + local LLM stages, staged to fit one 8 GB GPU.

    python scripts/run_b.py configs/B1.yaml --split dev_tune [--compare runs/A-full/pred_dev.json]

Stage 1  LLM (Ollama): B1 query analysis, B2 HyDE                       -> cached, model unloaded
Stage 2  GPU retrieval: Plan A + LLM lists, cross-encoder rerank          -> GPU models released
Stage 3  LLM: B3 pointwise rerank of the top-N, B4 abstention judge, B5 cited answers
All LLM calls are cached (data/cache/llm.sqlite), so ablations that share stages are cheap.
dev_holdout / test only with --test-note (logged).
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
from isr.eval.kit import HEADLINE_BUDGET, questions_for
from isr.eval.predictions import write_predictions
from isr.pack import pack
from isr.pipeline import Pipeline


def key_for_span(store, did: str, s: int, e: int) -> tuple[str | None, int | None]:
    """Structure key + unit id of the provision that overlaps the span most (None for gap windows)."""
    best, best_ov = None, 0
    for u in store.doc_units(did):
        if store.kind[u] != "provision":
            continue
        ov = min(e, store.end[u]) - max(s, store.start[u])
        if ov > best_ov:
            best, best_ov = int(u), ov
    if best is None:
        return None, None
    return str(store.units.at[best, "key"]), best


def passages_for(store, spans, max_items: int, max_chars: int) -> list[dict]:
    out = []
    for did, s, e in spans[:max_items]:
        key, uid = key_for_span(store, did, s, e)
        label = str(store.label[uid]) if uid is not None else "text"
        out.append({"document_id": did, "key": key, "title": store.title(did), "label": label,
                    "text": store.doc_text(did)[s:e][:max_chars]})
    return out


def fmt_passages(ps: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {p['title']} — {p['label']}\n{p['text']}" for i, p in enumerate(ps, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--split", required=True, choices=["dev", "dev_tune", "dev_holdout", "test"])
    ap.add_argument("--compare", type=Path)
    ap.add_argument("--test-note")
    args = ap.parse_args()
    if args.split in ("test", "dev_holdout") and not args.test_note:
        ap.error(f"{args.split} requires --test-note")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    lc = cfg.get("llm") or {}
    qs = questions_for(args.split)
    out_dir = RUNS_DIR / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}

    # ---- stage 1: LLM query analysis + HyDE --------------------------------------------------------
    from isr.llm.client import LLM
    from isr.llm import stages as S
    signals: dict[str, dict] = {}
    need_analysis = any(k in lc for k in ("act_bm25", "act_dense", "keywords_bm25", "jurisdiction"))
    if need_analysis or "hyde" in lc:
        t0 = time.time()
        llm = LLM(lc.get("model", "qwen3:8b"))
        for n, q in enumerate(qs, 1):
            pause.check(f"stage 1, question {n}/{len(qs)} (LLM replies so far are cached)")
            sig: dict = {}
            if need_analysis:
                a = S.analyse(llm, q["question"])
                sig.update(analysis=a, act_docs=S.resolve_laws(a["candidate_laws"] + a["named_laws"]),
                           keywords=a["keywords"][:8], jurisdiction=S.canon_jurisdiction(a["jurisdiction"]))
            if "hyde" in lc:
                sig["hyde"] = S.hyde(llm, q["question"])
            signals[q["question_id"]] = sig
            if n % 25 == 0:
                print(f"  stage1 {n}/{len(qs)} ({time.time() - t0:.0f}s, {llm.stats['cached']} cached)", flush=True)
        llm.unload()
        timing["stage1_s"] = round(time.time() - t0, 1)

    # ---- stage 2: retrieval on GPU ---------------------------------------------------------------
    t0 = time.time()
    pipe = Pipeline(cfg)
    ranked_all, traces = {}, {}
    for q in qs:
        ranked, tr = pipe.rank(q["question"], signals.get(q["question_id"]))
        ranked_all[q["question_id"]] = (ranked[:300], tr)
    pipe.close_gpu()
    gc.collect()
    timing["stage2_s"] = round(time.time() - t0, 1)
    store = pipe.store

    # ---- stage 3: LLM rerank, judge, answers ------------------------------------------------------
    t0 = time.time()
    rr, jd, an = lc.get("llm_rerank"), lc.get("judge"), cfg.get("answer")
    llm = LLM(lc.get("model", "qwen3:8b")) if (rr or jd) else None
    if an and an.get("mode") == "llm":      # the answer stage may use its own model / context size
        am, actx = an.get("model", lc.get("model", "qwen3:8b")), an.get("num_ctx", 8192)
        ans_llm = llm if (llm is not None and am == llm.model and actx == llm.num_ctx) else LLM(am, num_ctx=actx)
    else:
        ans_llm = None
    min_piece = cfg.get("pack", {}).get("min_piece", 200)
    run, answers = {}, {}
    for n, q in enumerate(qs, 1):
        pause.check(f"stage 3, question {n}/{len(qs)} (LLM replies so far are cached)")
        qid = q["question_id"]
        ranked, tr = ranked_all[qid]
        t = {"abstain_reason": tr.abstain_reason, "rerank_top": tr.rerank_top,
             "lists": {k: v[:300] for k, v in tr.lists.items()}}
        if qid in signals:
            t["signals"] = {k: v for k, v in signals[qid].items() if k != "act_docs"}
            t["signals"]["n_act_docs"] = len(signals[qid].get("act_docs", []))
        if rr and not tr.abstain_reason:
            pinned = set(tr.pinned)
            head = [u for u in ranked if u not in pinned][:rr.get("depth", 20)]
            mix = rr.get("mix", 0.5)
            rel = {u: S.relevance(llm, q["question"], store.title(store.doc[u]), str(store.label[u]),
                                  store.text(u)[:rr.get("chars", 2000)]) for u in head}
            score = {u: mix * rel[u] + (1 - mix) * tr.ce.get(u, 0.0) for u in head}
            new_head = sorted(head, key=lambda u: -score[u])
            hs = set(head)
            ranked = list(tr.pinned) + new_head + [u for u in ranked if u not in pinned and u not in hs]
            t["llm_rel"] = {str(u): round(rel[u], 4) for u in head}
        t["final"] = ranked[:100]
        spans = {} if tr.abstain_reason else {b: pack(ranked, b, store, min_piece=min_piece) for b in (2000, 8000, 32000)}
        abstain = tr.abstain_reason
        if jd and not abstain:
            ps = passages_for(store, spans[HEADLINE_BUDGET], jd.get("passages", 5), jd.get("chars", 1200))
            pj = S.judge(llm, q["question"], fmt_passages(ps))
            t["judge_p"] = round(pj, 4)
            if pj < jd.get("p", 0.5) and not tr.pinned and (tr.rerank_top or 0.0) < jd.get("ce_hi", 0.5):
                abstain = t["abstain_reason"] = "llm_judge"
        run[qid] = {b: [] for b in (2000, 8000, 32000)} if abstain else spans
        if an:
            if abstain:
                answers[qid] = {"text": "", "abstained": True, "citations": []}
            else:
                ps = passages_for(store, run[qid][HEADLINE_BUDGET], an.get("passages", 8), an.get("chars", 2500))
                if an.get("mode") == "llm":
                    o = S.answer(ans_llm, q["question"], fmt_passages(ps), prompt=an.get("prompt", "v1"),
                                 think=an.get("think", False), max_tokens=an.get("max_tokens", 400))
                    cited = [ps[i - 1] for i in o["citations"] if 1 <= i <= len(ps) and ps[i - 1]["key"]]
                    text = S.ground_quotes(o["answer"], [p["text"] for p in cited])   # cited only (= scorer haystack)
                    t["answer_raw"] = o
                    answers[qid] = {"text": "" if o["abstain"] else text, "abstained": o["abstain"] or not text.strip(),
                                    "citations": [{"document_id": p["document_id"], "key": p["key"]}
                                                  for p in cited if p["key"]]}
                else:   # extractive, no LLM: quote the opening of the top provision, cite it
                    p0 = next((p for p in ps if p["key"]), ps[0] if ps else None)
                    snippet = " ".join(p0["text"].split())[: an.get("snippet", 400)] if p0 else ""
                    answers[qid] = {"text": f'"{snippet}"' if snippet else "", "abstained": not snippet,
                                    "citations": [{"document_id": p0["document_id"], "key": p0["key"]}] if p0 and p0["key"] else []}
        traces[qid] = t
        if n % 25 == 0:
            print(f"  stage3 {n}/{len(qs)} ({time.time() - t0:.0f}s)", flush=True)
    for m in {id(x): x for x in (llm, ans_llm) if x is not None}.values():
        m.unload()
        timing.setdefault("llm_stats", {})[m.model] = m.stats
    timing["stage3_s"] = round(time.time() - t0, 1)

    (out_dir / f"trace_{args.split}.json").write_text(json.dumps(traces, ensure_ascii=False), encoding="utf-8")
    pred = write_predictions(out_dir / f"pred_{args.split}.json", system=cfg["name"], split=args.split, run=run,
                             answers=answers or None, meta={"config": cfg, "timing": timing})
    print(f"{len(qs)} questions; timing {json.dumps(timing)} -> {pred}")
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
