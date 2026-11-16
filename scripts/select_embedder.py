"""Phase 2.2: choose the dense embedder on a DEV-ONLY proxy pool (plan.md D9).

    python scripts/select_embedder.py [--models bge-m3 qwen3-0.6b e5-large-instruct inlegalbert]

Pool (representative units only): every unit of each dev gold document and its duplicate-
cluster mates, BM25's top-50 units per dev question (hard negatives), and 20k random
distractors. Each model embeds the pool once; we report unit-level recall@10/@50 and MRR@50
per category, alone and fused with BM25 (RRF), plus measured encoding throughput (to size
the full-corpus run). Results: runs/embedder_selection/<model>.json and summary.json.

A retrieved unit hits a gold location when they overlap by >= 50% of the shorter of the two.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from isr import DATA_DIR, RUNS_DIR
from isr.corpus.build import UNITS_DIR
from isr.eval.kit import questions_for
from isr.index.bm25 import BM25
from isr.index.dense import MAX_LEN, MODELS

OUT = RUNS_DIR / "embedder_selection"
POOL_PATH = DATA_DIR / "proxy_dev_pool.parquet"
SEED = 20261116


def build_pool(qs: list[dict], bm25: BM25) -> pd.DataFrame:
    if POOL_PATH.exists():
        return pd.read_parquet(POOL_PATH)
    docs = pd.read_parquet(DATA_DIR / "docs.parquet", columns=["document_id", "title", "cluster_id"])
    gold_docs = {loc["document_id"] for q in qs for g in q["gold_evidence"] for loc in g["locations"]}
    clusters = set(docs.loc[docs["document_id"].isin(gold_docs), "cluster_id"].dropna())
    gold_docs |= set(docs.loc[docs["cluster_id"].isin(clusters), "document_id"])
    hard = set()
    for q in qs:
        ids, _ = bm25.search(q["question"], k=50)
        hard.update(int(i) for i in ids)
    u = pq.read_table(UNITS_DIR, columns=["unit_id", "document_id", "kind", "label", "char_start",
                                          "char_end", "is_rep", "text"]).to_pandas()
    u = u[u["is_rep"]]
    rng = np.random.default_rng(SEED)
    rand = set(rng.choice(u["unit_id"].to_numpy(), size=20000, replace=False).tolist())
    pool = u[u["document_id"].isin(gold_docs) | u["unit_id"].isin(hard | rand)].copy()
    titles = dict(zip(docs["document_id"], docs["title"]))
    pool["doc_text"] = [f"{titles[d]}\n{l}\n{t}" for d, l, t in zip(pool["document_id"], pool["label"], pool["text"])]
    pool = pool.drop(columns=["text", "is_rep"]).reset_index(drop=True)
    pool.to_parquet(POOL_PATH, index=False)
    return pool


def gold_hits(q: dict, pool: pd.DataFrame, by_doc: dict) -> list[set[int]]:
    """Per gold group: set of pool row indices that hit it."""
    groups = []
    for g in q["gold_evidence"]:
        hit = set()
        for loc in g["locations"]:
            gs, ge = loc["char_start"], loc["char_end"]
            for row, s, e in by_doc.get(loc["document_id"], []):
                ov = min(e, ge) - max(s, gs)
                if ov > 0 and ov >= 0.5 * min(e - s, ge - gs):
                    hit.add(row)
        groups.append(hit)
    return groups


def metrics(rankings: dict[str, list[int]], qs: list[dict], golds: dict) -> dict:
    rows = []
    for q in qs:
        r = rankings[q["question_id"]]
        groups = golds[q["question_id"]]
        def recall(k):
            top = set(r[:k])
            return sum(bool(g & top) for g in groups) / len(groups)
        rr = []
        for g in groups:
            rank = next((i + 1 for i, x in enumerate(r[:50]) if x in g), None)
            rr.append(1 / rank if rank else 0.0)
        rows.append({"category": q["category"], "r10": recall(10), "r50": recall(50), "mrr": float(np.mean(rr))})
    df = pd.DataFrame(rows)
    out = {"all": df[["r10", "r50", "mrr"]].mean().round(4).to_dict()}
    out.update({c: g[["r10", "r50", "mrr"]].mean().round(4).to_dict() for c, g in df.groupby("category")})
    return out


def rrf(*rankings: list[int], k: int = 60, depth: int = 200) -> list[int]:
    score: dict[int, float] = {}
    for r in rankings:
        for i, x in enumerate(r[:depth]):
            score[x] = score.get(x, 0.0) + 1.0 / (k + i + 1)
    return [x for x, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def encode(model, texts: list[str], prompt: str, batch: int) -> tuple[np.ndarray, float, int]:
    tok = model.tokenizer
    lengths = [min(MAX_LEN, len(ids)) for ids in tok([prompt + t for t in texts], add_special_tokens=True,
                                                     truncation=True, max_length=MAX_LEN)["input_ids"]]
    order = np.argsort(lengths)[::-1]
    torch.cuda.synchronize()
    t0 = time.time()
    emb = model.encode([prompt + texts[i] for i in order], batch_size=batch, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
    torch.cuda.synchronize()
    secs = time.time() - t0
    out = np.empty_like(emb)
    out[order] = emb
    return out.astype(np.float32), secs, int(sum(lengths))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()
    from sentence_transformers import SentenceTransformer

    OUT.mkdir(parents=True, exist_ok=True)
    qs = [q for q in questions_for("dev") if not q["unanswerable"]]
    bm25 = BM25()
    pool = build_pool(qs, bm25)
    print(f"pool: {len(pool):,} units", flush=True)
    by_doc: dict[str, list] = {}
    for row, (d, s, e) in enumerate(zip(pool["document_id"], pool["char_start"], pool["char_end"])):
        by_doc.setdefault(d, []).append((row, s, e))
    golds = {q["question_id"]: gold_hits(q, pool, by_doc) for q in qs}
    assert all(all(g) for g in golds.values()), "every gold group must be reachable in the pool"

    # BM25 restricted to the pool (reference + fusion partner)
    row_of_uid = {int(u): i for i, u in enumerate(pool["unit_id"])}
    mask = np.zeros(len(bm25.unit_ids), bool)
    mask[bm25.row_of()[pool["unit_id"].to_numpy()]] = True
    bm25_rank = {}
    for q in qs:
        ids, _ = bm25.search(q["question"], k=200, mask=mask)
        bm25_rank[q["question_id"]] = [row_of_uid[int(i)] for i in ids]
    summary = {"pool_units": len(pool), "questions": len(qs), "bm25": metrics(bm25_rank, qs, golds)}
    print("bm25", summary["bm25"]["all"], flush=True)

    texts = pool["doc_text"].tolist()
    for name in args.models:
        spec = MODELS[name]
        t_load = time.time()
        model = SentenceTransformer(spec["id"], device="cuda", model_kwargs={"torch_dtype": torch.float16})
        model.max_seq_length = MAX_LEN
        d_emb, d_secs, d_tokens = encode(model, texts, spec["d"], args.batch)
        q_emb, _, _ = encode(model, [q["question"] for q in qs], spec["q"], args.batch)
        sims = q_emb @ d_emb.T
        dense_rank = {q["question_id"]: np.argsort(-sims[i])[:200].tolist() for i, q in enumerate(qs)}
        hybrid_rank = {qid: rrf(bm25_rank[qid], dense_rank[qid]) for qid in dense_rank}
        res = {"model": spec["id"], "dense": metrics(dense_rank, qs, golds),
               "hybrid_rrf": metrics(hybrid_rank, qs, golds),
               "encode_seconds": round(d_secs, 1), "tokens": d_tokens,
               "tokens_per_second": round(d_tokens / d_secs), "load_seconds": round(time.time() - t_load - d_secs, 1),
               "dim": int(d_emb.shape[1])}
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        summary[name] = res
        print(name, "dense", res["dense"]["all"], "hybrid", res["hybrid_rrf"]["all"],
              f"{res['tokens_per_second']:,} tok/s", flush=True)
        del model
        torch.cuda.empty_cache()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
