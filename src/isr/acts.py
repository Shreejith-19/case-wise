"""Plan C (plan.md Phase 4.1-4.2): Act index over document cards + Act router.

    python -m isr.acts build          (resumable; pause with a PAUSE file, see isr.pause)

A card is cheap metadata that Phase 1 already extracted -- no LLM summaries (D10 default):
    title / long title / "<type>; <jurisdiction>; parent: <parent Act>" / section headings  (<= CARD_CHARS)
Indexes: BM25 over cards (same tokenizer as units) and dense card embeddings (the unit embedder, encoded as
passages). ``ActIndex.route(question)`` fuses both with RRF and returns document ids best first.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from isr import INDEX_DIR, pause
from isr.corpus.build import DOCS_PATH
from isr.text import tokenize

ACT_DIR = INDEX_DIR / "acts"
CARD_CHARS = 2000
CHUNK = 1000
ROUTE_TASK = "Given a question about Indian law, retrieve the Act, Rules or Regulations whose provisions answer it"


def _s(x) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) or str(x) in ("None", "nan", "<NA>") else str(x)


def cards() -> tuple[list[str], list[str]]:
    d = pd.read_parquet(DOCS_PATH)
    ids, texts = [], []
    for r in d.itertuples(index=False):
        head = [_s(r.title)]
        if _s(r.long_title) and _s(r.long_title) != _s(r.title):
            head.append(_s(r.long_title))
        meta = f"{_s(r.document_type)}; {_s(r.jurisdiction)}"
        if _s(r.parent_title):
            meta += f"; under: {_s(r.parent_title)}"
        head.append(meta)
        texts.append(("\n".join(head) + "\n" + _s(r.headings))[:CARD_CHARS])
        ids.append(r.document_id)
    return ids, texts


def build(model: str = "qwen3-0.6b", batch: int = 32) -> dict:
    """Card BM25 (seconds) + dense card embeddings in checkpointed chunks (pausable)."""
    from isr.index.dense import MODELS, _encode_sorted, load_model
    ACT_DIR.mkdir(parents=True, exist_ok=True)
    ids, texts = cards()
    pause.save_json(ACT_DIR / "doc_ids.json", ids)
    pause.save_json(ACT_DIR / "cards_sample.json", texts[:20])
    t0 = time.time()
    # BM25 over cards (Lucene BM25, k1=1.2, b=0.75 -- the unit index's parameters)
    if not (ACT_DIR / "bm25.npz").exists():
        vocab: dict[str, int] = {}
        rows, cols, vals, lens = [], [], [], []
        for i, t in enumerate(texts):
            toks = tokenize(t)
            lens.append(len(toks))
            c: dict[int, int] = {}
            for tok in toks:
                j = vocab.setdefault(tok, len(vocab))
                c[j] = c.get(j, 0) + 1
            rows += [i] * len(c)
            cols += list(c)
            vals += list(c.values())
        tf = sp.csr_matrix((np.asarray(vals, np.float32), (rows, cols)), shape=(len(texts), len(vocab)))
        lens = np.asarray(lens, np.float32)
        df = np.bincount(tf.indices, minlength=len(vocab)).astype(np.float32)
        idf = np.log1p((len(texts) - df + 0.5) / (df + 0.5)).astype(np.float32)
        rn = 1.2 * (1 - 0.75 + 0.75 * lens / lens.mean())
        r = np.repeat(np.arange(len(texts)), np.diff(tf.indptr))
        tf.data = (idf[tf.indices] * tf.data * 2.2 / (tf.data + rn[r])).astype(np.float32)
        w = tf.tocsc()
        sp.save_npz(ACT_DIR / "bm25.tmp.npz", w)
        (ACT_DIR / "bm25.tmp.npz").replace(ACT_DIR / "bm25.npz")
        pause.save_json(ACT_DIR / "bm25_vocab.json", [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])])
        print(f"  card BM25: {len(texts)} cards, {len(vocab)} terms ({time.time() - t0:.0f}s)", flush=True)
    # dense card embeddings, chunk by chunk
    m = None
    n_chunks = (len(texts) + CHUNK - 1) // CHUNK
    for c in range(n_chunks):
        part = ACT_DIR / f"dense_{model}.part{c:03d}.npy"
        if part.exists():
            continue
        pause.check(f"card chunk {c}/{n_chunks}")
        if m is None:
            m = load_model(model)
        chunk = [MODELS[model]["d"] + t for t in texts[c * CHUNK:(c + 1) * CHUNK]]
        pause.save_npy(part, _encode_sorted(m, chunk, batch))
        print(f"  card embeddings {c + 1}/{n_chunks} ({time.time() - t0:.0f}s)", flush=True)
    parts = [np.load(ACT_DIR / f"dense_{model}.part{c:03d}.npy") for c in range(n_chunks)]
    pause.save_npy(ACT_DIR / f"dense_{model}.npy", np.concatenate(parts))
    info = {"cards": len(texts), "model": model, "card_chars": CARD_CHARS, "seconds": round(time.time() - t0, 1)}
    pause.save_json(ACT_DIR / "info.json", info)
    if m is not None:
        import torch
        del m
        torch.cuda.empty_cache()
    return info


class ActIndex:
    """Route a question to documents. Dense query encoding reuses a loaded ``DenseIndex`` (same embedder)."""

    def __init__(self, dense=None, model: str = "qwen3-0.6b") -> None:
        self.doc_ids = json.loads((ACT_DIR / "doc_ids.json").read_text(encoding="utf-8"))
        self.w = sp.load_npz(ACT_DIR / "bm25.npz").tocsc()
        terms = json.loads((ACT_DIR / "bm25_vocab.json").read_text(encoding="utf-8"))
        self.vocab = {t: j for j, t in enumerate(terms)}
        self.dense = dense
        self.emb = None
        if dense is not None:
            import torch
            self.emb = torch.from_numpy(np.load(ACT_DIR / f"dense_{model}.npy")).cuda()

    def bm25(self, question: str, k: int) -> list[int]:
        ids = sorted({self.vocab[t] for t in tokenize(question) if t in self.vocab})
        if not ids:
            return []
        s = np.asarray(self.w[:, ids].sum(axis=1)).ravel()
        k = min(k, int((s > 0).sum()))
        top = np.argpartition(-s, k - 1)[:k] if k else np.empty(0, int)
        return [int(i) for i in top[np.argsort(-s[top], kind="stable")]]

    def dense_rank(self, question: str, k: int) -> list[int]:
        if self.emb is None:
            return []
        import torch
        q = self.dense.encode_queries([question], instruction=ROUTE_TASK)
        sims = (q @ self.emb.T).float()[0]
        return [int(i) for i in torch.topk(sims, k=min(k, len(sims))).indices.cpu().numpy()]

    def route(self, question: str, k: int = 10, depth: int = 100, rrf_k: int = 60) -> list[str]:
        score: dict[int, float] = {}
        for ranked in (self.bm25(question, depth), self.dense_rank(question, depth)):
            for r, i in enumerate(ranked):
                score[i] = score.get(i, 0.0) + 1.0 / (rrf_k + r + 1)
        order = sorted(score, key=lambda i: (-score[i], i))
        return [self.doc_ids[i] for i in order[:k]]

    def route_dense(self, question: str, k: int = 5) -> list[str]:
        """Card-dense routing (the Phase 4 diagnostic showed card BM25 hurts the fused router)."""
        return [self.doc_ids[i] for i in self.dense_rank(question, k)]

    def close(self) -> None:
        self.emb = None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build"])
    p.add_argument("--model", default="qwen3-0.6b")
    args = p.parse_args()
    try:
        print(json.dumps(build(args.model), indent=1))
    except pause.Paused as e:
        print(e)
        return pause.EXIT_PAUSED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
