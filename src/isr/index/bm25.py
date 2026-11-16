"""BM25 over representative units, as a precomputed sparse weight matrix (memory-lean).

    python -m isr.index.bm25 build [--k1 1.2 --b 0.75 --workers 6]

Indexed text per unit = document title + unit label + unit text, so a unit can match
on the name of its Act. Scoring is Lucene BM25:
    w(t, d) = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    idf(t)  = ln(1 + (N - df + 0.5) / (df + 0.5))
stored as a CSC matrix (units x terms); a query sums the columns of its terms.
Only class representatives are indexed (plan.md Phase 1, dedup).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from isr import INDEX_DIR
from isr.corpus.build import DOCS_PATH, UNITS_DIR
from isr.text import tokenize

BM25_DIR = INDEX_DIR / "bm25_units"

_TITLES: dict[str, str] = {}


def _init_worker():
    global _TITLES
    docs = pd.read_parquet(DOCS_PATH, columns=["document_id", "title"])
    _TITLES = dict(zip(docs["document_id"], docs["title"]))


def _tokenize_shard(path: str):
    t = pq.read_table(path, columns=["unit_id", "document_id", "label", "text", "is_rep"]).to_pandas()
    t = t[t["is_rep"]]
    vocab: dict[str, int] = {}
    indptr, indices, data, lens = [0], [], [], []
    for did, label, text in zip(t["document_id"], t["label"], t["text"]):
        toks = tokenize(f"{_TITLES.get(did, '')}\n{label}\n{text}")
        counts: dict[int, int] = {}
        for tok in toks:
            j = vocab.setdefault(tok, len(vocab))
            counts[j] = counts.get(j, 0) + 1
        indices.extend(counts.keys())
        data.extend(counts.values())
        indptr.append(len(indices))
        lens.append(len(toks))
    terms = np.empty(len(vocab), dtype=object)
    for tok, j in vocab.items():
        terms[j] = tok
    return (t["unit_id"].to_numpy(np.int32), np.asarray(lens, np.int32), terms,
            np.asarray(indptr, np.int64), np.asarray(indices, np.int32), np.asarray(data, np.float32))


def build(k1: float = 1.2, b: float = 0.75, workers: int = 6, out: Path = BM25_DIR) -> dict:
    t0 = time.time()
    shards = sorted(str(p) for p in UNITS_DIR.glob("part-*.parquet"))
    vocab: dict[str, int] = {}
    unit_ids, lens, blocks = [], [], []
    with mp.Pool(workers, initializer=_init_worker) as pool:
        for n, (uids, ln, terms, indptr, idx, dat) in enumerate(pool.imap(_tokenize_shard, shards), 1):
            remap = np.fromiter((vocab.setdefault(tk, len(vocab)) for tk in terms), np.int32, len(terms))
            blocks.append((indptr, remap[idx] if len(idx) else idx, dat))
            unit_ids.append(uids)
            lens.append(ln)
            if n % 10 == 0 or n == len(shards):
                print(f"  tokenized {n}/{len(shards)} shards, vocab {len(vocab):,} ({time.time() - t0:.0f}s)",
                      flush=True)
    n_terms = len(vocab)
    tf = sp.vstack([sp.csr_matrix((d, i, p), shape=(len(p) - 1, n_terms)) for p, i, d in blocks],
                   format="csr", dtype=np.float32)
    del blocks
    unit_ids = np.concatenate(unit_ids)
    lens = np.concatenate(lens).astype(np.float32)
    n_docs = tf.shape[0]
    avgdl = float(lens.mean())
    df = np.bincount(tf.indices, minlength=n_terms).astype(np.float32)
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    # in-place BM25 weighting on the CSR data
    row_norm = k1 * (1 - b + b * lens / avgdl)
    rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
    tfd = tf.data
    tf.data = (idf[tf.indices] * tfd * (k1 + 1) / (tfd + row_norm[rows])).astype(np.float32)
    del rows, tfd
    w = tf.tocsc()
    del tf
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "data.npy", w.data)
    np.save(out / "indices.npy", w.indices)
    np.save(out / "indptr.npy", w.indptr)
    np.save(out / "unit_ids.npy", unit_ids)
    terms = [None] * n_terms
    for tok, j in vocab.items():
        terms[j] = tok
    (out / "vocab.json").write_text(json.dumps(terms), encoding="utf-8")
    params = {"k1": k1, "b": b, "units": int(n_docs), "terms": n_terms, "nnz": int(w.nnz),
              "avgdl": avgdl, "seconds": round(time.time() - t0, 1)}
    (out / "params.json").write_text(json.dumps(params, indent=1), encoding="utf-8")
    return params


class BM25:
    """Load once; ``search`` returns (unit_ids, scores) best first."""

    def __init__(self, path: Path = BM25_DIR, mmap: bool = True) -> None:
        mode = "r" if mmap else None
        data = np.load(path / "data.npy", mmap_mode=mode)
        indices = np.load(path / "indices.npy", mmap_mode=mode)
        indptr = np.load(path / "indptr.npy")
        self.unit_ids = np.load(path / "unit_ids.npy")
        terms = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        self.vocab = {t: j for j, t in enumerate(terms)}
        self.params = json.loads((path / "params.json").read_text(encoding="utf-8"))
        self.w = sp.csc_matrix((data, indices, indptr), shape=(len(self.unit_ids), len(terms)))
        self._row_of = None

    def row_of(self) -> np.ndarray:
        """unit_id -> row (or -1), for masks."""
        if self._row_of is None:
            r = np.full(int(self.unit_ids.max()) + 1, -1, np.int64)
            r[self.unit_ids] = np.arange(len(self.unit_ids))
            self._row_of = r
        return self._row_of

    def scores(self, query: str) -> np.ndarray:
        ids = sorted({self.vocab[t] for t in tokenize(query) if t in self.vocab})
        if not ids:
            return np.zeros(len(self.unit_ids), np.float32)
        return np.asarray(self.w[:, ids].sum(axis=1)).ravel()

    def search(self, query: str, k: int = 100, mask: Optional[np.ndarray] = None
               ) -> tuple[np.ndarray, np.ndarray]:
        s = self.scores(query)
        if mask is not None:
            s = np.where(mask, s, 0.0)
        k = min(k, int((s > 0).sum()))
        if k <= 0:
            return np.empty(0, np.int32), np.empty(0, np.float32)
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return self.unit_ids[top], s[top]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build"])
    p.add_argument("--k1", type=float, default=1.2)
    p.add_argument("--b", type=float, default=0.75)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    print(json.dumps(build(args.k1, args.b, args.workers), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
