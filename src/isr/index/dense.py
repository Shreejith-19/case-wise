"""Dense index over representative units: fp16 embeddings, exact search on the GPU.

    python -m isr.index.dense build --model qwen3-0.6b      (resumable, shard by shard)

Storage: indexes/dense_<model>/part-XXXX.npy (fp16) + part-XXXX.ids.npy, merged into
emb.npy / unit_ids.npy at the end. Search is a brute-force fp16 matmul on the GPU
(~1.6 GB for 784k x 1024) — exact, and small enough to load per batch stage, which is how
the 8 GB card is shared with the LLM in Phase B (plan.md §5: stage-wise pipeline).
Indexed text per unit = document title + unit label + unit text (same as BM25).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from isr import INDEX_DIR
from isr.corpus.build import DOCS_PATH, UNITS_DIR

TASK = "Given a question about Indian law, retrieve the statutory provision that answers it"
MAX_LEN = 512
MODELS = {
    "bge-m3": {"id": "BAAI/bge-m3", "q": "", "d": ""},
    "qwen3-0.6b": {"id": "Qwen/Qwen3-Embedding-0.6B", "q": f"Instruct: {TASK}\nQuery: ", "d": ""},
    "e5-large-instruct": {"id": "intfloat/multilingual-e5-large-instruct",
                          "q": f"Instruct: {TASK}\nQuery: ", "d": ""},
    "inlegalbert": {"id": "law-ai/InLegalBERT", "q": "", "d": ""},
}


def index_dir(model: str) -> Path:
    return INDEX_DIR / f"dense_{model}"


def load_model(model: str):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODELS[model]["id"], device="cuda", model_kwargs={"torch_dtype": torch.float16})
    m.max_seq_length = MAX_LEN
    return m


def _encode_sorted(m, texts: list[str], batch: int) -> np.ndarray:
    order = np.argsort([len(t) for t in texts])[::-1]
    emb = m.encode([texts[i] for i in order], batch_size=batch, normalize_embeddings=True,
                   convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
    out = np.empty_like(emb)
    out[order] = emb
    return out


def build(model: str, batch: int = 32) -> dict:
    out = index_dir(model)
    out.mkdir(parents=True, exist_ok=True)
    titles = dict(pd.read_parquet(DOCS_PATH, columns=["document_id", "title"]).itertuples(index=False))
    m = load_model(model)
    prefix = MODELS[model]["d"]
    shards = sorted(UNITS_DIR.glob("part-*.parquet"))
    t0, done_chars = time.time(), 0
    for n, shard in enumerate(shards, 1):
        target = out / f"{shard.stem}.npy"
        if target.exists():
            continue
        t = pq.read_table(shard, columns=["unit_id", "document_id", "label", "text", "is_rep"]).to_pandas()
        t = t[t["is_rep"]]
        texts = [f"{prefix}{titles[d]}\n{l}\n{x}" for d, l, x in zip(t["document_id"], t["label"], t["text"])]
        emb = _encode_sorted(m, texts, batch)
        np.save(out / f"{shard.stem}.ids.npy", t["unit_id"].to_numpy(np.int32))
        np.save(target, emb)                       # written last: its presence marks the shard done
        done_chars += sum(len(x) for x in texts)
        el = time.time() - t0
        print(f"  {n}/{len(shards)} shards  {el / 60:.1f} min  {done_chars / max(el, 1e-9) / 1e3:.0f}k chars/s",
              flush=True)
    del m
    torch.cuda.empty_cache()
    parts = sorted(p for p in out.glob("part-*.npy") if not p.name.endswith(".ids.npy"))
    ids = np.concatenate([np.load(p.with_name(p.stem + ".ids.npy")) for p in parts])
    dim = np.load(parts[0], mmap_mode="r").shape[1]
    merged = np.lib.format.open_memmap(out / "emb.npy", mode="w+", dtype=np.float16, shape=(len(ids), dim))
    pos = 0
    for p in parts:
        a = np.load(p)
        merged[pos:pos + len(a)] = a
        pos += len(a)
    merged.flush()
    del merged
    np.save(out / "unit_ids.npy", ids)
    for p in parts:
        p.unlink()
        p.with_name(p.stem + ".ids.npy").unlink()
    info = {"model": MODELS[model]["id"], "units": int(len(ids)), "dim": int(dim), "max_len": MAX_LEN,
            "minutes": round((time.time() - t0) / 60, 1)}
    (out / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


class DenseIndex:
    """Exact cosine search on the GPU. Load for a batch stage, then ``close()``."""

    def __init__(self, model: str, chunk: int = 200_000) -> None:
        d = index_dir(model)
        self.model_name = model
        self.unit_ids = np.load(d / "unit_ids.npy")
        emb = np.load(d / "emb.npy", mmap_mode="r")
        self.emb = torch.empty(emb.shape, dtype=torch.float16, device="cuda")
        for s in range(0, len(emb), chunk):
            self.emb[s:s + chunk] = torch.from_numpy(np.ascontiguousarray(emb[s:s + chunk])).cuda()
        self._model = None

    def encode_queries(self, queries: Sequence[str], batch: int = 32) -> torch.Tensor:
        if self._model is None:
            self._model = load_model(self.model_name)
        q = [MODELS[self.model_name]["q"] + x for x in queries]
        e = self._model.encode(q, batch_size=batch, normalize_embeddings=True, convert_to_tensor=True)
        return e.to(torch.float16)

    def search(self, queries: Sequence[str], k: int = 300) -> list[tuple[np.ndarray, np.ndarray]]:
        qe = self.encode_queries(queries)
        out = []
        for i in range(0, len(qe), 64):
            sims = qe[i:i + 64] @ self.emb.T
            s, ix = torch.topk(sims.float(), k=min(k, sims.shape[1]), dim=1)
            for row_s, row_ix in zip(s.cpu().numpy(), ix.cpu().numpy()):
                out.append((self.unit_ids[row_ix], row_s))
        return out

    def close(self) -> None:
        del self.emb
        self._model = None
        torch.cuda.empty_cache()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build"])
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()
    print(json.dumps(build(args.model, args.batch), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
