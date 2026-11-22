"""CPU variant of ``isr.index.dense.DenseIndex`` for interactive use (scripts/chat.py).

Same stored vectors (indexes/dense_<model>/emb.npy, held in RAM as fp32, ~3.2 GB) and the same query prefix, but
the query encoder and the exact search run on the CPU, so the 8 GB GPU stays free for the cross-encoder and the
LLM at the same time. Scores can differ from the GPU fp16 index in the last digits; rankings almost never do.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from isr.index.dense import MAX_LEN, MODELS, index_dir


class CpuDenseIndex:
    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer
        d = index_dir(model)
        self.model_name = model
        self.unit_ids = np.load(d / "unit_ids.npy")
        self.emb = np.asarray(np.load(d / "emb.npy", mmap_mode="r"), dtype=np.float32)
        self._model = SentenceTransformer(MODELS[model]["id"], device="cpu")
        self._model.max_seq_length = MAX_LEN
        self._row_of = None

    def encode_queries(self, queries: Sequence[str], as_document: bool = False,
                       instruction: str | None = None) -> np.ndarray:
        prefix = MODELS[self.model_name]["d" if as_document else "q"]
        if instruction and not as_document and prefix.startswith("Instruct: "):
            prefix = f"Instruct: {instruction}\nQuery: "
        return self._model.encode([prefix + q for q in queries], normalize_embeddings=True,
                                  convert_to_numpy=True).astype(np.float32)

    def search(self, queries: Sequence[str], k: int = 300,
               as_document: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
        out = []
        for q in self.encode_queries(queries, as_document=as_document):
            sims = self.emb @ q
            top = np.argpartition(-sims, k)[:k]
            top = top[np.argsort(-sims[top], kind="stable")]
            out.append((self.unit_ids[top], sims[top]))
        return out

    def search_within(self, query: str, uids: Sequence[int], k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        if self._row_of is None:
            self._row_of = np.full(int(self.unit_ids.max()) + 1, -1, np.int64)
            self._row_of[self.unit_ids] = np.arange(len(self.unit_ids))
        uids = np.asarray(uids, np.int64)
        rows = self._row_of[uids[uids < len(self._row_of)]]
        rows = rows[rows >= 0]
        if len(rows) == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        sims = self.emb[rows] @ self.encode_queries([query])[0]
        top = np.argsort(-sims, kind="stable")[:k]
        return self.unit_ids[rows[top]], sims[top]

    def close(self) -> None:
        pass
