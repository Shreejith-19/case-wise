"""Cross-encoder reranking with a persistent score cache.

Pair = (question, "<title>\\n<label>\\n<unit text>") truncated to MAX_LEN tokens.
Scores are cached in data/cache/rerank_<model>.sqlite keyed by (question hash, unit_id), so
ablations that reorder or re-fuse the same candidates cost nothing on re-run.
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from isr import DATA_DIR

if TYPE_CHECKING:
    from isr.store import UnitStore

RERANKERS = {
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
}
MAX_LEN = 512
PASSAGE_CHARS = 2400


def _qhash(question: str) -> str:
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]


def resolve_device(device: str | None = None) -> str:
    """Resolve reranker device. 'auto' picks CPU when GPU VRAM < 5 GiB so Ollama fits alongside."""
    if device and device != "auto":
        return device
    if not torch.cuda.is_available():
        return "cpu"
    if device == "auto":
        try:
            vram = torch.cuda.get_device_properties(0).total_memory
            if vram < 5 * (1024 ** 3):
                return "cpu"
        except Exception:
            pass
    return "cuda"


class Reranker:
    def __init__(self, name: str, store: UnitStore, batch: int = 32, device: str | None = None) -> None:
        self.name = name
        self.store = store
        self.batch = batch
        self.device = resolve_device(device)
        self._model = None
        path = DATA_DIR / "cache" / f"rerank_{name}.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS s (q TEXT, uid INTEGER, score REAL, PRIMARY KEY (q, uid))")

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            kwargs = {"torch_dtype": torch.float16} if self.device == "cuda" else {}
            self._model = CrossEncoder(RERANKERS[self.name], device=self.device, max_length=MAX_LEN,
                                       model_kwargs=kwargs)
        return self._model

    def passage(self, uid: int) -> str:
        did = self.store.doc[uid]
        return f"{self.store.title(did)}\n{self.store.label[uid]}\n{self.store.text(uid)[:PASSAGE_CHARS]}"

    def scores(self, question: str, uids: Sequence[int]) -> np.ndarray:
        qh = _qhash(question)
        uids = [int(u) for u in uids]
        cached = dict(self.db.execute(
            f"SELECT uid, score FROM s WHERE q = ? AND uid IN ({','.join('?' * len(uids))})", (qh, *uids)))
        missing = [u for u in uids if u not in cached]
        if missing:
            model = self._load()
            pairs = [(question, self.passage(u)) for u in missing]
            new = model.predict(pairs, batch_size=self.batch, show_progress_bar=False)
            self.db.executemany("INSERT OR REPLACE INTO s VALUES (?, ?, ?)",
                                [(qh, u, float(s)) for u, s in zip(missing, new)])
            self.db.commit()
            cached.update({u: float(s) for u, s in zip(missing, new)})
        return np.asarray([cached[u] for u in uids], np.float32)

    def rerank(self, question: str, ranked: Sequence[int], depth: int) -> tuple[list[int], np.ndarray]:
        """Rerank the top ``depth``; the tail keeps its fused order. Returns (order, top scores)."""
        head = [int(u) for u in ranked[:depth]]
        s = self.scores(question, head)
        order = np.argsort(-s, kind="stable")
        return [head[i] for i in order] + [int(u) for u in ranked[depth:]], s[order]

    def close(self) -> None:
        self._model = None
        torch.cuda.empty_cache()
