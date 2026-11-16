"""In-memory unit metadata (no text) + lazy document text. Shared by every pipeline stage."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from isr import CORPUS_DIR
from isr.corpus.build import DOCS_PATH, UNITS_DIR

COLUMNS = ["unit_id", "document_id", "kind", "key", "parent_id", "char_start", "char_end",
           "label", "region", "class_id", "is_rep", "class_size"]


class UnitStore:
    def __init__(self, text_cache_docs: int = 512) -> None:
        u = pq.read_table(UNITS_DIR, columns=COLUMNS).to_pandas()
        assert (u["unit_id"].to_numpy() == np.arange(len(u))).all()
        self.units = u
        self.doc = u["document_id"].to_numpy(dtype=object)
        self.kind = u["kind"].to_numpy(dtype=object)
        self.start = u["char_start"].to_numpy(np.int64)
        self.end = u["char_end"].to_numpy(np.int64)
        self.parent = u["parent_id"].to_numpy(np.int64)
        self.class_id = u["class_id"].to_numpy(np.int64)
        self.label = u["label"].to_numpy(dtype=object)
        self.docs = pd.read_parquet(DOCS_PATH).set_index("document_id")
        self._text: OrderedDict[str, str] = OrderedDict()
        self._cache = text_cache_docs
        self._members = None
        self.is_rep = u["is_rep"].to_numpy(bool)
        self.rep_of_class = np.full(int(self.class_id.max()) + 1, -1, np.int64)
        self.rep_of_class[self.class_id[self.is_rep]] = np.flatnonzero(self.is_rep)
        self._doc_units = None
        self._prov_by_number = None

    def doc_units(self, did: str) -> np.ndarray:
        """All unit ids (any copy status) of one document."""
        if self._doc_units is None:
            self._doc_units = {d: np.asarray(ix) for d, ix in self.units.groupby("document_id").indices.items()}
        return self._doc_units.get(did, np.empty(0, np.int64))

    def reps_for_docs(self, dids) -> np.ndarray:
        """Representative units covering every unit of the given documents (dedup-consistent)."""
        uids = [self.doc_units(d) for d in dids]
        if not uids:
            return np.empty(0, np.int64)
        return np.unique(self.rep_of_class[self.class_id[np.concatenate(uids)]])

    def provisions_numbered(self, did: str, number: str) -> list[int]:
        """Provision unit ids in a document whose last key component has this number (case-insensitive)."""
        if self._prov_by_number is None:
            idx: dict[tuple[str, str], list[int]] = {}
            prov = self.units[self.units["kind"] == "provision"]
            for uid, d, key in zip(prov["unit_id"], prov["document_id"], prov["key"]):
                num = key.rsplit("/", 1)[-1].split(":", 1)[-1].upper()
                idx.setdefault((d, num), []).append(int(uid))
            self._prov_by_number = idx
        return self._prov_by_number.get((did, number.upper()), [])

    def __len__(self) -> int:
        return len(self.units)

    def span(self, uid: int) -> tuple[str, int, int]:
        return self.doc[uid], int(self.start[uid]), int(self.end[uid])

    def doc_text(self, did: str) -> str:
        t = self._text.get(did)
        if t is None:
            with open(CORPUS_DIR / "text" / f"{did}.txt", encoding="utf-8", newline="") as fh:
                t = fh.read()
            self._text[did] = t
            if len(self._text) > self._cache:
                self._text.popitem(last=False)
        else:
            self._text.move_to_end(did)
        return t

    def text(self, uid: int) -> str:
        did, s, e = self.span(uid)
        return self.doc_text(did)[s:e]

    def title(self, did: str) -> str:
        return str(self.docs.at[did, "title"])

    def class_members(self, class_id: int) -> np.ndarray:
        if self._members is None:
            order = np.argsort(self.class_id, kind="stable")
            bounds = np.searchsorted(self.class_id[order], np.arange(self.class_id.max() + 2))
            self._members = (order, bounds)
        order, bounds = self._members
        return order[bounds[class_id]:bounds[class_id + 1]]
