"""Config-driven retrieval pipeline. Grows stage by stage (plan.md §7 experiment matrix).

Every signal is a ranked list of *representative* unit ids; lists are fused with weighted
RRF. Config keys (configs/*.yaml):

    retrieve:
      bm25:  {k: 300, weight: 1.0}          global sparse
      dense: {k: 300, weight: 1.0}          (Phase 2.3)
    metadata:                               rule-based query parsing (isr.query.rules)
      act_bm25:   {k: 100, weight: 1.0}     BM25 restricted to named Acts (all copies)
      exact_ref:  {weight: 2.0}             named Act + section/rule number -> that provision
      jur_bm25:   {k: 100, weight: 0.5}     BM25 restricted to the named state's documents
    abstain:
      missing_provision: true               named Act resolved, cited number exists in no copy
      unresolved_law: true                  question names a law ("... Act") not in the corpus
    fusion: {k: 60}
    pack:   {min_piece: 200}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from isr.eval.kit import BUDGETS
from isr.pack import pack
from isr.query.rules import ParsedQuery, parse
from isr.store import UnitStore


@dataclass
class Trace:
    parsed: ParsedQuery | None = None
    lists: dict[str, list[int]] = field(default_factory=dict)
    abstain_reason: str | None = None


def rrf(lists: dict[str, list[int]], weights: dict[str, float], k: int = 60) -> list[int]:
    score: dict[int, float] = {}
    for name, ranked in lists.items():
        w = weights.get(name, 1.0)
        for i, uid in enumerate(ranked):
            score[uid] = score.get(uid, 0.0) + w / (k + i + 1)
    return [u for u, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]


class Pipeline:
    def __init__(self, config: dict, store: UnitStore | None = None) -> None:
        self.config = config
        self.name = config["name"]
        self.store = store or UnitStore()
        self.bm25 = None
        if "bm25" in config.get("retrieve", {}) or config.get("metadata"):
            from isr.index.bm25 import BM25
            self.bm25 = BM25()
        self.jur_docs: dict[str, list[str]] = {}
        if config.get("metadata", {}).get("jur_bm25"):
            docs = self.store.docs
            self.jur_docs = {j: list(g.index) for j, g in docs.groupby("jurisdiction")}
        self._mask_cache: dict[tuple, np.ndarray] = {}

    # --- helpers ----------------------------------------------------------------------

    def _mask_for_docs(self, dids: Sequence[str]) -> np.ndarray:
        key = tuple(sorted(dids))
        m = self._mask_cache.get(key)
        if m is None:
            m = np.zeros(len(self.bm25.unit_ids), bool)
            rows = self.bm25.row_of()[self.store.reps_for_docs(key)]
            m[rows[rows >= 0]] = True
            if len(self._mask_cache) > 256:
                self._mask_cache.clear()
            self._mask_cache[key] = m
        return m

    def _bm25(self, text: str, k: int, docs: Sequence[str] | None = None) -> list[int]:
        mask = self._mask_for_docs(docs) if docs is not None else None
        ids, _ = self.bm25.search(text, k=k, mask=mask)
        return [int(i) for i in ids]

    # --- stages -----------------------------------------------------------------------

    def rank(self, question: str) -> tuple[list[int], Trace]:
        cfg = self.config
        tr = Trace()
        lists: dict[str, list[int]] = {}
        weights: dict[str, float] = {}
        r = cfg.get("retrieve", {})
        if "bm25" in r:
            lists["bm25"] = self._bm25(question, r["bm25"].get("k", 300))
            weights["bm25"] = r["bm25"].get("weight", 1.0)

        md = cfg.get("metadata") or {}
        ab = cfg.get("abstain") or {}
        if md or ab:
            p = tr.parsed = parse(question)
            act_docs = p.act_document_ids
            if "act_bm25" in md and act_docs:
                lists["act_bm25"] = self._bm25(question, md["act_bm25"].get("k", 100), act_docs)
                weights["act_bm25"] = md["act_bm25"].get("weight", 1.0)
            numbers = [n for kind, n in p.refs if kind in ("section", "article", "rule", "regulation")]
            exact: list[int] = []
            for n in numbers:
                for d in act_docs:
                    exact.extend(self.store.provisions_numbered(d, n))
            if "exact_ref" in md and exact:
                reps = [int(self.store.rep_of_class[self.store.class_id[u]]) for u in exact]
                lists["exact_ref"] = list(dict.fromkeys(reps))
                weights["exact_ref"] = md["exact_ref"].get("weight", 2.0)
            if "jur_bm25" in md and p.jurisdictions:
                jdocs = [d for j in p.jurisdictions for d in self.jur_docs.get(j, [])]
                if jdocs:
                    lists["jur_bm25"] = self._bm25(question, md["jur_bm25"].get("k", 100), jdocs)
                    weights["jur_bm25"] = md["jur_bm25"].get("weight", 0.5)
            if ab.get("missing_provision") and act_docs and numbers and not exact:
                tr.abstain_reason = "missing_provision"
            elif ab.get("unresolved_law") and p.named_law_unresolved:
                tr.abstain_reason = "unresolved_law"
        tr.lists = lists
        if not lists:
            raise ValueError("no retriever configured")
        fused = rrf(lists, weights, k=cfg.get("fusion", {}).get("k", 60)) if len(lists) > 1 else lists[next(iter(lists))]
        return fused, tr

    def run(self, question: str, budgets: Sequence[int] = BUDGETS) -> dict[int, list]:
        return self.run_traced(question, budgets)[0]

    def run_traced(self, question: str, budgets: Sequence[int] = BUDGETS) -> tuple[dict[int, list], Trace]:
        ranked, tr = self.rank(question)
        if tr.abstain_reason:
            return {b: [] for b in budgets}, tr
        p = self.config.get("pack", {})
        return {b: pack(ranked, b, self.store, min_piece=p.get("min_piece", 200)) for b in budgets}, tr
