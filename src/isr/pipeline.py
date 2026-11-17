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

import re
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
    rerank_top: float | None = None          # best cross-encoder score (for threshold abstention)
    final: list[int] = field(default_factory=list)
    ce: dict[int, float] = field(default_factory=dict)   # cross-encoder score of every reranked unit
    pinned: list[int] = field(default_factory=list)      # exact named-provision hits kept on top
    routed: list[str] = field(default_factory=list)      # Plan C: documents chosen by the Act router


_PARTS = re.compile(r",\s*and\s+|;\s*|\s+and\s+(?=(?:who|what|which|when|where|how|whether|can|does|do|is|are|must"
                    r"|may|will)\b)", re.I)


def question_parts(question: str) -> list[str]:
    """Sub-questions of a compound question ("Who can X, and what can Y?"); [] if it has only one part.
    Rule fixed before the Phase 4 ablations (plan.md Phase 4 log)."""
    ps = [p.strip(" ?") for p in _PARTS.split(question) if len(p.strip()) > 15]
    return ps if len(ps) > 1 else []


def rrf(lists: dict[str, list[int]], weights: dict[str, float], k: int = 60) -> list[int]:
    score: dict[int, float] = {}
    for name, ranked in lists.items():
        w = weights.get(name, 1.0)
        for i, uid in enumerate(ranked):
            score[uid] = score.get(uid, 0.0) + w / (k + i + 1)
    return [u for u, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]


def named_provisions(p: ParsedQuery, store: UnitStore) -> tuple[list[str], list[int]]:
    """Provision numbers named in the question, and the provision units that carry them in the resolved Acts."""
    numbers = [n for kind, n in p.refs if kind in ("section", "article", "rule", "regulation")]
    exact: list[int] = []
    for n in numbers:
        for d in p.act_document_ids:
            exact.extend(store.provisions_numbered(d, n))
    return numbers, exact


def rule_abstain_reason(p: ParsedQuery, numbers: list[str], exact: list[int], ab: dict) -> str | None:
    """Deterministic abstention: a resolved Act lacks the named provision, or a named law is not in the corpus."""
    if ab.get("missing_provision") and p.act_document_ids and numbers and not exact:
        return "missing_provision"
    if ab.get("unresolved_law") and p.named_law_unresolved:
        return "unresolved_law"
    return None


class Pipeline:
    def __init__(self, config: dict, store: UnitStore | None = None) -> None:
        self.config = config
        self.name = config["name"]
        self.store = store or UnitStore()
        self.bm25 = None
        if "bm25" in config.get("retrieve", {}) or config.get("metadata"):
            from isr.index.bm25 import BM25
            self.bm25 = BM25()
        self.dense = None
        if "dense" in config.get("retrieve", {}):
            from isr.index.dense import DenseIndex
            self.dense = DenseIndex(config["retrieve"]["dense"]["model"])
        self.reranker = None
        if config.get("rerank"):
            from isr.retrieve.rerank import Reranker
            self.reranker = Reranker(config["rerank"]["model"], self.store)
        pc = config.get("planc") or {}
        self.acts = None
        if pc.get("route"):
            from isr.acts import ActIndex
            self.acts = ActIndex(self.dense)
        self.xref = None
        if pc.get("xref"):
            from isr.xref import XrefGraph
            self.xref = XrefGraph()
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

    def rank(self, question: str, signals: dict | None = None) -> tuple[list[int], Trace]:
        """``signals``: Plan B stage-1 LLM outputs (see isr.llm.run): act_docs, keywords, hyde, jurisdiction."""
        cfg = self.config
        tr = Trace()
        lists: dict[str, list[int]] = {}
        weights: dict[str, float] = {}
        r = cfg.get("retrieve", {})
        if "bm25" in r:
            lists["bm25"] = self._bm25(question, r["bm25"].get("k", 300))
            weights["bm25"] = r["bm25"].get("weight", 1.0)
        if self.dense is not None:
            ids, _ = self.dense.search([question], k=r["dense"].get("k", 300))[0]
            lists["dense"] = [int(i) for i in ids]
            weights["dense"] = r["dense"].get("weight", 1.0)

        md = cfg.get("metadata") or {}
        ab = cfg.get("abstain") or {}
        if md or ab:
            p = tr.parsed = parse(question)
            act_docs = p.act_document_ids
            if "act_bm25" in md and act_docs:
                lists["act_bm25"] = self._bm25(question, md["act_bm25"].get("k", 100), act_docs)
                weights["act_bm25"] = md["act_bm25"].get("weight", 1.0)
            numbers, exact = named_provisions(p, self.store)
            if "exact_ref" in md and exact:
                reps = [int(self.store.rep_of_class[self.store.class_id[u]]) for u in exact]
                lists["exact_ref"] = list(dict.fromkeys(reps))
                weights["exact_ref"] = md["exact_ref"].get("weight", 2.0)
            if "jur_bm25" in md and p.jurisdictions:
                jdocs = [d for j in p.jurisdictions for d in self.jur_docs.get(j, [])]
                if jdocs:
                    lists["jur_bm25"] = self._bm25(question, md["jur_bm25"].get("k", 100), jdocs)
                    weights["jur_bm25"] = md["jur_bm25"].get("weight", 0.5)
            tr.abstain_reason = rule_abstain_reason(p, numbers, exact, ab)
        if signals and cfg.get("llm"):
            self._llm_lists(question, signals, cfg["llm"], md, tr, lists, weights)
        pc = cfg.get("planc") or {}
        if self.acts is not None:
            ro = pc["route"]
            docs = self.acts.route_dense(question, k=ro.get("k_docs", 5))
            tr.routed = docs
            lists["route_bm25"] = self._bm25(question, ro.get("k", 100), docs)
            weights["route_bm25"] = ro.get("weight", 1.0)
            if self.dense is not None and ro.get("dense", True):
                ids, _ = self.dense.search_within(question, self.store.reps_for_docs(docs), k=ro.get("k", 100))
                lists["route_dense"] = [int(i) for i in ids]
                weights["route_dense"] = ro.get("weight", 1.0)
        tr.lists = lists
        if not lists:
            raise ValueError("no retriever configured")
        fused = rrf(lists, weights, k=cfg.get("fusion", {}).get("k", 60)) if len(lists) > 1 else lists[next(iter(lists))]
        xo = pc.get("xref") or {}
        if self.xref is not None and xo.get("mode") == "pre":
            fused = self._expand(fused, xo, scores=None)
        if self.reranker is not None:
            depth = cfg["rerank"].get("depth", 100)
            # exact named-provision hits stay on top: the reranker only orders the rest
            pinned = [u for u in lists.get("exact_ref", [])] if cfg["rerank"].get("pin_exact", True) else []
            pinned_set = set(pinned)
            rest = [u for u in fused if u not in pinned_set]
            reranked, top = self.reranker.rerank(question, rest, depth)
            tr.rerank_top = float(top[0]) if len(top) else None
            tr.ce = {int(u): float(sc) for u, sc in zip(reranked[:len(top)], top)}
            tr.pinned = pinned
            fused = pinned + reranked
            thr = ab.get("rerank_threshold")
            if thr is not None and not tr.abstain_reason and not pinned and (tr.rerank_top is None or tr.rerank_top < thr):
                tr.abstain_reason = "low_rerank_score"
        if self.xref is not None and xo.get("mode") == "post" and not tr.abstain_reason:
            ps = question_parts(question)
            if not xo.get("multipart_only") or ps:
                fused = self._expand(fused, xo, scores=tr.ce, parts=ps if xo.get("part_scores") else None)
        tr.final = fused[:100]
        return fused, tr

    def _provision_rep(self, u: int) -> int | None:
        """Representative of the provision a ranked unit belongs to (sub-unit -> parent; gap window -> None)."""
        s = self.store
        if s.kind[u] == "subunit":
            u = int(s.parent[u])
        elif s.kind[u] != "provision":
            return None
        return int(s.rep_of_class[s.class_id[u]])

    def _expand(self, ranked: list[int], xo: dict, scores: dict[int, float] | None,
                parts: list[str] | None = None) -> list[int]:
        """Cross-reference expansion (Plan C, plan.md Phase 4.4): directly after each of the top-``m`` units,
        insert up to ``max_per`` provisions it cites (and, with ``incoming``, provisions citing it) that are not
        already ranked above. Neighbours are ordered by cross-encoder score when known, else graph order.
        With ``parts`` (sub-questions), neighbours are scored by the cross-encoder against each part and ordered
        by their best part score; ``min_ce`` then drops neighbours below that score."""
        m, max_per = xo.get("m", 3), xo.get("max_per", 2)
        seen_prov = {self._provision_rep(u) for u in ranked[:m]}
        out, placed = [], set()
        for i, u in enumerate(ranked):
            if u in placed:
                continue
            out.append(u)
            placed.add(u)
            if i >= m:
                continue
            p = self._provision_rep(u)
            if p is None:
                continue
            nb = list(self.xref.out.get(p, []))
            if xo.get("incoming"):
                nb += self.xref.inc.get(p, [])
            nb = [v for v in dict.fromkeys(nb) if v not in placed and v not in seen_prov]
            if parts and nb and self.reranker is not None:
                best = np.max(np.stack([self.reranker.scores(x, nb) for x in parts]), axis=0)
                sc = dict(zip(nb, best.tolist()))
                nb = [v for v in nb if sc[v] >= xo.get("min_ce", -1e9)]
                nb.sort(key=lambda v: -sc[v])
            elif scores:
                nb.sort(key=lambda v: -scores.get(v, -1e9))
            for v in nb[:max_per]:
                out.append(v)
                placed.add(v)
                seen_prov.add(v)
        return out

    def _llm_lists(self, question: str, sig: dict, lc: dict, md: dict, tr: Trace,
                   lists: dict[str, list[int]], weights: dict[str, float]) -> None:
        acts = sig.get("act_docs") or []
        kw = " ".join(sig.get("keywords") or [])
        if "act_bm25" in lc and acts:
            lists["llm_act_bm25"] = self._bm25(f"{question} {kw}", lc["act_bm25"].get("k", 100), acts)
            weights["llm_act_bm25"] = lc["act_bm25"].get("weight", 1.0)
        if "act_dense" in lc and acts and self.dense is not None:
            ids, _ = self.dense.search_within(question, self.store.reps_for_docs(acts), k=lc["act_dense"].get("k", 100))
            lists["llm_act_dense"] = [int(i) for i in ids]
            weights["llm_act_dense"] = lc["act_dense"].get("weight", 1.0)
        if "keywords_bm25" in lc and kw:
            lists["llm_kw_bm25"] = self._bm25(kw, lc["keywords_bm25"].get("k", 300))
            weights["llm_kw_bm25"] = lc["keywords_bm25"].get("weight", 0.5)
        if "hyde" in lc and sig.get("hyde") and self.dense is not None:
            ids, _ = self.dense.search([sig["hyde"]], k=lc["hyde"].get("k", 300), as_document=True)[0]
            lists["hyde"] = [int(i) for i in ids]
            weights["hyde"] = lc["hyde"].get("weight", 1.0)
        j = sig.get("jurisdiction")
        if lc.get("jurisdiction") and j and "jur_bm25" in md and "jur_bm25" not in lists:
            jdocs = self.jur_docs.get(j, [])
            if jdocs:
                lists["jur_bm25"] = self._bm25(question, md["jur_bm25"].get("k", 100), jdocs)
                weights["jur_bm25"] = md["jur_bm25"].get("weight", 0.5)

    def close_gpu(self) -> None:
        """Release GPU models (dense index, embedder, cross-encoder) so the LLM can use the GPU."""
        if self.dense is not None:
            self.dense.close()
            self.dense = None
        if self.reranker is not None:
            self.reranker.close()
        if self.acts is not None:
            self.acts.close()

    def run(self, question: str, budgets: Sequence[int] = BUDGETS) -> dict[int, list]:
        return self.run_traced(question, budgets)[0]

    def run_traced(self, question: str, budgets: Sequence[int] = BUDGETS) -> tuple[dict[int, list], Trace]:
        ranked, tr = self.rank(question)
        if tr.abstain_reason:
            return {b: [] for b in budgets}, tr
        p = self.config.get("pack", {})
        return {b: pack(ranked, b, self.store, min_piece=p.get("min_piece", 200)) for b in budgets}, tr
