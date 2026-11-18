"""Phase 6 (plan.md): E -- exploratory agentic upper bound with a local 8B model. DEV ONLY (test is spent).

The agent (qwen3:8b, one schema-constrained JSON action per step, temperature 0, cached) answers a question by
exploring the corpus with four tools:
    search(query, act?)  hybrid BM25 + dense (CPU) over units, optionally restricted to a named Act -> 8 hits
    toc(id)              provisions of the document that handle id belongs to (numbered list of headings)
    read(id)             full text of a provision (<= READ_CHARS) + the provisions it cites (Phase 4 xref graph)
    answer(...)          final answer text, cited handles, abstain flag
Every hit gets a short handle ("P12") so the model never copies long ids. The episode starts from the frozen
C-full top results (computed on the GPU before the LLM loads); the agent's own searches run on the CPU because the
8 GB GPU cannot hold the LLM and the retrieval models at once (no cross-encoder inside the loop).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from isr.corpus.units import normalise
from isr.query.rules import _match_acts
from isr.store import UnitStore

MAX_STEPS = 6
HITS = 8
SNIPPET = 220
READ_CHARS = 3000
TOC_MAX = 60
AGENT_V = "agent-v1"

SYSTEM = (
    "You are an expert on Indian statutory law answering questions from a corpus of Indian Acts, Rules and "
    "Regulations. You work in steps; each step you output exactly one JSON action:\n"
    '  {"action": "search", "query": "...", "act": "optional Act title to search inside"}\n'
    '  {"action": "toc", "id": "P3"}        list the provisions of the document containing P3\n'
    '  {"action": "read", "id": "P3"}       read the full text of provision P3\n'
    '  {"action": "answer", "answer": "...", "citations": ["P3"], "abstain": false}\n'
    "Rules: read a provision before citing it; cite only provisions you read and relied on. The answer is 1 to 3 "
    "sentences using the provisions' own wording, with the decisive phrase copied verbatim in double quotes. "
    "Questions may paraphrase the Act's name or describe a situation; search with statutory terms. If the "
    "question asks about a law or provision that is not in the corpus (e.g. foreign law, case law, a section "
    "number the Act does not have), answer with \"abstain\": true. You have at most "
    f"{MAX_STEPS} steps; be efficient.")

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["search", "toc", "read", "answer"]},
        "query": {"type": "string"}, "act": {"type": "string"}, "id": {"type": "string"},
        "answer": {"type": "string"}, "citations": {"type": "array", "items": {"type": "string"}},
        "abstain": {"type": "boolean"},
    },
    "required": ["action"],
}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": ["answer"]}, "answer": {"type": "string"},
                   "citations": {"type": "array", "items": {"type": "string"}}, "abstain": {"type": "boolean"}},
    "required": ["action", "answer", "citations", "abstain"],
}


class CpuSearch:
    """BM25 + dense over representative units, on the CPU (dense: fp32 matrix in RAM, CPU query encoder)."""

    def __init__(self, store: UnitStore, model: str = "qwen3-0.6b") -> None:
        from sentence_transformers import SentenceTransformer
        from isr.index.bm25 import BM25
        from isr.index.dense import MODELS, index_dir
        self.store = store
        self.bm25 = BM25()
        d = index_dir(model)
        self.unit_ids = np.load(d / "unit_ids.npy")
        self.emb = np.asarray(np.load(d / "emb.npy", mmap_mode="r"), dtype=np.float32)
        self.enc = SentenceTransformer(MODELS[model]["id"], device="cpu")
        self.enc.max_seq_length = 512
        self.prefix = MODELS[model]["q"]
        self.row_of = np.full(int(self.unit_ids.max()) + 1, -1, np.int64)
        self.row_of[self.unit_ids] = np.arange(len(self.unit_ids))

    def search(self, query: str, docs: list[str] | None = None, k: int = 100) -> list[int]:
        mask_rows = None
        if docs:
            reps = self.store.reps_for_docs(docs)
            mask_rows = self.row_of[reps[reps < len(self.row_of)]]
            mask_rows = mask_rows[mask_rows >= 0]
        # BM25
        bm_mask = None
        if docs:
            bm_mask = np.zeros(len(self.bm25.unit_ids), bool)
            r = self.bm25.row_of()[self.store.reps_for_docs(docs)]
            bm_mask[r[r >= 0]] = True
        b_ids, _ = self.bm25.search(query, k=k, mask=bm_mask)
        # dense
        q = self.enc.encode([self.prefix + query], normalize_embeddings=True, convert_to_numpy=True)[0]
        if mask_rows is not None:
            sims = self.emb[mask_rows] @ q
            top = np.argsort(-sims)[:k]
            d_ids = self.unit_ids[mask_rows[top]]
        else:
            sims = self.emb @ q
            top = np.argpartition(-sims, k)[:k]
            d_ids = self.unit_ids[top[np.argsort(-sims[top])]]
        score: dict[int, float] = {}
        for ranked in (b_ids, d_ids):
            for r, u in enumerate(ranked):
                score[int(u)] = score.get(int(u), 0.0) + 1.0 / (60 + r + 1)
        return sorted(score, key=lambda u: (-score[u], u))


@dataclass
class Episode:
    steps: list[dict] = field(default_factory=list)
    read: list[int] = field(default_factory=list)
    searched: list[int] = field(default_factory=list)
    answer: dict | None = None


class Agent:
    def __init__(self, llm, search: CpuSearch, xref=None) -> None:
        self.llm, self.search, self.store, self.xref = llm, search, search.store, xref

    # --- presentation -------------------------------------------------------------------------------
    def _provision(self, u: int) -> int:
        s = self.store
        if s.kind[u] == "subunit":
            return int(s.parent[u])
        return int(u)

    def _handle(self, u: int, h: dict, rev: dict) -> str:
        if u not in rev:
            rev[u] = f"P{len(rev) + 1}"
            h[rev[u]] = u
        return rev[u]

    def _hits(self, units: list[int], h: dict, rev: dict, n: int = HITS) -> str:
        s, lines, seen = self.store, [], set()
        for u in units:
            p = self._provision(u)
            if s.kind[p] != "provision" or p in seen:
                continue
            seen.add(p)
            snip = " ".join(s.text(p)[:SNIPPET].split())
            lines.append(f"{self._handle(p, h, rev)} | {s.title(s.doc[p])} | {s.label[p]} | {snip}")
            if len(lines) >= n:
                break
        return "\n".join(lines) or "(no results)"

    # --- tools --------------------------------------------------------------------------------------
    def _search(self, a: dict, h: dict, rev: dict, ep: Episode) -> str:
        query = str(a.get("query") or "").strip()
        if not query:
            return "error: search needs a non-empty \"query\""
        docs, note = None, ""
        act = str(a.get("act") or "").strip()
        if act:
            docs = [d for m in _match_acts(normalise(act)) for d in m["document_ids"]]
            if not docs:
                note = f"(no Act titled like {act!r} in the corpus; searched everything)\n"
        units = self.search.search(query, docs or None)
        ep.searched += [u for u in units[:HITS * 3] if u not in ep.searched]
        return note + self._hits(units, h, rev)

    def _toc(self, a: dict, h: dict, rev: dict) -> str:
        u = h.get(str(a.get("id", "")).strip())
        if u is None:
            return "error: unknown id"
        s = self.store
        did = s.doc[u]
        provs = [int(x) for x in s.doc_units(did) if s.kind[x] == "provision"]
        lines = [f"{self._handle(p, h, rev)} {s.label[p]}" for p in provs[:TOC_MAX]]
        more = f"\n... ({len(provs) - TOC_MAX} more)" if len(provs) > TOC_MAX else ""
        return f"{s.title(did)}:\n" + "\n".join(lines) + more

    def _read(self, a: dict, h: dict, rev: dict, ep: Episode) -> str:
        u = h.get(str(a.get("id", "")).strip())
        if u is None:
            return "error: unknown id"
        s = self.store
        if u not in ep.read:
            ep.read.append(u)
        out = f"{s.title(s.doc[u])} — {s.label[u]}\n{s.text(u)[:READ_CHARS]}"
        if self.xref is not None:
            rep = int(s.rep_of_class[s.class_id[u]])
            cites = self.xref.out.get(rep, [])[:6]
            if cites:
                out += "\nCites: " + "; ".join(f"{self._handle(c, h, rev)} {s.label[c]}" for c in cites)
        return out

    # --- loop ---------------------------------------------------------------------------------------
    def run(self, question: str, initial: list[int]) -> Episode:
        h: dict[str, int] = {}
        rev: dict[int, str] = {}
        ep = Episode()
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Question: {question}\n\nInitial search results:\n"
                                            f"{self._hits(initial, h, rev)}"}]
        for step in range(1, MAX_STEPS + 1):
            last = step == MAX_STEPS
            if last:
                msgs[-1]["content"] += "\n\nThis is your last step: output the answer action now."
            r = self.llm.chat(msgs, version=AGENT_V, schema=ANSWER_SCHEMA if last else SCHEMA, max_tokens=400)
            a = r.parsed if isinstance(r.parsed, dict) else {"action": "answer", "answer": "", "citations": [],
                                                             "abstain": True}
            msgs.append({"role": "assistant", "content": json.dumps(a, ensure_ascii=False)})
            act = a.get("action")
            if act == "answer":
                cites = [h[c] for c in a.get("citations") or [] if c in h]
                ep.answer = {"text": str(a.get("answer") or ""), "citations": cites, "abstain": bool(a.get("abstain"))}
                ep.steps.append({"action": "answer", "n_cites": len(cites)})
                break
            if act == "search":
                obs = self._search(a, h, rev, ep)
            elif act == "toc":
                obs = self._toc(a, h, rev)
            elif act == "read":
                obs = self._read(a, h, rev, ep)
            else:
                obs = "error: unknown action"
            ep.steps.append({k: a.get(k) for k in ("action", "query", "act", "id") if a.get(k)})
            msgs.append({"role": "user", "content": f"Observation:\n{obs}"})
        return ep
