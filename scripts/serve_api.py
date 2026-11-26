"""Lightweight HTTP JSON API bridge for the CaseWise React frontend.

    python scripts/serve_api.py [--host 127.0.0.1 --port 8000]

Serves:
  GET    /api/health                 KB + Ollama readiness, reranker device, frozen system metadata
  GET    /api/filters                Available jurisdictions and instrument types
  GET    /api/report                 Frozen C-full test metrics from runs/final/C-full/
  GET    /api/examples               Curated benchmark examples paired with C-full predictions
  GET    /api/document?id=<did>      Statute text (optional ?start=&end=&pad= windowing) & metadata
  GET    /api/provision              Single provision lookup (?document_id=&key= or ?uid=) + xrefs
  POST   /api/query                  Live C_full retrieval + follow-up contextualization + qwen3:8b answer
  GET    /api/chats                  List persistent chat threads
  POST   /api/chats                  Create a new chat thread
  GET    /api/chats/<id>             Get a chat thread and all turns
  PATCH  /api/chats/<id>             Rename a chat thread
  DELETE /api/chats[/<id>]           Delete one or all chat threads
  GET    /*                          Serves built frontend/dist/ when present
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Sequence

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parent / "src"
for _p in (str(HERE), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isr import CORPUS_DIR, DATA_DIR, INDEX_DIR, KIT_DIR, ROOT, RUNS_DIR  # noqa: E402
from isr.history import DEFAULT_HISTORY_DB, ChatStore  # noqa: E402

DIST_DIR = ROOT / "frontend" / "dist"

SAMPLE_QIDS = [
    "IN-STAT-0209",  # provision_lookup
    "IN-STAT-0093",  # cross_reference
    "IN-STAT-0161",  # jurisdictional
    "IN-STAT-0059",  # numeric_threshold
    "IN-STAT-0196",  # situational
    "IN-STAT-0014",  # definitional
    "IN-STAT-0339",  # unanswerable
]

_DOC_TYPE_MAP = {
    "central act": "central_act",
    "central_act": "central_act",
    "central_acts": "central_act",
    "state act": "state_act",
    "state_act": "state_act",
    "state_acts": "state_act",
    "rule": "rule",
    "rules": "rule",
    "regulation": "regulation",
    "regulations": "regulation",
    "ordinance": "ordinance",
}


def _kb_ready() -> bool:
    return (
        (CORPUS_DIR / "text").is_dir()
        and (DATA_DIR / "units").is_dir()
        and (DATA_DIR / "docs.parquet").is_file()
        and (INDEX_DIR / "bm25_units").is_dir()
        and (INDEX_DIR / "dense_qwen3-0.6b").is_dir()
    )


def _ollama_up() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2):
            return True
    except Exception:
        return False


def _norm_jurisdiction(val: Optional[str]) -> Optional[str]:
    v = (val or "").strip()
    if not v or v.lower() == "all":
        return None
    if v.lower() == "central":
        return "India"
    return v


def _norm_doc_type(val: Optional[str]) -> Optional[str]:
    v = (val or "").strip().lower()
    if not v or v == "all":
        return None
    return _DOC_TYPE_MAP.get(v, v.replace(" ", "_"))


class Runtime:
    """Lazy, thread-safe holder for the C_full pipeline + Ollama client."""

    def __init__(self, config_path: Path, rerank_device: str = "auto") -> None:
        self.config_path = config_path
        self.rerank_device = rerank_device
        self._lock = threading.Lock()
        self._pipe = None
        self._llm = None
        self._cfg = None

    def ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        with self._lock:
            if self._pipe is not None:
                return
            import yaml
            from isr.index.dense_cpu import CpuDenseIndex
            from isr.llm.client import LLM
            from isr.pipeline import Pipeline

            cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
            self._cfg = cfg
            self._pipe = Pipeline(
                cfg,
                dense=CpuDenseIndex(cfg["retrieve"]["dense"]["model"]),
                rerank_device=self.rerank_device,
            )
            an, lc = cfg["answer"], cfg.get("llm") or {}
            self._llm = LLM(an.get("model", lc.get("model", "qwen3:8b")), num_ctx=an.get("num_ctx", 8192))

    @property
    def active_rerank_device(self) -> str:
        if self._pipe is not None and self._pipe.reranker is not None:
            return self._pipe.reranker.device
        return self.rerank_device

    def filter_docs(self, jurisdiction: Optional[str], doc_type: Optional[str]) -> Optional[list[str]]:
        """Return matching document_ids for UI filters, or None when no filter is active."""
        jur = _norm_jurisdiction(jurisdiction)
        dtype = _norm_doc_type(doc_type)
        if jur is None and dtype is None:
            return None
        docs = self._pipe.store.docs
        mask = docs.index.notna()
        if jur is not None:
            j_col = docs["jurisdiction"].fillna("").astype(str).str.strip().str.lower()
            targets = {"india", "central"} if jur == "India" else {jur.lower()}
            mask = mask & j_col.isin(targets)
        if dtype is not None:
            dt_col = docs["document_type"].fillna("").astype(str).str.strip().str.lower()
            cat_col = docs["category"].fillna("").astype(str).str.strip().str.lower()
            mask = mask & ((dt_col == dtype) | (cat_col == dtype) | (cat_col == f"{dtype}s"))
        return [str(d) for d in docs.index[mask]]

    def filters(self) -> dict[str, list[str]]:
        if self._pipe is None:
            return {"jurisdictions": [], "doc_types": []}
        docs = self._pipe.store.docs
        jurs = sorted({str(x).strip() for x in docs["jurisdiction"].dropna() if str(x).strip()})
        dtypes = sorted({str(x).strip() for x in docs["document_type"].dropna() if str(x).strip()})
        return {"jurisdictions": jurs, "doc_types": dtypes}

    def query(
        self,
        question: str,
        *,
        budget: int = 8000,
        jurisdiction: Optional[str] = None,
        doc_type: Optional[str] = None,
        prior_turns: Sequence[dict[str, str]] = (),
    ) -> dict[str, Any]:
        from isr.llm import stages as S
        from isr.pack import pack
        from run_b import fmt_passages

        self.ensure_loaded()
        budget = max(500, min(32000, int(budget or 8000)))
        clean_q = question.strip()

        t0 = time.time()
        with self._lock:
            store = self._pipe.store
            an = self._cfg["answer"]
            effective_q = clean_q
            if prior_turns and S.needs_contextualize(clean_q, prior_turns):
                effective_q = S.contextualize(self._llm, clean_q, prior_turns)

            allowed_docs = self.filter_docs(jurisdiction, doc_type)
            ranked, tr = self._pipe.rank(effective_q, allowed_docs=allowed_docs)
            if tr.abstain_reason:
                return {
                    "mode": "live",
                    "question": clean_q,
                    "effective_question": effective_q,
                    "jurisdiction": jurisdiction or "All",
                    "doc_type": doc_type or "All",
                    "budget": budget,
                    "abstained": True,
                    "reason": tr.abstain_reason,
                    "answer": "",
                    "cited_numbers": [],
                    "cited": [],
                    "passages": [],
                    "trace": _serialize_trace(tr, store, ranked[:8]),
                    "seconds": round(time.time() - t0, 2),
                }

            member_of = _member_override(store, ranked, allowed_docs) if allowed_docs is not None else None
            spans = pack(
                ranked,
                budget,
                store,
                min_piece=self._cfg.get("pack", {}).get("min_piece", 200),
                member_of=member_of,
            )
            passages = _rich_passages(store, spans, self._pipe.xref, tr, an.get("passages", 8), an.get("chars", 2500))
            o = S.answer(
                self._llm,
                effective_q,
                fmt_passages(passages),
                prompt=an.get("prompt", "v1"),
                think=an.get("think", False),
                max_tokens=an.get("max_tokens", 400),
            )
            cited_indices = [i for i in o["citations"] if 1 <= i <= len(passages) and passages[i - 1]["key"]]
            cited = [passages[i - 1] for i in cited_indices]
            text = S.ground_quotes(o["answer"], [p["text"] for p in cited])
            return {
                "mode": "live",
                "question": clean_q,
                "effective_question": effective_q,
                "jurisdiction": jurisdiction or "All",
                "doc_type": doc_type or "All",
                "budget": budget,
                "abstained": bool(o["abstain"] or not text.strip()),
                "reason": "llm" if o["abstain"] else None,
                "answer": text,
                "cited_numbers": cited_indices,
                "cited": [
                    {"document_id": p["document_id"], "key": p["key"], "title": p["title"], "label": p["label"]}
                    for p in cited
                ],
                "passages": passages,
                "trace": _serialize_trace(tr, store, ranked[:8]),
                "seconds": round(time.time() - t0, 2),
            }

    def document(
        self,
        document_id: str,
        *,
        start: Optional[int] = None,
        end: Optional[int] = None,
        pad: int = 1200,
        full: bool = False,
    ) -> Optional[dict[str, Any]]:
        if "/" in document_id or "\\" in document_id or ".." in document_id:
            return None
        text_path = CORPUS_DIR / "text" / f"{document_id}.txt"
        if not text_path.is_file():
            return None
        full_text = text_path.read_text(encoding="utf-8", errors="replace")
        total = len(full_text)
        meta = self._doc_meta(document_id)

        if start is not None and end is not None:
            ws = max(0, min(total, int(start) - max(0, int(pad))))
            we = max(ws, min(total, int(end) + max(0, int(pad))))
            snippet = full_text[ws:we]
        elif full:
            ws, we, snippet = 0, total, full_text
        else:
            ws = 0
            we = min(total, 500_000)
            snippet = full_text[ws:we]

        return {
            "document_id": document_id,
            **meta,
            "char_count": total,
            "window_start": ws,
            "window_end": we,
            "truncated": ws > 0 or we < total,
            "text": snippet,
        }

    def provision(
        self,
        *,
        document_id: Optional[str] = None,
        key: Optional[str] = None,
        uid: Optional[int] = None,
        max_chars: int = 4000,
    ) -> Optional[dict[str, Any]]:
        self.ensure_loaded()
        store = self._pipe.store
        target_uid: Optional[int] = None
        if uid is not None and 0 <= uid < len(store):
            target_uid = int(uid)
        elif document_id and key:
            du = store.doc_units(document_id)
            keys = store.units["key"].to_numpy(dtype=object)
            for u in du:
                if store.kind[u] == "provision" and keys[u] == key:
                    target_uid = int(u)
                    break
        if target_uid is None:
            return None

        did, s, e = store.span(target_uid)
        doc_row = store.docs.loc[did] if did in store.docs.index else None
        ukey = str(store.units.at[target_uid, "key"])
        return {
            "index": 1,
            "document_id": did,
            "key": ukey,
            "title": store.title(did),
            "label": str(store.label[target_uid]),
            "jurisdiction": str(doc_row["jurisdiction"]) if doc_row is not None else "",
            "document_type": str(doc_row["document_type"]) if doc_row is not None else "",
            "year": _safe_year(doc_row["year"]) if doc_row is not None else None,
            "char_start": int(s),
            "char_end": int(e),
            "ce_score": None,
            "pinned": False,
            "xref": _xref_links(store, self._pipe.xref, target_uid),
            "text": store.doc_text(did)[s:e][:max_chars],
        }

    def _doc_meta(self, document_id: str) -> dict[str, Any]:
        if self._pipe is not None and document_id in self._pipe.store.docs.index:
            row = self._pipe.store.docs.loc[document_id]
            return {
                "title": str(row.get("title") or document_id),
                "jurisdiction": str(row.get("jurisdiction") or ""),
                "document_type": str(row.get("document_type") or ""),
                "year": _safe_year(row.get("year")),
                "enactment_date": str(row.get("enactment_date") or ""),
            }
        meta_path = CORPUS_DIR / "meta" / f"{document_id}.json"
        if meta_path.is_file():
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            return {
                "title": m.get("title") or document_id,
                "jurisdiction": m.get("jurisdiction") or "",
                "document_type": m.get("document_type") or "",
                "year": m.get("year"),
                "enactment_date": str(m.get("enactment_date") or ""),
            }
        return {"title": document_id, "jurisdiction": "", "document_type": "", "year": None, "enactment_date": ""}


def _safe_year(val: Any) -> Optional[int]:
    if val is None or str(val) in ("<NA>", "nan", "None", ""):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _member_override(store, ranked: Sequence[int], allowed_docs: Sequence[str]) -> dict[int, int]:
    """Ensure dedup-class representatives map to a member inside ``allowed_docs`` when filtered."""
    allowed_set = set(allowed_docs)
    override: dict[int, int] = {}
    seen_cids: set[int] = set()
    for u in ranked[:400]:
        cid = int(store.class_id[int(u)])
        if cid in seen_cids:
            continue
        seen_cids.add(cid)
        if store.doc[int(u)] in allowed_set:
            continue
        for m in store.class_members(cid):
            if store.doc[int(m)] in allowed_set:
                override[cid] = int(m)
                break
    return override


def _xref_links(store, xref, uid: Optional[int]) -> list[dict[str, Any]]:
    if xref is None or uid is None:
        return []
    rep = int(store.rep_of_class[store.class_id[uid]])
    keys = store.units["key"].to_numpy(dtype=object)
    links: list[dict[str, Any]] = []
    seen: set[int] = set()
    for direction, pool in (("cites", xref.out.get(rep, [])[:4]), ("cited_by", xref.inc.get(rep, [])[:4])):
        for target_uid in pool:
            t = int(target_uid)
            if t in seen:
                continue
            seen.add(t)
            did, s, e = store.span(t)
            links.append({
                "direction": direction,
                "uid": t,
                "document_id": str(did),
                "key": str(keys[t]),
                "char_start": int(s),
                "char_end": int(e),
                "label": str(store.label[t]),
                "title": store.title(did),
            })
    return links


def _rich_passages(store, spans, xref, tr, max_items: int, max_chars: int) -> list[dict[str, Any]]:
    from run_b import key_for_span

    pinned_set = set(tr.pinned)
    out = []
    for idx, (did, s, e) in enumerate(spans[:max_items], 1):
        key, uid = key_for_span(store, did, s, e)
        label = str(store.label[uid]) if uid is not None else "text"
        ce = tr.ce.get(uid) if uid is not None else None
        doc_row = store.docs.loc[did] if did in store.docs.index else None
        out.append({
            "index": idx,
            "document_id": did,
            "key": key,
            "title": store.title(did),
            "label": label,
            "jurisdiction": str(doc_row["jurisdiction"]) if doc_row is not None else "",
            "document_type": str(doc_row["document_type"]) if doc_row is not None else "",
            "year": _safe_year(doc_row["year"]) if doc_row is not None else None,
            "char_start": int(s),
            "char_end": int(e),
            "ce_score": round(float(ce), 4) if ce is not None else None,
            "pinned": bool(uid is not None and uid in pinned_set),
            "xref": _xref_links(store, xref, uid),
            "text": store.doc_text(did)[s:e][:max_chars],
        })
    return out


def _serialize_trace(tr, store, top_units) -> dict[str, Any]:
    p = tr.parsed
    return {
        "abstain_reason": tr.abstain_reason,
        "rerank_top": round(float(tr.rerank_top), 4) if tr.rerank_top is not None else None,
        "parsed": {
            "acts": [a["title"] for a in p.acts[:4]] if p else [],
            "refs": [f"{k} {n}" for k, n in p.refs] if p else [],
            "jurisdictions": list(p.jurisdictions) if p else [],
            "named_law_unresolved": bool(p.named_law_unresolved) if p else False,
        },
        "lists": {k: len(v) for k, v in tr.lists.items()},
        "top_hits": [
            {
                "title": store.title(store.doc[int(u)]),
                "label": str(store.label[int(u)]),
                "ce_score": round(float(tr.ce[int(u)]), 4) if int(u) in tr.ce else None,
            }
            for u in top_units[:5]
        ],
    }


def _load_examples() -> list[dict[str, Any]]:
    pred_path = RUNS_DIR / "final" / "C-full" / "pred_test.json"
    q_dir = KIT_DIR / "benchmark" / "data" / "questions"
    if not pred_path.is_file() or not q_dir.is_dir():
        return []
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    out = []
    for qid in SAMPLE_QIDS:
        qf = q_dir / f"{qid}.json"
        if not qf.is_file():
            continue
        q = json.loads(qf.read_text(encoding="utf-8"))
        ans = preds.get("answers", {}).get(qid, {})
        spans = preds.get("predictions", {}).get(qid, {}).get("8000", [])
        out.append({
            "question_id": qid,
            "category": q["category"],
            "question": q["question"],
            "answer": ans,
            "spans": spans[:5],
            "gold_citations": [l["citation"] for g in q.get("gold_evidence", []) for l in g.get("locations", [])[:1]],
        })
    return out


def _int_param(qs: dict[str, list[str]], name: str) -> Optional[int]:
    raw = (qs.get(name) or [None])[0]
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def make_handler(
    runtime: Runtime,
    history_store: ChatStore,
    *,
    kb_check=_kb_ready,
    ollama_check=_ollama_up,
    dist_dir: Path = DIST_DIR,
):
    class Handler(BaseHTTPRequestHandler):
        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _send(self, status: int, payload: dict | list) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, status: int, data: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> Optional[dict]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8"))
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return None

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)

            if path == "/api/health":
                kb = kb_check()
                ol = ollama_check()
                self._send(200, {
                    "status": "ok",
                    "mode": "live" if (kb and ol) else "demo",
                    "kb_ready": kb,
                    "ollama_up": ol,
                    "rerank_device": runtime.active_rerank_device,
                    "config": "configs/C_full.yaml",
                    "models": {
                        "embedder": "Qwen/Qwen3-Embedding-0.6B",
                        "reranker": "BAAI/bge-reranker-v2-m3",
                        "llm": "qwen3:8b",
                    },
                })
                return

            if path == "/api/filters":
                self._send(200, runtime.filters())
                return

            if path == "/api/examples":
                self._send(200, {"examples": _load_examples()})
                return

            if path == "/api/report":
                rep = RUNS_DIR / "final" / "C-full" / "pred_test.report.json"
                data = json.loads(rep.read_text(encoding="utf-8")) if rep.is_file() else {}
                self._send(200, data)
                return

            if path == "/api/document":
                did = (qs.get("id") or [""])[0]
                if not did:
                    self._send(400, {"error": "missing id"})
                    return
                start = _int_param(qs, "start")
                end = _int_param(qs, "end")
                pad = _int_param(qs, "pad")
                full = (qs.get("full") or ["0"])[0] == "1"
                doc = runtime.document(did, start=start, end=end, pad=pad if pad is not None else 1200, full=full)
                if doc is None:
                    self._send(404, {"error": "document not available on disk (corpus/text missing)"})
                    return
                self._send(200, doc)
                return

            if path == "/api/provision":
                if not kb_check():
                    self._send(503, {"error": "live_kb_unavailable", "message": "Knowledge base is not loaded."})
                    return
                uid = _int_param(qs, "uid")
                did = (qs.get("document_id") or [""])[0] or None
                key = (qs.get("key") or [""])[0] or None
                if uid is None and (not did or not key):
                    self._send(400, {"error": "provide uid or both document_id and key"})
                    return
                prov = runtime.provision(document_id=did, key=key, uid=uid)
                if prov is None:
                    self._send(404, {"error": "provision not found"})
                    return
                self._send(200, prov)
                return

            if path == "/api/chats":
                self._send(200, {"chats": history_store.list_chats()})
                return

            if path.startswith("/api/chats/"):
                chat_id = path[len("/api/chats/"):].strip("/")
                if not chat_id or "/" in chat_id:
                    self._send(400, {"error": "invalid chat id"})
                    return
                chat = history_store.get_chat(chat_id)
                if chat is None:
                    self._send(404, {"error": "chat not found"})
                    return
                self._send(200, chat)
                return

            if path.startswith("/api/"):
                self._send(404, {"error": "not found"})
                return

            # Serve built frontend SPA when available
            index_file = dist_dir / "index.html"
            if index_file.is_file():
                rel = path.lstrip("/") or "index.html"
                candidate = (dist_dir / rel).resolve()
                if str(candidate).startswith(str(dist_dir.resolve())) and candidate.is_file():
                    ctype = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
                    self._send_bytes(200, candidate.read_bytes(), ctype)
                    return
                self._send_bytes(200, index_file.read_bytes(), "text/html; charset=utf-8")
                return

            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path == "/api/chats":
                body = self._read_json()
                if body is None:
                    self._send(400, {"error": "invalid JSON"})
                    return
                chat = history_store.create_chat(
                    title=body.get("title"),
                    jurisdiction=body.get("jurisdiction") or "All",
                    doc_type=body.get("doc_type") or "All",
                    budget=int(body.get("budget") or 8000),
                )
                self._send(201, chat)
                return

            if path != "/api/query":
                self._send(404, {"error": "not found"})
                return

            body = self._read_json()
            if body is None:
                self._send(400, {"error": "invalid JSON"})
                return
            question = str(body.get("question") or "").strip()
            if not question:
                self._send(400, {"error": "question is required"})
                return

            kb = kb_check()
            ol = ollama_check()
            if not (kb and ol):
                self._send(503, {
                    "error": "live_kb_unavailable",
                    "kb_ready": kb,
                    "ollama_up": ol,
                    "message": "corpus/, data/, indexes/, and Ollama must be available for live queries.",
                })
                return

            chat_id = body.get("chat_id")
            save_history = body.get("save_history", True)
            prior_turns: list[dict[str, str]] = []
            if chat_id:
                existing = history_store.get_chat(str(chat_id))
                if existing is None:
                    self._send(404, {"error": "chat not found"})
                    return
                prior_turns = history_store.recent_turns(str(chat_id))
            elif isinstance(body.get("history"), list):
                prior_turns = [
                    {"question": str(t.get("question") or ""), "answer": str(t.get("answer") or "")}
                    for t in body["history"][-4:]
                    if isinstance(t, dict) and t.get("question")
                ]

            try:
                budget = int(body.get("budget") or 8000)
                jurisdiction = body.get("jurisdiction") or "All"
                doc_type = body.get("doc_type") or "All"
                res = runtime.query(
                    question,
                    budget=budget,
                    jurisdiction=jurisdiction,
                    doc_type=doc_type,
                    prior_turns=prior_turns,
                )
                if save_history:
                    if not chat_id:
                        chat = history_store.create_chat(
                            jurisdiction=jurisdiction,
                            doc_type=doc_type,
                            budget=budget,
                        )
                        chat_id = chat["id"]
                    saved = history_store.add_message(str(chat_id), res)
                    chat_meta = history_store.get_chat(str(chat_id))
                    res["chat_id"] = str(chat_id)
                    res["message_id"] = saved["id"]
                    res["chat_title"] = chat_meta["title"] if chat_meta else ""
                self._send(200, res)
            except Exception as exc:
                self._send(500, {"error": str(exc)})

        def do_PATCH(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            if not path.startswith("/api/chats/"):
                self._send(404, {"error": "not found"})
                return
            chat_id = path[len("/api/chats/"):].strip("/")
            body = self._read_json()
            if body is None or not str(body.get("title") or "").strip():
                self._send(400, {"error": "valid title is required"})
                return
            updated = history_store.rename_chat(chat_id, str(body["title"]))
            if updated is None:
                self._send(404, {"error": "chat not found"})
                return
            self._send(200, updated)

        def do_DELETE(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            if path == "/api/chats":
                deleted = history_store.clear_chats()
                self._send(200, {"deleted": deleted})
                return
            if path.startswith("/api/chats/"):
                chat_id = path[len("/api/chats/"):].strip("/")
                if not chat_id or "/" in chat_id:
                    self._send(400, {"error": "invalid chat id"})
                    return
                if not history_store.delete_chat(chat_id):
                    self._send(404, {"error": "chat not found"})
                    return
                self._send(200, {"deleted": True, "id": chat_id})
                return
            self._send(404, {"error": "not found"})

        def log_message(self, format: str, *args) -> None:
            print(f"[api] {self.address_string()} - {format % args}", flush=True)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "C_full.yaml")
    ap.add_argument(
        "--rerank-device",
        default=os.environ.get("ISR_RERANK_DEVICE", "auto"),
        choices=["auto", "cuda", "cpu"],
        help="Where to run bge-reranker-v2-m3 ('auto' picks CPU if GPU VRAM < 5 GiB).",
    )
    ap.add_argument("--history-db", type=Path, default=DEFAULT_HISTORY_DB)
    ap.add_argument("--preload", action="store_true", help="Load indexes and models before serving requests.")
    args = ap.parse_args()

    runtime = Runtime(args.config, rerank_device=args.rerank_device)
    if args.preload and _kb_ready() and _ollama_up():
        print("Preloading indexes and models ...", flush=True)
        runtime.ensure_loaded()
    history_store = ChatStore(args.history_db)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(runtime, history_store))
    print(
        f"CaseWise API listening on http://{args.host}:{args.port} "
        f"(kb_ready={_kb_ready()}, ollama={_ollama_up()}, rerank_device={args.rerank_device})"
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        history_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
