"""Lightweight HTTP JSON API bridge for the CaseWise React frontend.

    python scripts/serve_api.py [--host 127.0.0.1 --port 8000]

Serves:
  GET  /api/health              KB + Ollama readiness and frozen system metadata
  GET  /api/report              Frozen C-full test metrics from runs/final/C-full/
  GET  /api/examples            Curated benchmark examples paired with C-full predictions
  GET  /api/document?id=<did>   Full canonical statute text & metadata (when corpus/ is present)
  POST /api/query               Runs live C_full retrieval + qwen3:8b cited answer (when KB + Ollama ready)
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from isr import CORPUS_DIR, DATA_DIR, INDEX_DIR, KIT_DIR, ROOT, RUNS_DIR  # noqa: E402

SAMPLE_QIDS = [
    "IN-STAT-0209",  # provision_lookup
    "IN-STAT-0093",  # cross_reference
    "IN-STAT-0161",  # jurisdictional
    "IN-STAT-0059",  # numeric_threshold
    "IN-STAT-0196",  # situational
    "IN-STAT-0014",  # definitional
    "IN-STAT-0339",  # unanswerable
]


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


class Runtime:
    """Lazy, thread-safe holder for the C_full pipeline + Ollama client."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self._lock = threading.Lock()
        self._pipe = None
        self._llm = None
        self._cfg = None
        self._load_error: str | None = None

    def ensure_loaded(self):
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
            self._pipe = Pipeline(cfg, dense=CpuDenseIndex(cfg["retrieve"]["dense"]["model"]))
            an, lc = cfg["answer"], cfg.get("llm") or {}
            self._llm = LLM(an.get("model", lc.get("model", "qwen3:8b")), num_ctx=an.get("num_ctx", 8192))

    def query(self, question: str, *, budget: int = 8000, jurisdiction: str | None = None,
              doc_type: str | None = None) -> dict:
        from isr.llm import stages as S
        from isr.pack import pack
        from run_b import fmt_passages, key_for_span

        self.ensure_loaded()
        effective_q = question.strip()
        if jurisdiction and jurisdiction != "All" and jurisdiction.lower() not in effective_q.lower():
            effective_q = f"{effective_q} (Jurisdiction: {jurisdiction})"
        if doc_type and doc_type != "All" and doc_type.lower() not in effective_q.lower():
            effective_q = f"{effective_q} ({doc_type})"

        t0 = time.time()
        with self._lock:
            store = self._pipe.store
            an = self._cfg["answer"]
            ranked, tr = self._pipe.rank(effective_q)
            if tr.abstain_reason:
                return {
                    "mode": "live",
                    "question": question,
                    "effective_question": effective_q,
                    "abstained": True,
                    "reason": tr.abstain_reason,
                    "answer": "",
                    "cited": [],
                    "passages": [],
                    "trace": _serialize_trace(tr, store, ranked[:8]),
                    "seconds": round(time.time() - t0, 2),
                }
            spans = pack(ranked, budget, store, min_piece=self._cfg.get("pack", {}).get("min_piece", 200))
            passages = _rich_passages(store, spans, self._pipe.xref, tr, an.get("passages", 8), an.get("chars", 2500))
            o = S.answer(
                self._llm,
                question,
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
                "question": question,
                "effective_question": effective_q,
                "abstained": bool(o["abstain"] or not text.strip()),
                "reason": "llm" if o["abstain"] else None,
                "answer": text,
                "cited_numbers": cited_indices,
                "cited": [{"document_id": p["document_id"], "key": p["key"], "title": p["title"], "label": p["label"]}
                          for p in cited],
                "passages": passages,
                "trace": _serialize_trace(tr, store, ranked[:8]),
                "seconds": round(time.time() - t0, 2),
            }

    def document(self, document_id: str) -> dict | None:
        text_path = CORPUS_DIR / "text" / f"{document_id}.txt"
        if not text_path.is_file():
            return None
        text = text_path.read_text(encoding="utf-8", errors="replace")
        meta = {}
        if self._pipe is not None and document_id in self._pipe.store.docs.index:
            row = self._pipe.store.docs.loc[document_id]
            meta = {
                "title": str(row.get("title") or document_id),
                "jurisdiction": str(row.get("jurisdiction") or ""),
                "document_type": str(row.get("document_type") or ""),
                "year": None if row.get("year") is None or str(row.get("year")) == "<NA>" else int(row.get("year")),
                "enactment_date": str(row.get("enactment_date") or ""),
            }
        else:
            meta_path = CORPUS_DIR / "meta" / f"{document_id}.json"
            if meta_path.is_file():
                m = json.loads(meta_path.read_text(encoding="utf-8"))
                meta = {
                    "title": m.get("title") or document_id,
                    "jurisdiction": m.get("jurisdiction") or "",
                    "document_type": m.get("document_type") or "",
                    "year": m.get("year"),
                    "enactment_date": str(m.get("enactment_date") or ""),
                }
        return {"document_id": document_id, **meta, "char_count": len(text), "text": text[:200_000]}


def _rich_passages(store, spans, xref, tr, max_items: int, max_chars: int) -> list[dict]:
    from run_b import key_for_span

    pinned_set = set(tr.pinned)
    out = []
    for idx, (did, s, e) in enumerate(spans[:max_items], 1):
        key, uid = key_for_span(store, did, s, e)
        label = str(store.label[uid]) if uid is not None else "text"
        ce = tr.ce.get(uid) if uid is not None else None
        xref_links = []
        if xref is not None and uid is not None:
            rep = int(store.rep_of_class[store.class_id[uid]])
            for target_uid in (xref.out.get(rep, [])[:3]):
                xref_links.append({
                    "direction": "cites",
                    "document_id": str(store.doc[target_uid]),
                    "label": str(store.label[target_uid]),
                    "title": store.title(store.doc[target_uid]),
                })
        doc_row = store.docs.loc[did] if did in store.docs.index else None
        out.append({
            "index": idx,
            "document_id": did,
            "key": key,
            "title": store.title(did),
            "label": label,
            "jurisdiction": str(doc_row["jurisdiction"]) if doc_row is not None else "",
            "document_type": str(doc_row["document_type"]) if doc_row is not None else "",
            "year": int(doc_row["year"]) if doc_row is not None and str(doc_row["year"]) not in ("<NA>", "nan", "None") else None,
            "char_start": int(s),
            "char_end": int(e),
            "ce_score": round(float(ce), 4) if ce is not None else None,
            "pinned": bool(uid is not None and uid in pinned_set),
            "xref": xref_links,
            "text": store.doc_text(did)[s:e][:max_chars],
        })
    return out


def _serialize_trace(tr, store, top_units) -> dict:
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


def _load_examples() -> list[dict]:
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


def make_handler(runtime: Runtime):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict | list) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/health":
                kb = _kb_ready()
                ol = _ollama_up()
                self._send(200, {
                    "status": "ok",
                    "mode": "live" if (kb and ol) else "demo",
                    "kb_ready": kb,
                    "ollama_up": ol,
                    "config": "configs/C_full.yaml",
                    "models": {
                        "embedder": "Qwen/Qwen3-Embedding-0.6B",
                        "reranker": "BAAI/bge-reranker-v2-m3",
                        "llm": "qwen3:8b",
                    },
                })
                return
            if parsed.path == "/api/examples":
                self._send(200, {"examples": _load_examples()})
                return
            if parsed.path == "/api/report":
                rep = RUNS_DIR / "final" / "C-full" / "pred_test.report.json"
                data = json.loads(rep.read_text(encoding="utf-8")) if rep.is_file() else {}
                self._send(200, data)
                return
            if parsed.path == "/api/document":
                qs = urllib.parse.parse_qs(parsed.query)
                did = (qs.get("id") or [""])[0]
                if not did:
                    self._send(400, {"error": "missing id"})
                    return
                doc = runtime.document(did)
                if doc is None:
                    self._send(404, {"error": "document not available on disk (corpus/text missing)"})
                    return
                self._send(200, doc)
                return
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/api/query":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._send(400, {"error": "invalid JSON"})
                return
            question = str(body.get("question") or "").strip()
            if not question:
                self._send(400, {"error": "question is required"})
                return
            if not (_kb_ready() and _ollama_up()):
                self._send(503, {
                    "error": "live_kb_unavailable",
                    "kb_ready": _kb_ready(),
                    "ollama_up": _ollama_up(),
                    "message": "corpus/, data/, indexes/, and Ollama must be available for live queries.",
                })
                return
            try:
                res = runtime.query(
                    question,
                    budget=int(body.get("budget") or 8000),
                    jurisdiction=body.get("jurisdiction"),
                    doc_type=body.get("doc_type"),
                )
                self._send(200, res)
            except Exception as exc:
                self._send(500, {"error": str(exc)})

        def log_message(self, format: str, *args) -> None:
            print(f"[api] {self.address_string()} - {format % args}", flush=True)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "C_full.yaml")
    args = ap.parse_args()
    runtime = Runtime(args.config)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(f"CaseWise API listening on http://{args.host}:{args.port} (kb_ready={_kb_ready()}, ollama={_ollama_up()})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
