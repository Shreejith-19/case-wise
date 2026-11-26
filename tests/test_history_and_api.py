"""Tests for ChatStore, live-query backend helpers, and HTTP API routes."""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import serve_api  # noqa: E402
from isr.history import ChatStore  # noqa: E402
from isr.llm import stages as S  # noqa: E402
from isr.retrieve import rerank  # noqa: E402


def _req(url: str, method: str = "GET", body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_chat_store_crud_and_ordering(tmp_path):
    store = ChatStore(tmp_path / "hist.sqlite")
    try:
        c1 = store.create_chat(jurisdiction="Delhi", doc_type="State Act", budget=8000)
        assert c1["title"] == "New Research"
        m1 = store.add_message(
            c1["id"],
            {
                "question": "What is the limitation period under Section 5?",
                "answer": "Thirty days.",
                "abstained": False,
                "cited": [{"document_id": "d1", "key": "s:5", "title": "Act A", "label": "s. 5"}],
                "passages": [{"document_id": "d1", "key": "s:5", "char_start": 0, "char_end": 20}],
                "seconds": 1.25,
            },
        )
        assert m1["chat_id"] == c1["id"]

        loaded = store.get_chat(c1["id"])
        assert loaded is not None
        assert loaded["title"] == "What is the limitation period under Section 5?"
        assert len(loaded["messages"]) == 1
        assert loaded["messages"][0]["cited"][0]["key"] == "s:5"

        # Second chat updates later, so list_chats returns it first
        c2 = store.create_chat(title="RTI Inquiry")
        chats = store.list_chats()
        assert [c["id"] for c in chats] == [c2["id"], c1["id"]]

        # Add follow-up to c1 -> c1 bubbles back to top
        store.add_message(c1["id"], {"question": "What about exceptions to it?", "answer": "None.", "abstained": False})
        assert [c["id"] for c in store.list_chats()] == [c1["id"], c2["id"]]
        assert len(store.recent_turns(c1["id"])) == 2

        renamed = store.rename_chat(c1["id"], "  Limitation & Exceptions  ")
        assert renamed is not None and renamed["title"] == "Limitation & Exceptions"

        assert store.delete_chat(c1["id"]) is True
        assert store.get_chat(c1["id"]) is None
        assert store.delete_chat("missing") is False

        assert store.clear_chats() == 1
        assert store.list_chats() == []
    finally:
        store.close()


def test_rerank_resolve_device_auto(monkeypatch):
    monkeypatch.setattr(rerank.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        rerank.torch.cuda,
        "get_device_properties",
        lambda _idx: SimpleNamespace(total_memory=4 * 1024**3),
    )
    assert rerank.resolve_device("auto") == "cpu"

    monkeypatch.setattr(
        rerank.torch.cuda,
        "get_device_properties",
        lambda _idx: SimpleNamespace(total_memory=8 * 1024**3),
    )
    assert rerank.resolve_device("auto") == "cuda"
    assert rerank.resolve_device("cpu") == "cpu"


def test_needs_contextualize(monkeypatch):
    monkeypatch.setattr(S, "parse", lambda q: SimpleNamespace(acts=[{"title": "rti"}] if "rti" in q.lower() else []))
    assert S.needs_contextualize("What is the penalty?", []) is False
    assert S.needs_contextualize("What is the penalty for it?", [{"question": "q1", "answer": "a1"}]) is True
    assert S.needs_contextualize("Under the RTI Act, 2005, what is the fee?", [{"question": "q1", "answer": "a1"}]) is False


def test_http_api_chats_document_window_and_live_query(tmp_path, monkeypatch):
    # Create a > 200k char statute file to test document windowing beyond the old 200k truncation
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "text").mkdir(parents=True)
    (corpus_dir / "meta").mkdir(parents=True)
    doc_id = "long-act__handle-1"
    body = ("A" * 210_000) + "SECTION 99 TARGET TEXT" + ("B" * 5_000)
    (corpus_dir / "text" / f"{doc_id}.txt").write_text(body, encoding="utf-8")
    (corpus_dir / "meta" / f"{doc_id}.json").write_text(
        json.dumps({"title": "Long Act, 2020", "jurisdiction": "India", "document_type": "central_act", "year": 2020}),
        encoding="utf-8",
    )
    monkeypatch.setattr(serve_api, "CORPUS_DIR", corpus_dir)

    seen_calls: list[dict] = []

    class StubRuntime(serve_api.Runtime):
        def query(self, question, *, budget=8000, jurisdiction=None, doc_type=None, prior_turns=()):
            seen_calls.append({
                "question": question,
                "budget": budget,
                "jurisdiction": jurisdiction,
                "doc_type": doc_type,
                "prior_turns": list(prior_turns),
            })
            return {
                "mode": "live",
                "question": question,
                "effective_question": f"Resolved: {question}" if prior_turns else question,
                "jurisdiction": jurisdiction or "All",
                "doc_type": doc_type or "All",
                "budget": budget,
                "abstained": False,
                "reason": None,
                "answer": 'The law says "TARGET TEXT" here.',
                "cited_numbers": [1],
                "cited": [{"document_id": doc_id, "key": "s:99", "title": "Long Act, 2020", "label": "s. 99"}],
                "passages": [
                    {
                        "index": 1,
                        "document_id": doc_id,
                        "key": "s:99",
                        "title": "Long Act, 2020",
                        "label": "s. 99",
                        "char_start": 210_000,
                        "char_end": 210_022,
                        "text": "SECTION 99 TARGET TEXT",
                    }
                ],
                "trace": {"abstain_reason": None, "rerank_top": 0.91},
                "seconds": 0.42,
            }

    state = {"live": False}
    hist = ChatStore(tmp_path / "chats.sqlite")
    rt = StubRuntime(ROOT / "configs" / "C_full.yaml", rerank_device="cpu")
    srv = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        serve_api.make_handler(
            rt,
            hist,
            kb_check=lambda: state["live"],
            ollama_check=lambda: state["live"],
            dist_dir=tmp_path / "missing_dist",
        ),
    )
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"

    try:
        # 1. Health in demo mode + 503 on /api/query
        code, health = _req(f"{base}/api/health")
        assert code == 200 and health["mode"] == "demo"
        code, err = _req(f"{base}/api/query", method="POST", body={"question": "test"})
        assert code == 503 and err["error"] == "live_kb_unavailable"

        # 2. Document window past 200k chars
        code, doc = _req(f"{base}/api/document?id={doc_id}&start=210000&end=210022&pad=10")
        assert code == 200
        assert doc["window_start"] == 209_990
        assert doc["window_end"] == 210_032
        assert doc["truncated"] is True
        assert "SECTION 99 TARGET TEXT" in doc["text"]

        # 3. Live query creates chat & saves turn, then follow-up receives prior_turns
        state["live"] = True
        code, r1 = _req(
            f"{base}/api/query",
            method="POST",
            body={"question": "What does Section 99 say?", "jurisdiction": "Central", "budget": 2000},
        )
        assert code == 200 and r1["chat_id"] and r1["message_id"]
        assert seen_calls[-1]["prior_turns"] == []

        code, r2 = _req(
            f"{base}/api/query",
            method="POST",
            body={"question": "What about penalties?", "chat_id": r1["chat_id"]},
        )
        assert code == 200 and r2["chat_id"] == r1["chat_id"]
        assert len(seen_calls[-1]["prior_turns"]) == 1
        assert seen_calls[-1]["prior_turns"][0]["question"] == "What does Section 99 say?"

        # 4. Chat REST endpoints reflect both turns, rename, and delete
        code, chat = _req(f"{base}/api/chats/{r1['chat_id']}")
        assert code == 200 and len(chat["messages"]) == 2
        code, renamed = _req(f"{base}/api/chats/{r1['chat_id']}", method="PATCH", body={"title": "Sec 99"})
        assert code == 200 and renamed["title"] == "Sec 99"
        code, deleted = _req(f"{base}/api/chats/{r1['chat_id']}", method="DELETE")
        assert code == 200 and deleted["deleted"] is True
    finally:
        srv.shutdown()
        srv.server_close()
        hist.close()


def test_runtime_filter_docs_maps_ui_labels():
    import pandas as pd

    docs = pd.DataFrame(
        [
            {"document_id": "d_in", "jurisdiction": "India", "document_type": "central_act", "category": "central_acts"},
            {"document_id": "d_dl", "jurisdiction": "Delhi", "document_type": "state_act", "category": "state_acts"},
            {"document_id": "d_rule", "jurisdiction": "India", "document_type": "rule", "category": "rules"},
        ]
    ).set_index("document_id")

    rt = serve_api.Runtime(ROOT / "configs" / "C_full.yaml", rerank_device="cpu")
    rt._pipe = SimpleNamespace(store=SimpleNamespace(docs=docs))

    assert rt.filter_docs("All", "All") is None
    assert rt.filter_docs("Central", "All") == ["d_in", "d_rule"]
    assert rt.filter_docs("Central", "Central Act") == ["d_in"]
    assert rt.filter_docs("Delhi", "State Act") == ["d_dl"]
    assert rt.filter_docs("Delhi", "Central Act") == []


def test_pipeline_rank_allowed_docs_filtering():
    pytest.importorskip("pyarrow")
    import numpy as np
    from isr.pipeline import Pipeline

    class FakeBM25:
        unit_ids = np.array([0, 1, 2], dtype=np.int32)

        def row_of(self):
            return np.array([0, 1, 2], dtype=np.int64)

        def search(self, query, k=100, mask=None):
            ids = self.unit_ids if mask is None else self.unit_ids[mask]
            return ids[:k], np.ones(len(ids[:k]), dtype=np.float32)

    class FakeStore:
        doc = np.array(["docA", "docB", "docA"], dtype=object)
        class_id = np.array([0, 1, 2], dtype=np.int64)
        rep_of_class = np.array([0, 1, 2], dtype=np.int64)

        def reps_for_docs(self, dids):
            s = set(dids)
            return np.array([i for i, d in enumerate(self.doc) if d in s], dtype=np.int64)

    pipe = Pipeline.__new__(Pipeline)
    pipe.config = {"name": "test", "retrieve": {"bm25": {"k": 10, "weight": 1.0}}, "fusion": {"k": 60}}
    pipe.name = "test"
    pipe.store = FakeStore()
    pipe.bm25 = FakeBM25()
    pipe.dense = None
    pipe.reranker = None
    pipe.acts = None
    pipe.xref = None
    pipe.jur_docs = {}
    pipe._mask_cache = {}

    ranked, tr = pipe.rank("some query", allowed_docs=["docB"])
    assert ranked == [1]
    assert tr.abstain_reason is None

    empty_ranked, empty_tr = pipe.rank("some query", allowed_docs=[])
    assert empty_ranked == []
    assert empty_tr.abstain_reason == "filter_empty"
