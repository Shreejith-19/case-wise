"""Persistent SQLite store for interactive chat threads and Q&A turns."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from isr import DATA_DIR

DEFAULT_HISTORY_DB = DATA_DIR / "chat_history.sqlite"


def _make_title(question: str, limit: int = 72) -> str:
    clean = " ".join((question or "").split())
    if not clean:
        return "New Research"
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


class ChatStore:
    """Thread-safe SQLite repository for multi-turn legal research conversations."""

    def __init__(self, db_path: Path = DEFAULT_HISTORY_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    jurisdiction TEXT NOT NULL DEFAULT 'All',
                    doc_type TEXT NOT NULL DEFAULT 'All',
                    budget INTEGER NOT NULL DEFAULT 8000,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    created_at REAL NOT NULL,
                    question TEXT NOT NULL,
                    effective_question TEXT NOT NULL DEFAULT '',
                    mode TEXT NOT NULL DEFAULT 'live',
                    jurisdiction TEXT NOT NULL DEFAULT 'All',
                    doc_type TEXT NOT NULL DEFAULT 'All',
                    budget INTEGER NOT NULL DEFAULT 8000,
                    abstained INTEGER NOT NULL DEFAULT 0,
                    reason TEXT,
                    answer TEXT NOT NULL DEFAULT '',
                    cited_numbers_json TEXT NOT NULL DEFAULT '[]',
                    cited_json TEXT NOT NULL DEFAULT '[]',
                    passages_json TEXT NOT NULL DEFAULT '[]',
                    trace_json TEXT NOT NULL DEFAULT '{}',
                    seconds REAL NOT NULL DEFAULT 0.0
                );
                CREATE INDEX IF NOT EXISTS idx_messages_chat_time
                    ON messages(chat_id, created_at ASC);
                CREATE INDEX IF NOT EXISTS idx_chats_updated
                    ON chats(updated_at DESC);
                """
            )
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def create_chat(
        self,
        *,
        title: Optional[str] = None,
        jurisdiction: str = "All",
        doc_type: str = "All",
        budget: int = 8000,
        chat_id: Optional[str] = None,
    ) -> dict[str, Any]:
        cid = chat_id or uuid.uuid4().hex[:12]
        now = time.time()
        clean_title = (title or "").strip() or "New Research"
        with self._lock:
            self._db.execute(
                "INSERT INTO chats (id, title, jurisdiction, doc_type, budget, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (cid, clean_title, jurisdiction or "All", doc_type or "All", int(budget), now, now),
            )
            self._db.commit()
        return {
            "id": cid,
            "title": clean_title,
            "jurisdiction": jurisdiction or "All",
            "doc_type": doc_type or "All",
            "budget": int(budget),
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
            "last_question": "",
            "messages": [],
        }

    def list_chats(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT c.id, c.title, c.jurisdiction, c.doc_type, c.budget, c.created_at, c.updated_at,
                       COUNT(m.id) AS message_count,
                       COALESCE(
                           (SELECT m2.question FROM messages m2
                            WHERE m2.chat_id = c.id
                            ORDER BY m2.created_at DESC LIMIT 1),
                           ''
                       ) AS last_question
                FROM chats c
                LEFT JOIN messages m ON m.chat_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_chat(self, chat_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            chat = self._db.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if chat is None:
                return None
            msgs = self._db.execute(
                "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC", (chat_id,)
            ).fetchall()
        out = dict(chat)
        out["messages"] = [self._row_to_message(m) for m in msgs]
        out["message_count"] = len(out["messages"])
        out["last_question"] = out["messages"][-1]["question"] if out["messages"] else ""
        return out

    def rename_chat(self, chat_id: str, title: str) -> Optional[dict[str, Any]]:
        clean = " ".join((title or "").split())
        if not clean:
            return None
        now = time.time()
        with self._lock:
            cur = self._db.execute(
                "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                (clean[:120], now, chat_id),
            )
            self._db.commit()
            if cur.rowcount == 0:
                return None
        return self.get_chat(chat_id)

    def delete_chat(self, chat_id: str) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            self._db.commit()
            return cur.rowcount > 0

    def clear_chats(self) -> int:
        with self._lock:
            cur = self._db.execute("DELETE FROM chats")
            self._db.commit()
            return int(cur.rowcount)

    def recent_turns(self, chat_id: str, limit: int = 4) -> list[dict[str, str]]:
        """Lightweight (question, answer) pairs for multi-turn query contextualization."""
        with self._lock:
            rows = self._db.execute(
                "SELECT question, answer, abstained FROM messages WHERE chat_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (chat_id, int(limit)),
            ).fetchall()
        return [
            {"question": r["question"], "answer": r["answer"] if not r["abstained"] else ""}
            for r in reversed(rows)
        ]

    def add_message(self, chat_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        mid = uuid.uuid4().hex[:12]
        now = time.time()
        question = str(payload.get("question") or "")
        jurisdiction = str(payload.get("jurisdiction") or "All")
        doc_type = str(payload.get("doc_type") or "All")
        budget = int(payload.get("budget") or 8000)

        with self._lock:
            row = self._db.execute("SELECT title FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown chat_id {chat_id!r}")
            new_title = _make_title(question) if row["title"] == "New Research" else row["title"]
            self._db.execute(
                """
                INSERT INTO messages (
                    id, chat_id, created_at, question, effective_question, mode,
                    jurisdiction, doc_type, budget, abstained, reason, answer,
                    cited_numbers_json, cited_json, passages_json, trace_json, seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mid,
                    chat_id,
                    now,
                    question,
                    str(payload.get("effective_question") or question),
                    str(payload.get("mode") or "live"),
                    jurisdiction,
                    doc_type,
                    budget,
                    int(bool(payload.get("abstained"))),
                    payload.get("reason"),
                    str(payload.get("answer") or ""),
                    json.dumps(payload.get("cited_numbers") or [], ensure_ascii=False),
                    json.dumps(payload.get("cited") or [], ensure_ascii=False),
                    json.dumps(payload.get("passages") or [], ensure_ascii=False),
                    json.dumps(payload.get("trace") or {}, ensure_ascii=False),
                    float(payload.get("seconds") or 0.0),
                ),
            )
            self._db.execute(
                "UPDATE chats SET title = ?, jurisdiction = ?, doc_type = ?, budget = ?, updated_at = ? WHERE id = ?",
                (new_title, jurisdiction, doc_type, budget, now, chat_id),
            )
            self._db.commit()

        return {
            "id": mid,
            "chat_id": chat_id,
            "created_at": now,
            "question": question,
            "effective_question": str(payload.get("effective_question") or question),
            "mode": str(payload.get("mode") or "live"),
            "jurisdiction": jurisdiction,
            "doc_type": doc_type,
            "budget": budget,
            "abstained": bool(payload.get("abstained")),
            "reason": payload.get("reason"),
            "answer": str(payload.get("answer") or ""),
            "cited_numbers": list(payload.get("cited_numbers") or []),
            "cited": list(payload.get("cited") or []),
            "passages": list(payload.get("passages") or []),
            "trace": dict(payload.get("trace") or {}),
            "seconds": float(payload.get("seconds") or 0.0),
        }

    @staticmethod
    def _row_to_message(r: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": r["id"],
            "chat_id": r["chat_id"],
            "created_at": r["created_at"],
            "question": r["question"],
            "effective_question": r["effective_question"],
            "mode": r["mode"],
            "jurisdiction": r["jurisdiction"],
            "doc_type": r["doc_type"],
            "budget": r["budget"],
            "abstained": bool(r["abstained"]),
            "reason": r["reason"],
            "answer": r["answer"],
            "cited_numbers": json.loads(r["cited_numbers_json"]),
            "cited": json.loads(r["cited_json"]),
            "passages": json.loads(r["passages_json"]),
            "trace": json.loads(r["trace_json"]),
            "seconds": r["seconds"],
        }
