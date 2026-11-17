"""Local LLM client (Ollama HTTP API) with a persistent, content-addressed cache.

Determinism: temperature 0, fixed seed, explicit num_ctx, Qwen3 thinking off unless asked.
Cache key = (model digest, prompt version, messages, schema, options, logprobs): a changed prompt
or model can never be served a stale answer, and re-running an ablation is free.

The GPU cannot hold the LLM and the retrieval models at once (8 GB), so pipelines run LLM stages
in batches (warming this cache) and later stages read with ``cache_only=True``.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from isr import DATA_DIR

HOST = "http://localhost:11434"
CACHE = DATA_DIR / "cache" / "llm.sqlite"
SEED = 20261117


class CacheMiss(RuntimeError):
    pass


@dataclass
class Reply:
    content: str
    parsed: Any = None
    logprobs: Optional[list] = None
    prompt_tokens: int = 0
    eval_tokens: int = 0
    seconds: float = 0.0
    thinking: str = ""
    cached: bool = False

    def p_yes(self) -> Optional[float]:
        """P(first answer token is 'yes') renormalised over yes/no, from top logprobs of the first token."""
        if not self.logprobs:
            return None
        top = self.logprobs[0].get("top_logprobs") or []
        yes = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower().startswith("yes"))
        no = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower().startswith("no"))
        return yes / (yes + no) if yes + no > 0 else None


def _post(path: str, body: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(HOST + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def model_digest(model: str) -> str:
    with urllib.request.urlopen(HOST + "/api/tags", timeout=30) as r:
        for m in json.loads(r.read())["models"]:
            if m["name"] == model:
                return m["digest"]
    raise RuntimeError(f"model {model!r} is not pulled in Ollama")


class LLM:
    def __init__(self, model: str = "qwen3:8b", *, num_ctx: int = 8192, cache_only: bool = False,
                 cache_path: Path = CACHE, digest: Optional[str] = None) -> None:
        self.model, self.num_ctx, self.cache_only = model, num_ctx, cache_only
        self.digest = digest or (None if cache_only else model_digest(model))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(cache_path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, model TEXT, version TEXT, "
                         "reply TEXT, created REAL)")
        self._db.execute("CREATE TABLE IF NOT EXISTS digests (model TEXT PRIMARY KEY, digest TEXT)")
        if self.digest:
            self._db.execute("INSERT OR REPLACE INTO digests VALUES (?, ?)", (model, self.digest))
            self._db.commit()
        else:
            row = self._db.execute("SELECT digest FROM digests WHERE model=?", (model,)).fetchone()
            if row is None:
                raise CacheMiss(f"no cached digest for {model}; run the LLM stage first")
            self.digest = row[0]
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "cached": 0, "prompt_tokens": 0, "eval_tokens": 0, "seconds": 0.0}

    def _key(self, payload: dict, version: str) -> str:
        blob = json.dumps({"digest": self.digest, "version": version, **payload}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def chat(self, messages: list[dict], *, version: str, schema: Optional[dict] = None,
             max_tokens: int = 512, think: bool = False, logprobs: int = 0, retries: int = 2) -> Reply:
        options = {"temperature": 0, "seed": SEED, "num_ctx": self.num_ctx, "num_predict": max_tokens}
        payload = {"model": self.model, "messages": messages, "schema": schema, "options": options,
                   "think": think, "logprobs": logprobs}
        key = self._key(payload, version)
        with self._lock:
            row = self._db.execute("SELECT reply FROM replies WHERE key=?", (key,)).fetchone()
        self.stats["calls"] += 1
        if row is not None:
            self.stats["cached"] += 1
            return Reply(**{**json.loads(row[0]), "cached": True})
        if self.cache_only:
            raise CacheMiss(f"{version}: not cached (run the LLM stage first)")
        body = {"model": self.model, "messages": messages, "stream": False, "think": think,
                "options": dict(options), "keep_alive": "10m"}
        if schema is not None:
            body["format"] = schema
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = logprobs
        reply = None
        for attempt in range(retries + 1):
            t0 = time.time()
            out = _post("/api/chat", body)
            content = out["message"]["content"]
            reply = Reply(content=content, logprobs=out.get("logprobs"), thinking=out["message"].get("thinking") or "",
                          prompt_tokens=out.get("prompt_eval_count", 0), eval_tokens=out.get("eval_count", 0),
                          seconds=time.time() - t0)
            if schema is None:
                break
            try:
                reply.parsed = json.loads(content)
                break
            except json.JSONDecodeError:
                body["options"]["seed"] = SEED + attempt + 1        # deterministic retry with a new seed
        self.stats["prompt_tokens"] += reply.prompt_tokens
        self.stats["eval_tokens"] += reply.eval_tokens
        self.stats["seconds"] += reply.seconds
        stored = {k: v for k, v in reply.__dict__.items() if k != "cached"}
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO replies VALUES (?, ?, ?, ?, ?)",
                             (key, self.model, version, json.dumps(stored, ensure_ascii=False), time.time()))
            self._db.commit()
        return reply

    def unload(self) -> None:
        """Free VRAM so the retrieval models can use the GPU."""
        if not self.cache_only:
            try:
                _post("/api/generate", {"model": self.model, "keep_alive": 0}, timeout=60)
            except Exception:
                pass
