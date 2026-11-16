"""Phase 0.4: verify each local LLM runs fully on GPU with an explicit context, and measure speed.

    python scripts/ollama_check.py qwen3:8b llama3.1:8b
    -> runs/env/ollama_check.json

Reports, per model: digest, quantisation, share of weights in VRAM, prompt-processing and
generation tokens/s for a ~3k-token prompt at num_ctx=8192, and whether the prompt was
truncated (prompt_eval_count far below the expected token count = silent truncation).
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

from isr import CORPUS_DIR, RUNS_DIR

HOST = "http://localhost:11434"
NUM_CTX = 8192


def _post(path: str, body: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(HOST + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(path: str) -> dict:
    with urllib.request.urlopen(HOST + path, timeout=30) as r:
        return json.loads(r.read())


def _long_prompt() -> str:
    """~3k tokens of real statute text (a central Act's opening), plus a short task."""
    text = (CORPUS_DIR / "text" /
            "prevention-and-control-of-infectious-and-contagious-diseases__handle-21045.txt"
            ).read_text(encoding="utf-8")[4800:16800]
    return ("Read the statute extract and list, as JSON {\"defined_terms\": [...]}, every term "
            "defined in section 2. Output JSON only.\n\n" + text)


def check(model: str) -> dict:
    tags = {m["name"]: m for m in _get("/api/tags")["models"]}
    info = tags.get(model) or tags.get(model + ":latest")
    if info is None:
        return {"model": model, "error": "not pulled"}
    opts = {"temperature": 0, "seed": 0, "num_ctx": NUM_CTX, "num_predict": 200}
    _post("/api/generate", {"model": model, "prompt": "Say OK.", "stream": False,
                            "options": opts, "think": False})            # warm-up / load
    ps = {m["name"]: m for m in _get("/api/ps")["models"]}.get(info["name"], {})
    prompt = _long_prompt()
    t0 = time.time()
    r = _post("/api/generate", {"model": model, "prompt": prompt, "stream": False,
                                "options": opts, "think": False})
    wall = time.time() - t0
    approx_tokens = len(prompt) / 3.6       # rough chars/token for English legal text
    return {
        "model": info["name"],
        "digest": info["digest"],
        "size_gb": round(info["size"] / 1e9, 2),
        "quantization": info["details"].get("quantization_level"),
        "parameter_size": info["details"].get("parameter_size"),
        "vram_share": round(ps.get("size_vram", 0) / ps["size"], 3) if ps.get("size") else None,
        "num_ctx": NUM_CTX,
        "prompt_tokens": r.get("prompt_eval_count"),
        "prompt_tokens_expected_approx": int(approx_tokens),
        "truncated_suspect": bool(r.get("prompt_eval_count", 0) < 0.6 * approx_tokens),
        "prompt_tok_per_s": round(r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9), 1),
        "gen_tokens": r.get("eval_count"),
        "gen_tok_per_s": round(r["eval_count"] / (r["eval_duration"] / 1e9), 1),
        "wall_s": round(wall, 1),
        "sample_output": r.get("response", "")[:300],
    }


def main() -> int:
    models = sys.argv[1:] or ["qwen3:8b", "llama3.1:8b"]
    version = _get("/api/version")["version"]
    results = {"ollama_version": version, "models": [check(m) for m in models]}
    out = RUNS_DIR / "env" / "ollama_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
