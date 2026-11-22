"""Interactive chatbot over the frozen final system (configs/C_full.yaml): retrieval + abstention rules +
cross-reference expansion + qwen3:8b cited answer. Needs Ollama running (`ollama serve` or the Ollama app).

    python scripts/chat.py                        # interactive
    python scripts/chat.py "your question"        # one question, then exit
    python scripts/chat.py --config configs/C_full.yaml

In the chat:  :sources  all provisions given to the LLM for the last answer
              :why      retrieval details for the last question (abstention reason, top hits, scores)
              :quit     exit

GPU sharing: the batch runner stages LLM and retrieval because both do not fit on an 8 GB GPU. Here the dense
search runs on the CPU (isr.index.dense_cpu), so the cross-encoder and the LLM can stay loaded together.
Retrieval logic, prompts and post-processing are those of the evaluated system (run_b.py, B-full/C-full).
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from isr import ROOT  # noqa: E402

WHY = {
    "missing_provision": "the question names an Act that is in the corpus, but no copy of that Act has the "
                         "section / rule number asked about",
    "unresolved_law": "the question names a law that does not match any title in the corpus",
    "low_rerank_score": "no provision in the corpus matches the question closely enough "
                        "(best cross-encoder score below the 0.041 threshold)",
    "llm": "the retrieved provisions were judged not to answer the question (the model declined)",
}
WIDTH = 100


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(" ".join(text.split()), WIDTH, initial_indent=indent, subsequent_indent=indent)


class Bot:
    def __init__(self, config: Path) -> None:
        from isr.index.dense_cpu import CpuDenseIndex
        from isr.llm.client import LLM
        from isr.pipeline import Pipeline
        self.cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        t0 = time.time()
        print("Loading indexes and models (about a minute) ...", flush=True)
        self.pipe = Pipeline(self.cfg, dense=CpuDenseIndex(self.cfg["retrieve"]["dense"]["model"]))
        an, lc = self.cfg["answer"], self.cfg.get("llm") or {}
        self.llm = LLM(an.get("model", lc.get("model", "qwen3:8b")), num_ctx=an.get("num_ctx", 8192))
        print(f"Ready in {time.time() - t0:.0f}s: {self.cfg['name']} "
              f"({len(self.pipe.store.docs):,} documents). Ask a question about Indian statutory law.\n")
        self.last: dict = {}

    def ask(self, question: str) -> dict:
        from isr.llm import stages as S
        from isr.pack import pack
        from run_b import fmt_passages, passages_for
        t0 = time.time()
        store, an = self.pipe.store, self.cfg["answer"]
        ranked, tr = self.pipe.rank(question)
        out = {"question": question, "trace": tr, "ranked": ranked}
        if tr.abstain_reason:
            out.update(abstained=True, reason=tr.abstain_reason)
        else:
            spans = pack(ranked, 8000, store, min_piece=self.cfg.get("pack", {}).get("min_piece", 200))
            ps = passages_for(store, spans, an.get("passages", 8), an.get("chars", 2500))
            o = S.answer(self.llm, question, fmt_passages(ps), prompt=an.get("prompt", "v1"),
                         think=an.get("think", False), max_tokens=an.get("max_tokens", 400))
            cited = [ps[i - 1] for i in o["citations"] if 1 <= i <= len(ps) and ps[i - 1]["key"]]
            text = S.ground_quotes(o["answer"], [p["text"] for p in cited])
            out.update(passages=ps, cited=cited, answer=text,
                       abstained=o["abstain"] or not text.strip(), reason="llm" if o["abstain"] else None)
        out["seconds"] = time.time() - t0
        self.last = out
        return out

    def show(self, r: dict) -> None:
        if r["abstained"]:
            print("\nNo answer (abstained).")
            print(wrap(f"Reason: {WHY.get(r['reason'], r['reason'])}."))
        else:
            print("\nAnswer:")
            print(wrap(r["answer"]))
            if r["cited"]:
                print("Cited provisions:")
                for p in {(p["document_id"], p["key"]): p for p in r["cited"]}.values():   # same provision once
                    print(wrap(f"- {p['title']} — {p['label']}  [{p['key']}]", "  "))
        print(f"({r['seconds']:.1f}s)\n")

    def sources(self) -> None:
        for i, p in enumerate(self.last.get("passages", []), 1):
            print(wrap(f"[{i}] {p['title']} — {p['label']}  [{p['key']}]", ""))
            print(wrap(p["text"][:300] + ("..." if len(p["text"]) > 300 else ""), "      "))
        if not self.last.get("passages"):
            print("  (no provisions: the last question was abstained before answering)")
        print()

    def why(self) -> None:
        tr, store = self.last.get("trace"), self.pipe.store
        if tr is None:
            print("  (ask a question first)\n")
            return
        p = tr.parsed
        print(f"  abstention: {tr.abstain_reason or 'none'}   best cross-encoder score: "
              f"{tr.rerank_top if tr.rerank_top is not None else 'n/a'}")
        if p is not None:
            print(f"  parsed: Acts={[a['title'] for a in p.acts][:3]} refs={p.refs} States={p.jurisdictions}")
        print(f"  lists: {', '.join(f'{k}({len(v)})' for k, v in tr.lists.items())}")
        for u in self.last.get("ranked", [])[:5]:
            sc = tr.ce.get(int(u))
            print(wrap(f"  {store.title(store.doc[u])} — {store.label[u]}"
                       f"{f'  (CE {sc:.3f})' if sc is not None else ''}", "  "))
        print()


def ollama_up() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=3):
            return True
    except Exception:
        return False


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "C_full.yaml")
    args = ap.parse_args()
    if not ollama_up():
        print("Ollama is not running. Start it with `ollama serve` (or open the Ollama app) and try again.")
        return 2
    bot = Bot(args.config)
    if args.question:
        bot.show(bot.ask(" ".join(args.question)))
        return 0
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        if q in (":quit", ":q", "quit", "exit"):
            return 0
        if q == ":sources":
            bot.sources()
        elif q == ":why":
            bot.why()
        else:
            bot.show(bot.ask(q))


if __name__ == "__main__":
    raise SystemExit(main())
