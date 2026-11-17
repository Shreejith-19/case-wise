"""Plan B LLM stages (plan.md Phase 3). Every prompt carries a version string that is part of the cache key.

Stage 1 (before retrieval): ``analyse`` (B1) and ``hyde`` (B2).
Stage 3 (after retrieval):  ``relevance`` (B3 pointwise rerank), ``judge`` (B4 abstention), ``answer`` (B5).

Prompts were written once from the task definition, not iterated on dev questions; any revision must
bump the version and be logged in plan.md (dev_tune only).
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from isr.corpus.units import normalise
from isr.llm.client import LLM
from isr.query.rules import JURISDICTIONS, _match_acts, parse

SYSTEM = ("You are an expert on Indian statutory law (Central and State Acts, Rules and Regulations). "
          "Be precise and concise. Never invent provisions.")

# --- B1 query analysis --------------------------------------------------------------------------------
ANALYSE_V = "analyse-v1"
ANALYSE_SCHEMA = {
    "type": "object",
    "properties": {
        "named_laws": {"type": "array", "items": {"type": "string"}},
        "candidate_laws": {"type": "array", "items": {"type": "string"}},
        "jurisdiction": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["named_laws", "candidate_laws", "jurisdiction", "keywords"],
}
ANALYSE_PROMPT = """Analyse this question about Indian law.

Question: {q}

Return JSON with:
- "named_laws": laws the question names explicitly, copied as written (empty list if none).
- "candidate_laws": up to 3 Indian statutes (Acts, Rules or Regulations) whose provisions most likely answer it, \
each as its full short title with year, e.g. "The Negotiable Instruments Act, 1881". Include the named law if any. \
If the question concerns a particular State or Union Territory, prefer that State's own law when one exists.
- "jurisdiction": the Indian State or Union Territory the question is about, "Central" if it is about a Central law \
or India generally, or "" if unclear.
- "keywords": 3 to 8 words or short phrases likely to appear verbatim in the provision's text \
(statutory terms, not the question's wording)."""


def analyse(llm: LLM, question: str) -> dict:
    r = llm.chat([{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": ANALYSE_PROMPT.format(q=question)}],
                 version=ANALYSE_V, schema=ANALYSE_SCHEMA, max_tokens=300)
    out = r.parsed if isinstance(r.parsed, dict) else {}
    return {k: out.get(k, [] if k != "jurisdiction" else "") for k in ANALYSE_SCHEMA["required"]}


def resolve_laws(names: Sequence[str], allow_year_mismatch: bool = True) -> list[str]:
    """Corpus document ids for law names (title gazetteer of the rule parser; unresolved names dropped).

    v2 (dev_tune decision, plan.md Phase 3 log): the LLM's years are unreliable (e.g. "Rajasthan Housing Board
    Act, 1982" for the 1970 Act), so a title match with a year no copy has is kept (all copies of that title).
    """
    ids: list[str] = []
    for name in names[:4]:
        for a in _match_acts(normalise(str(name))):
            if allow_year_mismatch or a["how"] != "title(year-mismatch)":
                ids.extend(a["document_ids"])
    return list(dict.fromkeys(ids))


def canon_jurisdiction(text: str) -> Optional[str]:
    if not text or text.strip().lower() in ("central", "india", ""):
        return None
    js = parse(text).jurisdictions
    return js[0] if js else None


# --- B2 HyDE ------------------------------------------------------------------------------------------
HYDE_V = "hyde-v1"
HYDE_PROMPT = """Write the text of the Indian statutory provision (a section, rule or article) that would answer \
the question below, in the drafting style of Indian legislation. 60 to 150 words. Output only the provision text, \
starting with its number and heading; no commentary.

Question: {q}"""


def hyde(llm: LLM, question: str) -> str:
    r = llm.chat([{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": HYDE_PROMPT.format(q=question)}],
                 version=HYDE_V, max_tokens=300)
    return r.content.strip()


# --- B3 pointwise relevance ----------------------------------------------------------------------------
REL_V = "relevance-v1"
REL_PROMPT = """Question: {q}

Provision ({title}, {label}):
{text}

Does this provision itself contain the answer to the question? Answer only "yes" or "no"."""


def relevance(llm: LLM, question: str, title: str, label: str, text: str) -> float:
    r = llm.chat([{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": REL_PROMPT.format(q=question, title=title, label=label, text=text)}],
                 version=REL_V, max_tokens=1, logprobs=5)
    p = r.p_yes()
    if p is None:
        p = 1.0 if r.content.strip().lower().startswith("yes") else 0.0
    return p


# --- B4 abstention judge -------------------------------------------------------------------------------
JUDGE_V = "judge-v1"
JUDGE_PROMPT = """Question: {q}

Retrieved statutory text:
{passages}

Can the question be answered from the retrieved text above? Answer "no" if the question refers to a law, \
section or rule that does not appear above, or asks about something these provisions do not cover. \
Answer only "yes" or "no"."""


def judge(llm: LLM, question: str, passages: str) -> float:
    r = llm.chat([{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": JUDGE_PROMPT.format(q=question, passages=passages)}],
                 version=JUDGE_V, max_tokens=1, logprobs=5)
    p = r.p_yes()
    if p is None:
        p = 1.0 if r.content.strip().lower().startswith("yes") else 0.0
    return p


# --- B5 cited answer ----------------------------------------------------------------------------------
ANSWER_V = "answer-v1"
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"},
                   "citations": {"type": "array", "items": {"type": "integer"}},
                   "abstain": {"type": "boolean"}},
    "required": ["answer", "citations", "abstain"],
}
ANSWER_PROMPT = """Answer the question using ONLY the numbered statutory provisions below.

Question: {q}

Provisions:
{passages}

Rules:
- Answer in 1 to 3 sentences, using the provisions' own wording for the key terms, numbers and conditions.
- Quote the decisive phrase verbatim in double quotes (at least 12 characters, copied exactly from a provision).
- "citations": the numbers of every provision you relied on (usually 1 or 2).
- If the provisions do not answer the question, set "abstain" to true, "answer" to "" and "citations" to []."""


def answer(llm: LLM, question: str, passages: str) -> dict:
    r = llm.chat([{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": ANSWER_PROMPT.format(q=question, passages=passages)}],
                 version=ANSWER_V, schema=ANSWER_SCHEMA, max_tokens=400)
    out = r.parsed if isinstance(r.parsed, dict) else {}
    return {"answer": str(out.get("answer", "")), "citations": [c for c in out.get("citations", []) if isinstance(c, int)],
            "abstain": bool(out.get("abstain", False))}


_QUOTE = re.compile(r'"([^"\n]+)"|“([^”\n]+)”')


def ground_quotes(text: str, sources: Sequence[str]) -> str:
    """Keep double quotes only around strings that occur in the cited sources, under the official scorer's own
    normalisation (benchmark.terms.normalise: case/Unicode/whitespace-folded, punctuation kept).

    v2 (bug fix, plan.md Phase 3 log): every other double quote becomes a single quote, so the scorer's quote
    regex (>= QUOTE_MIN_CHARS, pairs across short quotes) sees exactly the pairs verified here; a final pass with
    the scorer's own ``quotes()`` drops all double quotes if any detected quote is still unverified.
    ``sources`` must be the texts of the *cited* provisions only (the scorer's haystack is gold + cited docs)."""
    from isr.eval.kit import kit_score
    from benchmark.terms import normalise as kit_normalise
    hay = " ".join(kit_normalise(s) for s in sources)

    def ok(s: str) -> bool:
        n = kit_normalise(s)
        return bool(n) and len(s) >= kit_score.QUOTE_MIN_CHARS and n in hay

    def fix(m: re.Match) -> str:
        inner = m.group(1) or m.group(2)
        return f'"{inner}"' if ok(inner) else f"'{inner}'"
    out = _QUOTE.sub(fix, text)
    # safety net: an unpaired leftover mark, or any scorer-detected quote still unverified -> no double quotes at all
    if out.count('"') % 2 or not all(ok(q) for q in kit_score.quotes(out)):
        out = re.sub('["“”]', "'", out)
    return out
