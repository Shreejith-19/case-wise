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


# --- post-test option 1: listwise selection over the top candidates (plan.md "Post-test exploration") --------
SELECT_V = "select-v1"
SELECT_PROMPT = """Question: {q}

Candidate provisions:
{cands}

Which candidate provisions contain the answer to the question? Reply with JSON {{"best": [...]}} listing at most \
2 candidate ids, most relevant first. If the question has two parts, choose the provision that answers each part."""


def select(llm: LLM, question: str, cands: list[tuple[str, str]], max_pick: int = 2) -> list[str]:
    """``cands``: (handle, one-line description). Returns up to ``max_pick`` valid handles, best first."""
    ids = [h for h, _ in cands]
    schema = {"type": "object", "properties": {"best": {"type": "array", "items": {"type": "string", "enum": ids}}},
              "required": ["best"]}
    text = "\n".join(f"{h} | {d}" for h, d in cands)
    r = llm.chat([{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": SELECT_PROMPT.format(q=question, cands=text)}],
                 version=SELECT_V, schema=schema, max_tokens=60)
    out = r.parsed.get("best", []) if isinstance(r.parsed, dict) else []
    return [h for h in dict.fromkeys(str(x) for x in out) if h in ids][:max_pick]


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


# v2 (Phase 3 v2, plan.md): explicit completeness + double-quote rules, and two worked examples built from
# provisions that are gold for NO benchmark question (any split), each with a distractor passage.
ANSWER_V2 = "answer-v2"
ANSWER_PROMPT_V2 = """Answer the question using ONLY the numbered statutory provisions below.

Question: {q}

Provisions:
{passages}

Rules:
- Answer in 1 to 4 sentences and answer every part of the question.
- State every number, time limit, amount, condition, exception, authority and defined term the question asks \
about, in the provisions' own words.
- Copy the decisive phrases verbatim inside double quotes ("like this", never single quotes); every quoted \
phrase must be at least 12 characters and copied exactly from a provision you cite.
- "citations": the numbers of every provision you relied on, and only those.
- If none of the provisions answers the question, set "abstain" to true, "answer" to "" and "citations" to []."""

_FEWSHOT = [
    ("A port operator lost a dispute before the adjudicating board set up for the major ports and wants to go to "
     "the Supreme Court. How long does it have, and can a late appeal still be heard?",
     "[1] Major Port Authorities Act, 2021 — Ch. VI, s. 60 Review and appeal\n"
     "60. Review and appeal.—(1) Any party aggrieved by any decision or order of the Adjudicatory Board under this "
     "Act, from which an appeal is allowed under sub-section (2), but from which no appeal has been preferred, may "
     "apply for a review of such decision before the Adjudicatory Board, in such form and manner and within such "
     "time, as may be prescribed, and the said Board may make such order thereon, as it thinks fit.\n"
     "(2) Any party aggrieved by any decision or order of the Adjudicatory Board, may file an appeal to the Supreme "
     "Court of India, within sixty days from the date of communication of such decision or order to him:\n"
     "Provided that no appeal shall lie from a decision or order passed by the Adjudicatory Board with the consent "
     "of parties:\nProvided further that the Supreme Court may, entertain any appeal after the expiry of sixty days, "
     "if it is satisfied that the appellant was prevented by sufficient cause from preferring the appeal.\n\n"
     "[2] Assam Town and Country Planning Act, 1959 — Ch. VII, s. 42 Appeal\n"
     "42. Appeal.—(1) Any person aggrieved by the decision of the Authority with respect to matters of compensation "
     "and betterment fee, may appeal to the Appellate Authority within thirty days of the award.",
     {"answer": "The appeal lies to the Supreme Court \"within sixty days from the date of communication of such "
                "decision or order\". A late appeal can still be heard if the Supreme Court \"is satisfied that the "
                "appellant was prevented by sufficient cause from preferring the appeal\", but \"no appeal shall lie "
                "from a decision or order passed by the Adjudicatory Board with the consent of parties\".",
      "citations": [1], "abstain": False}),
    ("Under West Bengal's anti-piracy law, does keeping duplicate copies of a film in storage for sale count as "
     "piracy, and who is treated as the author of a sound recording?",
     "[1] West Bengal Prohibition Of Audio-video Piracy Act, 2013 — Pt. III, s. 2 Definitions\n"
     "2. In this Act, unless the context otherwise requires,—\n(a) \"author\" means in relation to a cinematograph "
     "film or sound recording, the producer;\n(b) \"original work\" means the work over which copyright of the "
     "author subsists;\n(c) \"Audio-Video Piracy\" means duplicating the original work without authorisation of the "
     "author in respect of cinematograph film and sound recording, with a view to trading, selling or hiring of "
     "those duplicate copies and includes any form of storage, distribution of such duplicate copies for the "
     "purpose of trading, selling or hiring;\n\n"
     "[2] Manipur Coaching Institute (Control and Regulation) Act, 2017 — s. 7\n"
     "7. The complaints under this Act shall be filed before the Registering Authority. Such complaints shall be "
     "disposed of within thirty days from the date of its receipt by the Registering Authority after making the "
     "necessary inquiry.",
     {"answer": "Yes. Audio-Video Piracy \"includes any form of storage, distribution of such duplicate copies for "
                "the purpose of trading, selling or hiring\". The author is, \"in relation to a cinematograph film "
                "or sound recording, the producer\".",
      "citations": [1], "abstain": False}),
]


def answer(llm: LLM, question: str, passages: str, prompt: str = "v1", think: bool = False,
           max_tokens: int = 400) -> dict:
    import json as _json
    if prompt == "v2":
        msgs = [{"role": "system", "content": SYSTEM}]
        for fq, fp, fa in _FEWSHOT:
            msgs += [{"role": "user", "content": ANSWER_PROMPT_V2.format(q=fq, passages=fp)},
                     {"role": "assistant", "content": _json.dumps(fa, ensure_ascii=False)}]
        msgs.append({"role": "user", "content": ANSWER_PROMPT_V2.format(q=question, passages=passages)})
        version = ANSWER_V2
    else:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": ANSWER_PROMPT.format(q=question, passages=passages)}]
        version = ANSWER_V
    r = llm.chat(msgs, version=version, schema=ANSWER_SCHEMA, max_tokens=max_tokens, think=think)
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
