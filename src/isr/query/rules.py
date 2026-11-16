"""Rule-based query parsing (no LLM): jurisdictions, provision references, named Acts.

Named Acts are found by matching the corpus's own normalised titles as contiguous token
sequences inside the normalised question (titles contain parentheses and commas that make
phrase-extraction regexes brittle). A year written next to the title filters by year.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from isr.corpus.build import DOCS_PATH
from isr.corpus.units import normalise

# canonical jurisdiction (as in docs.jurisdiction) -> extra aliases (lowercase, normalised)
JURISDICTIONS = {
    "Andaman and Nicobar Islands": ["andaman", "nicobar"], "Andhra Pradesh": [], "Arunachal Pradesh": [],
    "Assam": [], "Bihar": [], "Chandigarh": [], "Chhattisgarh": ["chattisgarh"],
    "Dadra and Nagar Haveli and Daman and Diu": ["dadra", "nagar haveli", "daman", "diu"],
    "Delhi": ["nct of delhi", "national capital territory"], "Goa": [], "Gujarat": [], "Haryana": [],
    "Himachal Pradesh": [], "Jammu and Kashmir": ["j k", "jammu", "kashmir"], "Jharkhand": [],
    "Karnataka": ["mysore state"], "Kerala": [], "Ladakh": [], "Madhya Pradesh": [],
    "Maharashtra": ["bombay state"], "Manipur": [], "Meghalaya": [], "Mizoram": [], "Nagaland": [],
    "Odisha": ["orissa"], "Puducherry": ["pondicherry"], "Punjab": ["east punjab"], "Rajasthan": [],
    "Sikkim": [], "Tamil Nadu": ["madras state"], "Telangana": [], "Tripura": [],
    "Uttar Pradesh": ["united provinces"], "Uttarakhand": ["uttaranchal"], "West Bengal": ["bengal"],
}
CENTRAL_HINTS = ["central government", "parliament", "union of india", "central act"]
_REF = re.compile(r"\b(section|sec|article|art|rule|regulation|reg|schedule|clause|paragraph)\.?\s+"
                  r"(\d+[a-z]{0,2}|[ivxlc]+)\b", re.I)
_REF_TYPE = {"sec": "section", "art": "article", "reg": "regulation"}
_DOC_TYPE_HINT = re.compile(r"\b(rules|regulations|ordinance|order|notification|scheme|bye-?laws)\b", re.I)
_NAMED_LAW = re.compile(r"\b(act|code|rules|regulations|ordinance|adhiniyam|sanhita)\b(?:,?\s*(?:of\s+)?(\d{4}))?",
                        re.I)
_YEAR_AFTER = re.compile(r"^\s*(?:of\s+)?(\d{4})\b")
_YEAR_AFTER_COMPACT = re.compile(r"^(?:of)?((?:1[6-9]|20)\d{2})")
_GENERIC = {"act", "rules", "regulations", "code", "order", "the act", "amendment act", "principal act"}


@dataclass
class ParsedQuery:
    text: str
    jurisdictions: list[str] = field(default_factory=list)
    central_hint: bool = False
    refs: list[tuple[str, str]] = field(default_factory=list)          # ("section", "32")
    doc_type_hint: str | None = None
    acts: list[dict] = field(default_factory=list)                      # {title, year, document_ids, how}
    names_a_law: bool = False                                           # "... Act[, YYYY]" phrase present
    named_law_unresolved: bool = False                                  # such a phrase, but no corpus match

    @property
    def act_document_ids(self) -> list[str]:
        return [d for a in self.acts for d in a["document_ids"]]


def _strip_title(title: str) -> tuple[str, int | None]:
    n = normalise(title)
    toks = n.split()
    year = None
    if toks and re.fullmatch(r"(1[6-9]|20)\d{2}", toks[-1]):
        year = int(toks[-1])
        toks = toks[:-1]
    while toks and toks[0] in ("the", "a"):
        toks = toks[1:]
    # leading enumeration noise from India Code titles ("1 ", "10 2017 ")
    while toks and toks[0].isdigit():
        toks = toks[1:]
    return " ".join(toks), year


@lru_cache(maxsize=1)
def title_index() -> dict[str, list[tuple[str, int | None]]]:
    """normalised title (no year) -> [(document_id, year)]."""
    docs = pd.read_parquet(DOCS_PATH, columns=["document_id", "title", "short_title", "year"])
    idx: dict[str, list] = {}
    for did, title, short, year in zip(docs["document_id"], docs["title"], docs["short_title"], docs["year"]):
        y = None if pd.isna(year) else int(year)
        for t in {title, short}:
            if not t:
                continue
            nt, ty = _strip_title(t)
            if len(nt.split()) < 2 or nt in _GENERIC:
                continue
            idx.setdefault(nt, []).append((did, ty or y))
    return {k: sorted(set(v)) for k, v in idx.items()}


def _match_acts(q_norm: str) -> list[dict]:
    padded = f" {q_norm} "
    compact = q_norm.replace(" ", "")
    hits = []
    for nt, docs in title_index().items():
        pos = padded.find(f" {nt} ")
        if pos >= 0:
            after = padded[pos + len(nt) + 2:]
        else:
            # space-insensitive fallback ("PrasarBharati" vs "Prasar Bharati"); long titles only
            if nt.count(" ") < 2:
                continue
            ct = nt.replace(" ", "")
            cpos = compact.find(ct)
            if cpos < 0:
                continue
            pos = cpos  # approximate position (compact coordinates), only used for overlap pruning
            ym = _YEAR_AFTER_COMPACT.match(compact[cpos + len(ct):])
            after = f" {ym.group(1)} " if ym else ""
        m = _YEAR_AFTER.match(after)
        year = int(m.group(1)) if m else None
        hits.append((nt, pos, year, docs))
    # keep maximal matches: drop a title contained in a longer matched title at an overlapping position
    hits.sort(key=lambda h: -len(h[0]))
    kept: list[tuple] = []
    for nt, pos, year, docs in hits:
        if any(pos >= kp and pos + len(nt) <= kp + len(kt) for kt, kp, _, _ in kept):
            continue
        kept.append((nt, pos, year, docs))
    out = []
    for nt, pos, year, docs in kept:
        ids = [d for d, y in docs if year is None or y is None or y == year]
        how = "title+year" if year is not None else "title"
        if not ids:              # title matched but no copy has that year
            ids, how = [d for d, _ in docs], "title(year-mismatch)"
        out.append({"title": nt, "year": year, "document_ids": ids, "how": how})
    return out


def parse(question: str) -> ParsedQuery:
    q_norm = normalise(question)
    padded = f" {q_norm} "
    pq_ = ParsedQuery(text=question)
    for canon, aliases in JURISDICTIONS.items():
        names = [normalise(canon)] + aliases
        if any(f" {a} " in padded for a in names):
            pq_.jurisdictions.append(canon)
    pq_.central_hint = any(f" {h} " in padded for h in CENTRAL_HINTS)
    for m in _REF.finditer(question):
        kind = m.group(1).lower()
        pq_.refs.append((_REF_TYPE.get(kind, kind), m.group(2).upper() if m.group(2).isalpha() else m.group(2)))
    m = _DOC_TYPE_HINT.search(question)
    pq_.doc_type_hint = m.group(1).lower() if m else None
    pq_.acts = _match_acts(q_norm)
    pq_.names_a_law = bool(re.search(r"\b[A-Z][\w'()-]*(?:\s+[\w'()&,-]+){0,12}\s+"
                                     r"(?:Act|Code|Rules|Regulations|Ordinance|Adhiniyam|Sanhita)\b", question))
    pq_.named_law_unresolved = pq_.names_a_law and not pq_.acts
    return pq_
