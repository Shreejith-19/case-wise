"""Retrieval units for one document: provisions, sub-units of long provisions, gap windows.

Offsets are Python ``str`` indices (Unicode code points, half-open) into
``corpus/text/<document_id>.txt`` — the coordinate system the scorer uses.

* **provision** — every entry of ``structure.provisions`` (sections / articles). On the
  dev split every gold location is exactly one provision span.
* **subunit** — a provision longer than ``SUB_MAX`` is tiled, without gaps or overlap,
  into pieces of about ``SUB_TARGET`` chars along its children in the unit tree
  (subsections, clauses, provisos); a child that is itself too long is split
  recursively, and text without usable children is split at line breaks.
* **window** — text covered by no provision (preamble, schedules, forms, documents
  with no parsed structure) as ``WINDOW``/``STRIDE`` sliding windows, labelled with
  the region they fall in. Kept deliberately: dropping it would exploit the fact that
  gold is always a provision (see plan.md Phase 1).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Optional

SUB_MAX = 2000        # provisions longer than this get sub-units
SUB_TARGET = 1500     # aim for sub-units about this long
WINDOW = 1200         # gap-window length (same geometry as the shipped BM25 baseline)
STRIDE = 900
MIN_GAP_CHARS = 40    # gaps with fewer non-space chars than this are ignored

_TOKEN = re.compile(r"[0-9a-z]+")


@dataclass
class Unit:
    kind: str                       # provision | subunit | window
    char_start: int
    char_end: int
    key: Optional[str] = None       # provision key (sub-units carry their parent's)
    parent: Optional[int] = None    # index (within this document's list) of the parent provision
    ordinal: int = 0                # position among siblings (sub-units)
    label: str = ""                 # human-readable locator, e.g. "s. 2 Definitions"
    heading: str = ""
    region: str = ""                # window region: preamble | schedule | appendix | part | chapter | unparsed | other
    evidence_confidence: str = ""
    key_ambiguous: bool = False
    norm_hash: str = ""
    extra: dict = field(default_factory=dict)


def normalise(text: str) -> str:
    """NFKC, lowercase, alphanumeric tokens only — robust to footnote digits/whitespace noise."""
    return " ".join(_TOKEN.findall(unicodedata.normalize("NFKC", text).lower()))


def norm_hash(text: str) -> str:
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()[:16]


# --- labels ---------------------------------------------------------------------------

_ABBR = {"section": "s.", "article": "art.", "rule": "r.", "regulation": "reg.", "chapter": "Ch.",
         "part": "Pt.", "schedule": "Sch.", "subsection": "", "clause": "", "subclause": "",
         "proviso": "proviso", "explanation": "Expl.", "appendix": "App."}


def provision_label(p: dict) -> str:
    """'Ch. I, s. 2 Definitions' from a structure provision."""
    parts = []
    for step in p.get("unit_path") or []:
        abbr = _ABBR.get(step["unit_type"], step["unit_type"])
        parts.append(f"{abbr} {step['number']}".strip())
    head = (p.get("heading") or "").strip()
    return ", ".join(parts) + (f" {head}" if head else "")


def _child_label(u: dict) -> str:
    t, n = u["unit_type"], u.get("number")
    if t in ("subsection",):
        return f"({n})"
    if t in ("clause", "subclause"):
        return f"({n})"
    return f"{_ABBR.get(t, t)} {n or ''}".strip()


# --- sub-units ------------------------------------------------------------------------


def _split_at_lines(text: str, start: int, end: int, target: int = SUB_TARGET) -> list[tuple[int, int]]:
    """Tile [start, end) into ~target pieces, cutting at the last newline before the target."""
    pieces = []
    s = start
    while end - s > SUB_MAX:
        hard = s + target
        cut = text.rfind("\n", s + target // 2, hard)
        if cut == -1:
            cut = text.rfind(" ", s + target // 2, hard)
        cut = hard if cut == -1 else cut + 1
        pieces.append((s, cut))
        s = cut
    pieces.append((s, end))
    return pieces


def _tile(text: str, start: int, end: int, children: list[dict]) -> list[tuple[int, int, str]]:
    """Tile [start, end) along *children*; returns (start, end, label) pieces covering it exactly."""
    kids = sorted((c for c in children if start <= c["char_start"] < c["char_end"] <= end),
                  key=lambda c: c["char_start"])
    if not kids:
        return [(s, e, "") for s, e in _split_at_lines(text, start, end)]
    # Segments: [start, kid0.start) lead-in (merged into first piece), then each kid extended to next kid.
    bounds = [k["char_start"] for k in kids] + [end]
    segs: list[tuple[int, int, dict]] = []
    for i, k in enumerate(kids):
        seg_start = start if i == 0 else bounds[i]
        segs.append((seg_start, bounds[i + 1], k))
    out: list[tuple[int, int, str]] = []
    cur_s, cur_e, cur_labels = None, None, []
    for s, e, k in segs:
        if e - s > SUB_MAX:                       # oversized child: flush, then split it recursively
            if cur_s is not None:
                out.append((cur_s, cur_e, _span_label(cur_labels)))
                cur_s, cur_e, cur_labels = None, None, []
            for ps, pe, pl in _tile(text, s, e, k.get("children") or []):
                out.append((ps, pe, _child_label(k) + (pl and " " + pl)))
            continue
        if cur_s is None:
            cur_s, cur_e, cur_labels = s, e, [_child_label(k)]
        elif e - cur_s <= SUB_TARGET:
            cur_e = e
            cur_labels.append(_child_label(k))
        else:
            out.append((cur_s, cur_e, _span_label(cur_labels)))
            cur_s, cur_e, cur_labels = s, e, [_child_label(k)]
    if cur_s is not None:
        out.append((cur_s, cur_e, _span_label(cur_labels)))
    return out


def _span_label(labels: list[str]) -> str:
    labels = [x for x in labels if x]
    if not labels:
        return ""
    return labels[0] if len(labels) == 1 else f"{labels[0]}–{labels[-1]}"


def _index_units(units: Iterable[dict], by_key: dict) -> None:
    for u in units:
        if u.get("key"):
            by_key[u["key"]] = u
        _index_units(u.get("children") or [], by_key)


# --- windows over uncovered text ------------------------------------------------------


def _gaps(n: int, covered: list[tuple[int, int]]) -> list[tuple[int, int]]:
    gaps, pos = [], 0
    for s, e in sorted(covered):
        if s > pos:
            gaps.append((pos, s))
        pos = max(pos, e)
    if pos < n:
        gaps.append((pos, n))
    return gaps


def _windows(start: int, end: int) -> list[tuple[int, int]]:
    if end - start <= WINDOW:
        return [(start, end)]
    out, s = [], start
    while True:
        e = min(s + WINDOW, end)
        out.append((s, e))
        if e >= end:
            return out
        s += STRIDE


def _region(pos: int, structure: dict, has_structure: bool) -> str:
    if not has_structure:
        return "unparsed"
    pre = structure.get("preamble")
    if pre and pre["char_start"] <= pos < pre["char_end"]:
        return "preamble"
    for u in structure.get("units") or []:
        if u["char_start"] <= pos < u["char_end"]:
            return u["unit_type"]
    return "other"


# --- entry point ----------------------------------------------------------------------


def document_units(text: str, structure: dict) -> list[Unit]:
    """All retrieval units of one document, provisions first then sub-units then windows."""
    provisions = sorted(structure.get("provisions") or [], key=lambda p: p["char_start"])
    by_key: dict[str, dict] = {}
    _index_units(structure.get("units") or [], by_key)

    units: list[Unit] = []
    for p in provisions:
        s, e = p["char_start"], p["char_end"]
        units.append(Unit("provision", s, e, key=p["key"], label=provision_label(p),
                          heading=(p.get("heading") or "").strip(),
                          evidence_confidence=p.get("evidence_confidence", ""),
                          key_ambiguous=bool(p.get("key_ambiguous")),
                          norm_hash=norm_hash(text[s:e])))
    n_prov = len(units)
    for i in range(n_prov):
        pu = units[i]
        if pu.char_end - pu.char_start <= SUB_MAX:
            continue
        node = by_key.get(pu.key) or {}
        pieces = _tile(text, pu.char_start, pu.char_end, node.get("children") or [])
        for j, (s, e, lab) in enumerate(pieces):
            units.append(Unit("subunit", s, e, key=pu.key, parent=i, ordinal=j,
                              label=f"{pu.label} {lab}".strip() if lab else f"{pu.label} [part {j + 1}/{len(pieces)}]",
                              heading=pu.heading, evidence_confidence=pu.evidence_confidence,
                              key_ambiguous=pu.key_ambiguous, norm_hash=norm_hash(text[s:e])))

    has_structure = bool(provisions)
    for gs, ge in _gaps(len(text), [(p["char_start"], p["char_end"]) for p in provisions]):
        if sum(not c.isspace() for c in text[gs:ge]) < MIN_GAP_CHARS:
            continue
        # trim surrounding whitespace so windows start on text
        while gs < ge and text[gs].isspace():
            gs += 1
        while ge > gs and text[ge - 1].isspace():
            ge -= 1
        for j, (s, e) in enumerate(_windows(gs, ge)):
            region = _region(s, structure, has_structure)
            units.append(Unit("window", s, e, ordinal=j, region=region,
                              label=f"[{region}]", norm_hash=norm_hash(text[s:e])))
    return units
