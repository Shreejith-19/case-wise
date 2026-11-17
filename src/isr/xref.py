"""Plan C (plan.md Phase 4.3): cross-reference graph between provisions.

    python -m isr.xref build          (resumable; pause with a PAUSE file, see isr.pause)

For every provision, references written in its text -- "section 7", "sections 4 and 5", "sections 10 to 12",
"rule 3", "regulation 9", "article 21", "section 8 of the Act" -- are resolved to provision units:
  * by default in the same document (a statute citing its own sections);
  * "... of the Act" / "... of the said Act" / a bare "section N" inside Rules/Regulations -> the parent Act
    (``parent_document_id``), since Rules number their own provisions as rules;
  * "... of the <Other Title> Act/Code" -> skipped (cross-document title resolution: not built; see plan.md).
Self-references and references to the provision's own number are dropped. Edges are stored between class
representatives (dedup-consistent with the indexes): indexes/xref/edges.npy, int64 (src, dst) rows.
"""

from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np

from isr import INDEX_DIR, pause

XREF_DIR = INDEX_DIR / "xref"
CHUNK_DOCS = 1000
RANGE_MAX = 10

_REF = re.compile(
    r"\b(sections?|rules?|regulations?|articles?)\s+"
    r"((?:\d{1,4}[A-Z]{0,3}(?:\s*\([0-9a-zA-Z]{1,4}\))*"
    r"(?:\s*,\s*|\s+and\s+|\s+or\s+|\s+to\s+|\s*-\s*)?)+)", re.I)
_NUM = re.compile(r"(?<![\(\w])(\d{1,4}[A-Z]{0,3})(?![\w\)])")
_RANGE = re.compile(r"(\d{1,4})\s*(?:to|-)\s*(\d{1,4})(?![\w\)])")
_AFTER_EXTERNAL = re.compile(r"\s*(?:\([0-9a-zA-Z]{1,4}\)\s*)*of\s+the\s+(?!said\s+act|act\b|principal\s+act)"
                             r"[A-Z][^.;:]{0,80}?\b(Act|Code|Rules|Regulations|Ordinance|Constitution)\b")
_AFTER_PARENT = re.compile(r"\s*(?:\([0-9a-zA-Z]{1,4}\)\s*)*of\s+the\s+(said\s+)?act\b", re.I)


def extract(text: str) -> list[tuple[str, str, str]]:
    """(kind, number, target) for each reference; target is 'self', 'parent' or 'external'."""
    out = []
    for m in _REF.finditer(text):
        kind = m.group(1).lower().rstrip("s")
        body = m.group(2)
        nums = set(_NUM.findall(body))
        for a, b in _RANGE.findall(body):
            a, b = int(a), int(b)
            if 0 < b - a <= RANGE_MAX:
                nums |= {str(i) for i in range(a, b + 1)}
        tail = text[m.end():m.end() + 120]
        if _AFTER_EXTERNAL.match(tail):
            target = "external"
        elif _AFTER_PARENT.match(tail):
            target = "parent"
        else:
            target = "self"
        out += [(kind, n.upper(), target) for n in sorted(nums)]
    return out


def build() -> dict:
    from isr.store import UnitStore
    XREF_DIR.mkdir(parents=True, exist_ok=True)
    s = UnitStore()
    prov = s.units[s.units["kind"] == "provision"]
    docs = s.docs
    by_doc = prov.groupby("document_id")["unit_id"].apply(list)
    doc_ids = sorted(by_doc.index)
    n_chunks = (len(doc_ids) + CHUNK_DOCS - 1) // CHUNK_DOCS
    t0 = time.time()
    for c in range(n_chunks):
        part = XREF_DIR / f"edges.part{c:03d}.npy"
        if part.exists():
            continue
        pause.check(f"xref chunk {c}/{n_chunks}")
        edges, stats = [], {"self": 0, "parent": 0, "external": 0, "unresolved": 0}
        for did in doc_ids[c * CHUNK_DOCS:(c + 1) * CHUNK_DOCS]:
            dtype = str(docs.at[did, "document_type"]) if did in docs.index else ""
            parent = docs.at[did, "parent_document_id"] if did in docs.index else None
            parent = parent if isinstance(parent, str) and parent and parent != "None" else None
            text = s.doc_text(did)
            for u in by_doc[did]:
                own = str(s.units.at[u, "key"]).rsplit("/", 1)[-1].split(":", 1)[-1].upper()
                for kind, n, target in extract(text[s.start[u]:s.end[u]]):
                    stats[target] += 1
                    if target == "external":
                        continue
                    tdoc = did
                    if target == "parent" or (kind == "section" and dtype in ("rule", "regulation") and parent):
                        tdoc = parent or did
                    if tdoc == did and n == own:
                        continue
                    dst = s.provisions_numbered(tdoc, n)
                    if not dst:
                        stats["unresolved"] += 1
                        continue
                    src_rep = int(s.rep_of_class[s.class_id[u]])
                    for d in dst:
                        dst_rep = int(s.rep_of_class[s.class_id[d]])
                        if dst_rep != src_rep:
                            edges.append((src_rep, dst_rep))
        arr = np.unique(np.asarray(edges, np.int64).reshape(-1, 2), axis=0)
        pause.save_npy(part, arr)
        pause.save_json(XREF_DIR / f"stats.part{c:03d}.json", stats)
        print(f"  xref chunk {c + 1}/{n_chunks}: {len(arr)} edges ({time.time() - t0:.0f}s)", flush=True)
    allp = [np.load(XREF_DIR / f"edges.part{c:03d}.npy") for c in range(n_chunks)]
    edges = np.unique(np.concatenate(allp), axis=0)
    pause.save_npy(XREF_DIR / "edges.npy", edges)
    tot: dict[str, int] = {}
    for c in range(n_chunks):
        for k, v in json.loads((XREF_DIR / f"stats.part{c:03d}.json").read_text(encoding="utf-8")).items():
            tot[k] = tot.get(k, 0) + v
    info = {"edges": int(len(edges)), "references": tot, "seconds": round(time.time() - t0, 1)}
    pause.save_json(XREF_DIR / "info.json", info)
    return info


class XrefGraph:
    """Out- and in-neighbours of representative provision units."""

    def __init__(self) -> None:
        e = np.load(XREF_DIR / "edges.npy")
        self.out: dict[int, list[int]] = {}
        self.inc: dict[int, list[int]] = {}
        for a, b in e:
            self.out.setdefault(int(a), []).append(int(b))
            self.inc.setdefault(int(b), []).append(int(a))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build"])
    args = p.parse_args()
    try:
        print(json.dumps(build(), indent=1))
    except pause.Paused as e:
        print(e)
        return pause.EXIT_PAUSED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
