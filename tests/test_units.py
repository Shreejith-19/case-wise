import hashlib
import json

import pytest

from isr import CORPUS_DIR
from isr.corpus import units as U


def _section(key, s, e, children=(), heading="H"):
    return {"unit_type": "section", "number": key.split(":")[-1], "key": key, "heading": heading,
            "char_start": s, "char_end": e, "children": list(children)}


def _child(t, n, s, e, children=()):
    return {"unit_type": t, "number": n, "char_start": s, "char_end": e, "children": list(children)}


def _prov(key, s, e):
    return {"key": key, "unit_type": "section", "heading": "H", "char_start": s, "char_end": e,
            "unit_path": [{"unit_type": "section", "number": key.split(":")[-1]}],
            "evidence_confidence": "high", "key_ambiguous": False}


def _tiles_exactly(pieces, start, end):
    assert pieces[0].char_start == start and pieces[-1].char_end == end
    for a, b in zip(pieces, pieces[1:]):
        assert a.char_end == b.char_start


def test_short_document_provisions_and_gap_windows():
    text = "PREAMBLE " * 20 + "\n" + "x" * 500 + "\n" + "SCHEDULE " * 300
    p_s = text.index("x")
    p_e = p_s + 500
    st = {"provisions": [_prov("section:1", p_s, p_e)], "units": [_section("section:1", p_s, p_e)],
          "preamble": {"char_start": 0, "char_end": p_s}}
    us = U.document_units(text, st)
    provs = [u for u in us if u.kind == "provision"]
    wins = [u for u in us if u.kind == "window"]
    assert len(provs) == 1 and not [u for u in us if u.kind == "subunit"]
    assert wins[0].region == "preamble" and wins[0].char_end <= p_s
    # every non-space char outside the provision is inside some window
    covered = set()
    for w in wins:
        covered.update(range(w.char_start, w.char_end))
    for i, c in enumerate(text):
        if not c.isspace() and not (p_s <= i < p_e):
            assert i in covered
    assert all(w.char_end - w.char_start <= U.WINDOW for w in wins)


def test_long_provision_tiles_along_children():
    body = "".join(f"({i}) " + "w" * 595 + "\n" for i in range(1, 9))  # 8 subsections of 600 chars
    text = "s. 1 Heading\n" + body
    s, e = 0, len(text)
    kids = [_child("subsection", str(i + 1), 13 + i * 600, 13 + (i + 1) * 600) for i in range(8)]
    st = {"provisions": [_prov("section:1", s, e)], "units": [_section("section:1", s, e, kids)]}
    subs = [u for u in U.document_units(text, st) if u.kind == "subunit"]
    _tiles_exactly(subs, s, e)
    assert all(u.char_end - u.char_start <= U.SUB_MAX for u in subs)
    assert subs[0].label.endswith("(1)–(2)")         # lead-in merged with first children
    assert all(u.parent == 0 and u.key == "section:1" for u in subs)


def test_oversized_child_and_no_children_fall_back_to_lines():
    text = ("line of text here\n" * 400)   # 7200 chars, no children
    st = {"provisions": [_prov("section:9", 0, len(text))], "units": []}
    subs = [u for u in U.document_units(text, st) if u.kind == "subunit"]
    _tiles_exactly(subs, 0, len(text))
    assert all(u.char_end - u.char_start <= U.SUB_MAX for u in subs)
    assert all(text[u.char_end - 1] == "\n" for u in subs)  # cut at line ends


def test_unstructured_document_is_all_windows():
    text = "word " * 1000
    us = U.document_units(text, {"provisions": [], "units": []})
    assert us and all(u.kind == "window" and u.region == "unparsed" for u in us)


def test_normalise_ignores_case_and_punctuation():
    assert U.norm_hash("The  Act, 2005.") == U.norm_hash("the act 2005")
    assert U.norm_hash("section 2") != U.norm_hash("section 3")


@pytest.mark.parametrize("did", [
    "prevention-and-control-of-infectious-and-contagious-diseases__handle-21045",
])
def test_real_document_invariants(did):
    text = (CORPUS_DIR / "text" / f"{did}.txt").read_text(encoding="utf-8")
    st = json.loads((CORPUS_DIR / "structure" / f"{did}.json").read_text(encoding="utf-8"))
    us = U.document_units(text, st)
    provs = [u for u in us if u.kind == "provision"]
    assert len(provs) == len(st["provisions"])
    by_key = {p["key"]: p for p in st["provisions"]}
    for u in provs:  # offsets round-trip: the provision text hashes to the corpus's own sha256
        p = by_key[u.key]
        assert hashlib.sha256(text[u.char_start:u.char_end].encode("utf-8")).hexdigest() == p["text_sha256"]
    for i, pu in enumerate(us):
        if pu.kind != "provision" or pu.char_end - pu.char_start <= U.SUB_MAX:
            continue
        _tiles_exactly([u for u in us if u.kind == "subunit" and u.parent == i], pu.char_start, pu.char_end)
