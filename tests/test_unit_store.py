"""Integrity of the built unit store (skipped until `python -m isr.corpus.build` has run)."""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from isr import CORPUS_DIR
from isr.corpus.build import DOCS_PATH, UNITS_DIR

pytestmark = pytest.mark.skipif(not UNITS_DIR.exists(), reason="unit store not built")


@pytest.fixture(scope="module")
def units():
    return pq.read_table(UNITS_DIR, columns=["unit_id", "document_id", "kind", "key", "parent_id",
                                             "char_start", "char_end", "class_id", "is_rep",
                                             "class_size"]).to_pandas()


def test_ids_contiguous_and_parents_valid(units):
    assert (units["unit_id"].to_numpy() == np.arange(len(units))).all()
    subs = units[units["kind"] == "subunit"]
    parents = units.set_index("unit_id").loc[subs["parent_id"]]
    assert (parents["kind"] == "provision").all()
    assert (parents["document_id"].to_numpy() == subs["document_id"].to_numpy()).all()
    assert (units.loc[units["kind"] != "subunit", "parent_id"] == -1).all()


def test_exactly_one_representative_per_class(units):
    reps = units.groupby("class_id")["is_rep"].sum()
    assert (reps == 1).all()
    assert (units.groupby("class_id").size() == units.groupby("class_id")["class_size"].first()).all()


def test_spans_forward_and_text_matches(units):
    assert (units["char_end"] > units["char_start"]).all()
    sample = pq.read_table(UNITS_DIR / "part-0005.parquet",
                           columns=["document_id", "char_start", "char_end", "text"]).to_pandas().sample(
        300, random_state=0)
    for r in sample.itertuples():
        text = (CORPUS_DIR / "text" / f"{r.document_id}.txt").read_text(encoding="utf-8")
        assert text[r.char_start:r.char_end] == r.text


def test_provisions_complete_and_docs_table(units):
    docs = pd.read_parquet(DOCS_PATH)
    assert len(docs) == 15325 and docs["document_id"].is_unique
    n_prov = units[units["kind"] == "provision"].groupby("document_id").size()
    expected = docs.set_index("document_id")["n_provisions"]
    assert (n_prov.reindex(expected.index, fill_value=0) == expected).all()
