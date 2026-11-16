import json

import pytest

from isr.eval import evaluate as ev
from isr.eval.kit import BUDGETS, questions_for
from isr.eval.predictions import write_predictions


def test_bootstrap_ci_constant_and_empty():
    c = ev.bootstrap_ci([1.0] * 10, n_boot=200)
    assert c["mean"] == c["lo"] == c["hi"] == 1.0 and c["n"] == 10
    assert ev.bootstrap_ci([]) is None
    assert ev.bootstrap_ci([None, None]) is None


def test_bootstrap_ci_brackets_mean():
    c = ev.bootstrap_ci([0, 1] * 20, n_boot=500)
    assert c["lo"] <= c["mean"] == 0.5 <= c["hi"]


def test_paired_bootstrap_direction_and_significance():
    better = ev.paired_bootstrap([1.0] * 30, [0.0] * 30, n_boot=300)
    assert better["diff"] == 1.0 and better["p"] < 0.05
    same = ev.paired_bootstrap([0.5] * 30, [0.5] * 30, n_boot=300)
    assert same["diff"] == 0.0 and same["p"] == 1.0


def test_split_sizes():
    assert len(questions_for("dev")) == 150
    assert len(questions_for("test")) == 352


def _oracle_run(questions):
    run = {}
    for q in questions:
        spans = [(loc["document_id"], loc["char_start"], loc["char_end"])
                 for g in q["gold_evidence"] for loc in g["locations"][:1]]
        run[q["question_id"]] = {b: spans for b in BUDGETS}
    return run


def test_end_to_end_matches_official_and_compares(tmp_path):
    qs = questions_for("dev")
    oracle = write_predictions(tmp_path / "o.json", system="o", split="dev", run=_oracle_run(qs))
    empty = write_predictions(tmp_path / "e.json", system="e", split="dev",
                              run={q["question_id"]: {b: [] for b in BUDGETS} for q in qs})
    res = ev.evaluate(oracle, "dev", compare_path=empty, n_boot=100)
    off = res["official"]["retrieval"]["by_category"]
    for cat, by_b in res["ci"]["retrieval"].items():
        for b, ms in by_b.items():
            assert ms["evidence_recall"]["mean"] == pytest.approx(off[cat][b]["evidence_recall"])
    # abstain-everything run: oracle has fewer abstentions on unanswerable, none on answerable
    assert res["comparison"]["abstention"]["false_abstention"]["diff"] == pytest.approx(-1.0)
    assert res["comparison"]["retrieval"]["situational"]["8000"]["evidence_recall"]["diff"] == 1.0


def test_split_mismatch_rejected(tmp_path):
    qs = questions_for("dev")
    f = write_predictions(tmp_path / "o.json", system="o", split="test", run=_oracle_run(qs))
    with pytest.raises(ValueError, match="says split"):
        ev.evaluate(f, "dev", n_boot=10)
