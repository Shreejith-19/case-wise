from isr.eval import evaluate as ev


def test_holdout_and_test_runs_go_to_separate_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "TEST_LOG", tmp_path / "TEST_RUNS.md")
    monkeypatch.setattr(ev, "HOLDOUT_LOG", tmp_path / "HOLDOUT_RUNS.md")
    res = {"system": "S", "predictions": "p.json", "official": {"corpus": {"checksums_sha256": "ab" * 20}}}
    ev._log_test_run({**res, "split": "dev_holdout"}, "holdout note")
    assert not (tmp_path / "TEST_RUNS.md").exists()
    assert "holdout note" in (tmp_path / "HOLDOUT_RUNS.md").read_text(encoding="utf-8")
    ev._log_test_run({**res, "split": "test"}, "test note")
    assert "holdout note" not in (tmp_path / "TEST_RUNS.md").read_text(encoding="utf-8")
