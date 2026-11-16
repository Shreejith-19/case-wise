"""Score a predictions file with the official scorer, plus bootstrap CIs and paired comparison.

    python -m isr.eval.evaluate runs/x/pred_dev.json --split dev
    python -m isr.eval.evaluate runs/x/pred_dev.json --split dev --compare runs/baselines/bm25-windows_dev.json
    python -m isr.eval.evaluate runs/x/pred_test.json --split test --test-note "frozen A-full, commit abc123"

Every headline number comes from the kit's ``score_run``; this module only adds
uncertainty (per-category bootstrap 95% CIs over questions) and a paired bootstrap
test against a reference run. Test-split evaluations must carry ``--test-note`` and
are appended to TEST_RUNS.md (plan.md decision D6).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from isr import CORPUS_DIR, ROOT
from isr.eval.kit import BUDGETS, HEADLINE_BUDGET, TAU, kit_score, load_predictions, questions_for

METRICS = ("evidence_recall", "evidence_precision", "all_groups", "document_recall", "mrr")
DEFAULT_BOOT = 1000
DEFAULT_SEED = 20261116
TEST_LOG = ROOT / "TEST_RUNS.md"


# --- per-question rows ----------------------------------------------------------------


def question_rows(questions: Sequence[dict], run: dict, budgets: Sequence[int] = BUDGETS,
                  tau: float = TAU) -> dict:
    """{'retrieval': {budget: {qid: row}}, 'abstained': {qid: bool at headline budget}}."""
    retrieval: dict[int, dict[str, dict]] = {}
    for b in budgets:
        retrieval[b] = {}
        for q in questions:
            if q["unanswerable"]:
                continue
            spans = run.get(q["question_id"], {}).get(b, [])
            retrieval[b][q["question_id"]] = kit_score.score_question(q, spans, b, tau=tau)
    abstained = {
        q["question_id"]: kit_score.within_budget(run.get(q["question_id"], {}).get(HEADLINE_BUDGET, []),
                                                  HEADLINE_BUDGET).is_empty()
        for q in questions}
    return {"retrieval": retrieval, "abstained": abstained}


# --- statistics -----------------------------------------------------------------------


def bootstrap_ci(values: Sequence[float], n_boot: int = DEFAULT_BOOT, seed: int = DEFAULT_SEED,
                 alpha: float = 0.05) -> Optional[dict]:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(v.size)}


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = DEFAULT_BOOT,
                     seed: int = DEFAULT_SEED, alpha: float = 0.05) -> Optional[dict]:
    """Difference mean(a) - mean(b) over paired questions; two-sided bootstrap p-value."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "p": float(min(1.0, p)), "n": int(d.size)}


# --- aggregation ----------------------------------------------------------------------


def _by_category(questions: Sequence[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for q in questions:
        if not q["unanswerable"]:
            out.setdefault(q["category"], []).append(q["question_id"])
    return dict(sorted(out.items()))


def confidence_intervals(questions: Sequence[dict], rows: dict, n_boot: int, seed: int) -> dict:
    cats = _by_category(questions)
    ci: dict = {"retrieval": {}, "abstention": {}}
    for cat, qids in cats.items():
        ci["retrieval"][cat] = {}
        for b, by_q in rows["retrieval"].items():
            ci["retrieval"][cat][str(b)] = {
                m: bootstrap_ci([by_q[q].get(m) for q in qids], n_boot, seed) for m in METRICS}
    unans = [q["question_id"] for q in questions if q["unanswerable"]]
    ans = [q["question_id"] for q in questions if not q["unanswerable"]]
    ci["abstention"] = {
        "budget": HEADLINE_BUDGET,
        "abstention_rate": bootstrap_ci([float(rows["abstained"][q]) for q in unans], n_boot, seed),
        "false_abstention": bootstrap_ci([float(rows["abstained"][q]) for q in ans], n_boot, seed),
    }
    return ci


def compare(questions: Sequence[dict], rows: dict, ref_rows: dict, n_boot: int, seed: int) -> dict:
    cats = _by_category(questions)
    out: dict = {"retrieval": {}, "abstention": {}}
    for cat, qids in cats.items():
        out["retrieval"][cat] = {}
        for b in rows["retrieval"]:
            mine, ref = rows["retrieval"][b], ref_rows["retrieval"][b]
            out["retrieval"][cat][str(b)] = {}
            for m in METRICS:
                pairs = [(mine[q].get(m), ref[q].get(m)) for q in qids]
                if m == "evidence_precision":
                    # Undefined when a system returned nothing; count it as 0 for pairing.
                    pairs = [(x or 0.0, y or 0.0) for x, y in pairs]
                pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                out["retrieval"][cat][str(b)][m] = paired_bootstrap(
                    [x for x, _ in pairs], [y for _, y in pairs], n_boot, seed)
    unans = [q["question_id"] for q in questions if q["unanswerable"]]
    ans = [q["question_id"] for q in questions if not q["unanswerable"]]
    out["abstention"] = {
        "abstention_rate": paired_bootstrap([float(rows["abstained"][q]) for q in unans],
                                            [float(ref_rows["abstained"][q]) for q in unans], n_boot, seed),
        "false_abstention": paired_bootstrap([float(rows["abstained"][q]) for q in ans],
                                             [float(ref_rows["abstained"][q]) for q in ans], n_boot, seed),
    }
    return out


# --- formatting -----------------------------------------------------------------------


def _fmt(c: Optional[dict]) -> str:
    if not c:
        return "      -          "
    return f"{c['mean']:.3f} [{c['lo']:.2f},{c['hi']:.2f}]"


def _fmt_diff(c: Optional[dict]) -> str:
    if not c:
        return "-"
    star = "*" if c["p"] < 0.05 else " "
    return f"{c['diff']:+.3f}{star}"


def format_summary(system: str, split: str, ci: dict, comparison: Optional[dict] = None,
                   ref_name: Optional[str] = None) -> str:
    lines = [f"system: {system}   split: {split}   tau: {TAU}   (mean [95% bootstrap CI])"]
    head = f"{'category':>18} {'B':>6}  {'recall':<19}{'precision':<19}{'all-groups':<19}{'mrr':<19}"
    if comparison:
        head += f" Δrecall vs {ref_name} (* p<.05)"
    lines.append(head)
    for cat, by_b in ci["retrieval"].items():
        for b, ms in by_b.items():
            line = (f"{cat:>18} {b:>6}  {_fmt(ms['evidence_recall']):<19}{_fmt(ms['evidence_precision']):<19}"
                    f"{_fmt(ms['all_groups']):<19}{_fmt(ms['mrr']):<19}")
            if comparison:
                line += f" {_fmt_diff(comparison['retrieval'][cat][b]['evidence_recall'])}"
            lines.append(line)
    a = ci["abstention"]
    line = (f"abstention (B={a['budget']}): unanswerable {_fmt(a['abstention_rate'])}   "
            f"false {_fmt(a['false_abstention'])}")
    if comparison:
        line += (f"   Δ {_fmt_diff(comparison['abstention']['abstention_rate'])} / "
                 f"{_fmt_diff(comparison['abstention']['false_abstention'])}")
    lines.append(line)
    return "\n".join(lines)


# --- entry point ----------------------------------------------------------------------


def evaluate(pred_path: Path, split: str, *, compare_path: Optional[Path] = None,
             n_boot: int = DEFAULT_BOOT, seed: int = DEFAULT_SEED) -> dict:
    from benchmark.questions import Corpus  # kit module, path set up by isr.eval.kit

    questions = questions_for(split)
    loaded = load_predictions(pred_path, questions)
    if loaded["split"] and loaded["split"] != split:
        raise ValueError(f"{pred_path} says split {loaded['split']!r}, asked to score {split!r}")
    official = kit_score.score_run(
        questions, loaded["run"], system=loaded["system"], corpus_dir=CORPUS_DIR,
        corpus=Corpus(CORPUS_DIR), answers=loaded["answers"], budgets=BUDGETS, tau=TAU,
        split=split, tuned_on_test=False)
    rows = question_rows(questions, loaded["run"])
    result = {"system": loaded["system"], "split": split, "predictions": str(pred_path),
              "missing": loaded["missing"], "official": official,
              "ci": confidence_intervals(questions, rows, n_boot, seed),
              "bootstrap": {"resamples": n_boot, "seed": seed}}
    if compare_path:
        ref = load_predictions(compare_path, questions)
        result["compared_to"] = {"system": ref["system"], "predictions": str(compare_path)}
        result["comparison"] = compare(questions, rows, question_rows(questions, ref["run"]), n_boot, seed)
    return result


def _log_test_run(result: dict, note: str) -> None:
    fp = result["official"]["corpus"]
    row = (f"| {dt.datetime.now().isoformat(timespec='minutes')} | {result['system']} | "
           f"`{result['predictions']}` | {fp['checksums_sha256'][:12]} | {note} |\n")
    if not TEST_LOG.exists():
        TEST_LOG.write_text("# Test-split evaluations\n\n| when | system | predictions | corpus | note |\n"
                            "|---|---|---|---|---|\n", encoding="utf-8")
    with open(TEST_LOG, "a", encoding="utf-8") as fh:
        fh.write(row)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m isr.eval.evaluate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("predictions", type=Path)
    p.add_argument("--split", required=True, choices=["dev", "test", "all"])
    p.add_argument("--compare", type=Path, help="Reference predictions file (same split) for paired tests.")
    p.add_argument("--out", type=Path, help="Write the full JSON report here (default: next to predictions).")
    p.add_argument("--boot", type=int, default=DEFAULT_BOOT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--test-note", help="Required for --split test/all: why this test run is happening.")
    args = p.parse_args(argv)

    if args.split in ("test", "all") and not args.test_note:
        p.error("test-split evaluation requires --test-note (it is logged to TEST_RUNS.md; plan.md D6)")

    result = evaluate(args.predictions, args.split, compare_path=args.compare,
                      n_boot=args.boot, seed=args.seed)
    if result["missing"]:
        print(f"WARNING: {len(result['missing'])} question(s) missing from predictions "
              f"(scored as abstentions), e.g. {result['missing'][0]}")
    print(format_summary(result["system"], args.split, result["ci"], result.get("comparison"),
                         result.get("compared_to", {}).get("system")))
    if result["official"].get("answers"):
        print("answers:", json.dumps(result["official"]["answers"], indent=1))
    out = args.out or args.predictions.with_name(args.predictions.stem + ".report.json")
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    if args.split in ("test", "all"):
        _log_test_run(result, args.test_note)
        print(f"logged to {TEST_LOG.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
