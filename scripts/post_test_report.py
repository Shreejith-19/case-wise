"""Post-test exploration report (C-LS vs frozen C-full) -> runs/final/POST_TEST.md. Reads predictions only.

    python scripts/post_test_report.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from isr import RUNS_DIR
from isr.eval import evaluate as ev
from isr.eval.kit import kit_score, questions_for

FINAL = RUNS_DIR / "final"


def rows(name: str, split: str, qs: list[dict]) -> dict:
    return ev.question_rows(qs, ev._load(RUNS_DIR / name / f"pred_{split}.json", split, qs)["run"])["retrieval"]


def ci(a, b) -> str:
    d = ev.paired_bootstrap(a, b, n_boot=5000)
    s = "**" if d["lo"] > 0 or d["hi"] < 0 else ""
    return f"{s}{d['diff']:+.3f}{s} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def main() -> int:
    out = ["# Post-test exploration — C-LS (LLM listwise selection) vs frozen C-full\n",
           "**Not part of the frozen results.** Designed after the test split was spent (plan.md, 'Post-test "
           "exploration'); dev gate fixed in advance and passed (pooled dev R@8k +0.048 [+0.004, +0.093], n=135); "
           "then run once on test (TEST_RUNS.md). Δ = C-LS − C-full, paired bootstrap 95% CI; bold = excludes 0.\n"]
    for split in ("dev_tune", "dev_holdout", "test"):
        qs = questions_for(split)
        ans = [q for q in qs if not q["unanswerable"]]
        L, C = rows("C-LS", split, qs), rows("C-full", split, qs)
        cats = sorted({q["category"] for q in ans})
        out.append(f"\n## {split} (n = {len(ans)} answerable)\n")
        out.append("| budget | recall C-LS | recall C-full | Δ recall | Δ MRR |")
        out.append("|---|---|---|---|---|")
        ids = [q["question_id"] for q in ans]
        for b in (2000, 8000, 32000):
            a = [L[b][i]["evidence_recall"] for i in ids]
            c = [C[b][i]["evidence_recall"] for i in ids]
            out.append(f"| {b} | {sum(a) / len(a):.3f} | {sum(c) / len(c):.3f} | {ci(a, c)} | "
                       f"{ci([L[b][i]['mrr'] for i in ids], [C[b][i]['mrr'] for i in ids])} |")
        out.append("\n| category @8k | C-LS | C-full |")
        out.append("|---|---|---|")
        for cat in cats:
            x = [q["question_id"] for q in ans if q["category"] == cat]
            out.append(f"| {cat} ({len(x)}) | {sum(L[8000][i]['evidence_recall'] for i in x) / len(x):.3f} | "
                       f"{sum(C[8000][i]['evidence_recall'] for i in x) / len(x):.3f} |")
        qmap = {q["question_id"]: q for q in qs}
        import json
        A = {n: json.loads((RUNS_DIR / n / f"pred_{split}.json").read_text(encoding="utf-8"))["answers"]
             for n in ("C-LS", "C-full")}
        S = {n: {i: kit_score.score_answer(qmap[i], A[n].get(i)) for i in ids} for n in A}
        out.append("\n| answers | C-LS | C-full | Δ |")
        out.append("|---|---|---|---|")
        for m in ("fact_coverage", "citation_recall", "citation_precision"):
            a = [S["C-LS"][i].get(m) or 0.0 for i in ids]
            c = [S["C-full"][i].get(m) or 0.0 for i in ids]
            out.append(f"| {m} | {sum(a) / len(a):.3f} | {sum(c) / len(c):.3f} | {ci(a, c)} |")
    FINAL.mkdir(exist_ok=True)
    (FINAL / "POST_TEST.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    dst = FINAL / "C-LS"
    dst.mkdir(exist_ok=True)
    for f in ("pred_test.json", "pred_test.report.json"):
        shutil.copy2(RUNS_DIR / "C-LS" / f, dst / f)
    print(f"wrote {FINAL / 'POST_TEST.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
