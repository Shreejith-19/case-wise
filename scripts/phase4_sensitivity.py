"""Phase 4 sensitivity check (DEV_TUNE ONLY, report only -- the frozen C-full is not changed by it).

    python scripts/phase4_sensitivity.py

Varies one cross-reference expansion setting at a time around C-full (m, max_per, min_ce, incoming) and reports
cross_reference all-groups / recall and the number of recall changes in the other categories at B=8000.
A setting that only works at one exact value would show up here as a lucky peak.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from isr.eval import evaluate as ev
from isr.eval.kit import questions_for
from isr.pack import pack
from isr.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [("frozen", {}), ("m=2", {"m": 2}), ("m=5", {"m": 5}), ("max_per=1", {"max_per": 1}),
            ("max_per=3", {"max_per": 3}), ("min_ce=0.018", {"min_ce": 0.018}), ("min_ce=0.1", {"min_ce": 0.1}),
            ("no incoming", {"incoming": False}), ("no expansion", None)]


def main() -> int:
    base = yaml.safe_load((ROOT / "configs" / "C_full.yaml").read_text(encoding="utf-8"))
    base.pop("llm", None)
    base.pop("answer", None)
    qs = questions_for("dev_tune")
    pipe = Pipeline(base)
    ref = None
    print(f"{'variant':<14} {'xref all-groups':>16} {'xref recall':>12} {'other recall +/-':>18}")
    for name, change in VARIANTS:
        cfg = copy.deepcopy(base)
        if change is None:
            cfg.pop("planc")
        else:
            cfg["planc"]["xref"].update(change)
        pipe.config = cfg
        run = {}
        for q in qs:
            ranked, tr = pipe.rank(q["question"])
            run[q["question_id"]] = ({b: [] for b in (2000, 8000, 32000)} if tr.abstain_reason else
                                     {b: pack(ranked, b, pipe.store, min_piece=200) for b in (2000, 8000, 32000)})
        rows = ev.question_rows(qs, run)["retrieval"][8000]
        x = [q["question_id"] for q in qs if q["category"] == "cross_reference"]
        oth = [q["question_id"] for q in qs if q["category"] != "cross_reference" and not q["unanswerable"]]
        if ref is None:
            ref = rows
        ag = sum(rows[i]["all_groups"] for i in x) / len(x)
        rc = sum(rows[i]["evidence_recall"] for i in x) / len(x)
        d = [rows[i]["evidence_recall"] - ref[i]["evidence_recall"] for i in oth]
        print(f"{name:<14} {ag:>16.3f} {rc:>12.3f} {sum(v > 0 for v in d):>8}/{sum(v < 0 for v in d):<9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
