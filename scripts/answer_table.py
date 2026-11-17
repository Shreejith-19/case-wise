"""Side-by-side answer metrics (official kit numbers) from evaluate reports.

    python scripts/answer_table.py runs/B0-*/pred_dev_tune.report.json runs/B6-*/pred_dev_tune.report.json ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

METRICS = (("fact_coverage", "fact"), ("citation_recall", "c_rec"), ("citation_precision", "c_prec"),
           ("quote_grounding", "quote"))


def main(paths: list[str]) -> int:
    reps = [(Path(p).parent.name, json.loads(Path(p).read_text(encoding="utf-8"))["official"]["answers"]) for p in paths]
    cats = [c for c in reps[0][1]["by_category"] if c != "unanswerable"]
    for key, short in METRICS:
        print(f"\n{short:<8}" + "".join(f"{n[:22]:>24}" for n, _ in reps))
        for c in cats + ["MEAN"]:
            cells = []
            for _, a in reps:
                if c == "MEAN":
                    vals = [a["by_category"][k][key] for k in cats if a["by_category"][k][key] is not None]
                    v = sum(vals) / len(vals) if vals else None
                else:
                    v = a["by_category"][c][key]
                cells.append("-" if v is None else f"{v:.3f}")
            print(f"{c[:8]:<8}" + "".join(f"{x:>24}" for x in cells))
    print("\nabstained (answerable)" + "".join(
        f"{sum(a['by_category'][c]['abstained'] for c in cats):>8}" for _, a in reps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
