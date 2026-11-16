"""Phase 1.8: alignment of the unit store with gold evidence — DEV SPLIT ONLY (plan.md D6).

    python scripts/check_units.py   -> data/units_alignment_dev.json

Per dev gold location / group:
* exact     — a provision unit has exactly the gold span (expected: all)
* rep_hit   — some location of the group is itself an indexed representative unit
* rep_cover — the group's gold provision class has a representative whose span lies in
              a document that is one of the group's acceptable locations
              (else the packer must swap the representative for a member copy)
* b2000     — best coverage of the gold span by sub-units fitting in 2,000 chars
              (contiguous run), vs. the oracle's own truncation at 2,000
"""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq

from isr import DATA_DIR
from isr.corpus.build import UNITS_DIR
from isr.eval.kit import questions_for

TAU = 0.5


def main() -> int:
    qs = questions_for("dev")
    gold_docs = {loc["document_id"] for q in qs for g in q["gold_evidence"] for loc in g["locations"]}
    u = pq.read_table(UNITS_DIR, columns=["unit_id", "document_id", "kind", "key", "parent_id",
                                          "char_start", "char_end", "class_id", "is_rep"]).to_pandas()
    reps_by_class = u[u["is_rep"]].set_index("class_id")
    gu = u[u["document_id"].isin(gold_docs)]
    prov = {(r.document_id, r.char_start, r.char_end): r for r in gu[gu["kind"] == "provision"].itertuples()}
    subs = gu[gu["kind"] == "subunit"].groupby("parent_id")

    c = Counter()
    b2000 = []
    for q in qs:
        for g in q["gold_evidence"]:
            c["groups"] += 1
            locs = g["locations"]
            group_docs = {loc["document_id"] for loc in locs}
            hit_rep = False
            rep_in_group = False
            for loc in locs:
                c["locations"] += 1
                r = prov.get((loc["document_id"], loc["char_start"], loc["char_end"]))
                if r is None:
                    c["not_exact"] += 1
                    continue
                c["exact"] += 1
                hit_rep |= bool(r.is_rep)
                rep = reps_by_class.loc[r.class_id]
                rep_in_group |= rep["document_id"] in group_docs
            c["groups_rep_hit"] += hit_rep
            c["groups_rep_in_group"] += rep_in_group
            # best coverage within 2,000 chars, using the first location's own sub-units
            loc = locs[0]
            length = loc["char_end"] - loc["char_start"]
            r = prov.get((loc["document_id"], loc["char_start"], loc["char_end"]))
            if r is not None and length > 2000:
                pieces = subs.get_group(r.unit_id).sort_values("char_start") if r.unit_id in subs.groups else None
                best = 0.0
                if pieces is not None:
                    spans = list(zip(pieces["char_start"], pieces["char_end"]))
                    for i in range(len(spans)):            # contiguous runs that fit the budget
                        for j in range(i, len(spans)):
                            if spans[j][1] - spans[i][0] > 2000:
                                break
                            best = max(best, (spans[j][1] - spans[i][0]) / length)
                b2000.append({"question_id": q["question_id"], "len": length, "best_cover": round(best, 3),
                              "oracle_cover": round(2000 / length, 3)})
    out = {"split": "dev", "counts": dict(c),
           "long_gold_locations_over_2000": len(b2000),
           "long_gold_reaching_tau_0.5_at_2000": sum(x["best_cover"] >= TAU for x in b2000),
           "oracle_reaching_tau_0.5_at_2000": sum(x["oracle_cover"] >= TAU for x in b2000),
           "long_gold": b2000}
    (DATA_DIR / "units_alignment_dev.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "long_gold"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
