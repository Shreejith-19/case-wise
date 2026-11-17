"""Diagnostic (DEV ONLY): candidate recall of each retrieval list in a pipeline trace.

    python scripts/candidate_recall.py runs/<system>/trace_dev.json [more traces ...]

For every answerable dev gold group, a ranked unit is a hit if it is (a dedup copy of) the
gold provision or one of its sub-units. Reports the fraction of gold groups hit within the
top k of each list (bm25, dense, act_bm25, ..., final) — i.e. where gold is lost: before
fusion, by fusion, or by the reranker. Abstained questions are counted from their lists.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from isr.eval.kit import questions_for
from isr.store import UnitStore

KS = (10, 20, 50, 100, 300)


def gold_classes(qs, store: UnitStore) -> dict[str, list[set[int]]]:
    prov = store.units[store.units["kind"] == "provision"]
    by_span = {(d, s, e): u for u, d, s, e in zip(prov["unit_id"], prov["document_id"],
                                                  prov["char_start"], prov["char_end"])}
    sub = store.units[store.units["kind"] == "subunit"]
    subs_of = defaultdict(list)
    for u, p in zip(sub["unit_id"], sub["parent_id"]):
        subs_of[int(p)].append(int(u))
    out = {}
    for q in qs:
        groups = []
        for g in q["gold_evidence"]:
            cls = set()
            for loc in g["locations"]:
                u = by_span.get((loc["document_id"], loc["char_start"], loc["char_end"]))
                if u is None:
                    continue
                cls.add(int(store.class_id[u]))
                cls.update(int(store.class_id[s]) for s in subs_of.get(int(u), []))
            groups.append(cls)
        out[q["question_id"]] = groups
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("traces", nargs="+", type=Path)
    args = p.parse_args()
    qs = [q for q in questions_for("dev") if not q["unanswerable"]]
    cat = {q["question_id"]: q["category"] for q in qs}
    store = UnitStore()
    gold = gold_classes(qs, store)
    for path in args.traces:
        traces = json.loads(path.read_text(encoding="utf-8"))
        # hits[list][k][category] -> [hit groups, total groups]
        hits = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
        for qid, groups in gold.items():
            t = traces[qid]
            lists = dict(t["lists"])
            lists["final"] = t["final"]
            for name, ranked in lists.items():
                ranks = [int(store.class_id[u]) for u in ranked]
                for k in KS:
                    top = set(ranks[:k])
                    for g in groups:
                        cell = hits[name][k]
                        for c in (cat[qid], "ALL"):
                            cell[c][1] += 1
                            cell[c][0] += bool(g & top)
        print(f"\n== {path.parent.name}  (gold groups hit in top-k; answerable dev)")
        cats = sorted({c for c in cat.values()}) + ["ALL"]
        for name in hits:
            print(f"  {name}")
            for k in KS:
                row = "  ".join(f"{c[:6]}={hits[name][k][c][0] / max(1, hits[name][k][c][1]):.2f}"
                                for c in cats if hits[name][k][c][1])
                print(f"    @{k:<4} {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
