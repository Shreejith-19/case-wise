"""Overfitting audit for the frozen Plan A system — never touches the test split.

    python scripts/overfit_audit.py threshold-cv runs/A-full/trace_dev.json
    python scripts/overfit_audit.py probe        [--n 400]
    python scripts/overfit_audit.py sensitivity  configs/A_full.yaml

threshold-cv  Repeated stratified 5-fold CV of the abstention-threshold rule on dev traces: the
              threshold is re-chosen on 4/5 of dev and applied to the held-out 1/5. Out-of-fold
              rates vs in-sample rates = the optimism of the tuned threshold.
probe         Held-out check of the hand-written parser + abstention rules on synthetic questions
              about Acts that do NOT appear in dev gold (no benchmark labels used):
                answerable   real section of an unseen Act       -> must not abstain, must pin it
                fake_section same Act, section number past its end -> should abstain (missing_provision)
                swapped_state state Act renamed to another state    -> should abstain
                foreign      foreign statutes                      -> should abstain
              Each Act is asked with its raw corpus title and a cleaned title. Same probe on dev for
              comparison (in-sample). Templated phrasing: measures generalisation over Acts/titles,
              not over question style.
sensitivity   One-at-a-time perturbation of every hyperparameter that was set by default (never
              searched). Flat = the frozen config is not sitting on a lucky dev peak. The frozen
              config is NOT changed based on this.
Outputs -> runs/audit/*.json
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from isr import RUNS_DIR
from isr.eval.kit import questions_for

OUT = RUNS_DIR / "audit"
LAM, MAX_FALSE = 2.0, 0.05
ANSWERABLE_CATS = ["provision_lookup", "definitional", "numeric_threshold", "cross_reference",
                   "jurisdictional", "situational"]


# ----------------------------------------------------------------------------- threshold CV
def _rows(trace_path: Path):
    traces = json.loads(trace_path.read_text(encoding="utf-8"))
    qs = {q["question_id"]: q for q in questions_for("dev")}
    return [(bool(qs[qid]["unanswerable"]), t["abstain_reason"] not in (None, "low_rerank_score"), bool(t["lists"].get("exact_ref")),
             t["rerank_top"] if t["rerank_top"] is not None else -1e9) for qid, t in traces.items()]


def _rates(rows, thr):
    n_u = sum(r[0] for r in rows)
    n_a = len(rows) - n_u
    ab_u = ab_a = 0
    for unans, ruled, pinned, top in rows:
        abst = ruled or (not pinned and top < thr)
        ab_u += unans and abst
        ab_a += (not unans) and abst
    return ab_u, n_u, ab_a, n_a


def choose_threshold(rows) -> float:
    """The pre-registered rule: maximise abstention - LAM*false subject to false <= MAX_FALSE."""
    cands = sorted({r[3] for r in rows if r[3] > -1e9})
    grid = [-1e9] + [(a + b) / 2 for a, b in zip(cands, cands[1:])]
    best = (-1e18, -1e9)
    for thr in grid:
        ab_u, n_u, ab_a, n_a = _rates(rows, thr)
        false = ab_a / max(1, n_a)
        if false > MAX_FALSE:
            continue
        obj = ab_u / max(1, n_u) - LAM * false
        if obj > best[0] + 1e-12:
            best = (obj, thr)
    return best[1]


def threshold_cv(trace: Path, repeats: int = 500, folds: int = 5) -> dict:
    rows = _rows(trace)
    thr_in = choose_threshold(rows)
    ab_u, n_u, ab_a, n_a = _rates(rows, thr_in)
    rng = np.random.default_rng(20261117)
    u_idx = [i for i, r in enumerate(rows) if r[0]]
    a_idx = [i for i, r in enumerate(rows) if not r[0]]
    oof_u, oof_a, chosen = [], [], []
    for _ in range(repeats):
        fold_of = {}
        for idx in (u_idx, a_idx):                      # stratified on answerability
            perm = rng.permutation(idx)
            for j, i in enumerate(perm):
                fold_of[int(i)] = j % folds
        cu = ca = 0
        for f in range(folds):
            train = [rows[i] for i in range(len(rows)) if fold_of[i] != f]
            held = [rows[i] for i in range(len(rows)) if fold_of[i] == f]
            thr = choose_threshold(train)
            chosen.append(thr)
            hu, _, ha, _ = _rates(held, thr)
            cu += hu
            ca += ha
        oof_u.append(cu / n_u)
        oof_a.append(ca / n_a)
    no_thr = _rates(rows, -1e9)
    fin = [c for c in chosen if c > -1e9]
    res = {
        "in_sample": {"threshold": thr_in, "abstention": ab_u / n_u, "false_abstention": ab_a / n_a},
        "out_of_fold": {"abstention_mean": float(np.mean(oof_u)), "abstention_p05_p95": np.percentile(oof_u, [5, 95]).tolist(),
                        "false_mean": float(np.mean(oof_a)), "false_p05_p95": np.percentile(oof_a, [5, 95]).tolist()},
        "rules_only_no_threshold": {"abstention": no_thr[0] / no_thr[1], "false_abstention": no_thr[2] / no_thr[3]},
        "chosen_threshold_distribution": {"never": sum(c == -1e9 for c in chosen) / len(chosen),
                                          "p05_p50_p95": np.percentile(fin, [5, 50, 95]).tolist() if fin else None},
        "folds": folds, "repeats": repeats, "lam": LAM, "max_false": MAX_FALSE,
    }
    print(json.dumps(res, indent=1))
    return res


# ----------------------------------------------------------------------------- synthetic probe
_SMALL = {"of", "and", "the", "for", "in", "to", "on", "by", "with", "from", "or", "an", "a", "at"}
TEMPLATES = ["What does section {n} of the {t} provide?",
             "Under section {n} of the {t}, what is required?",
             "What is laid down in section {n} of the {t}?",
             "According to section {n} of the {t}, who is responsible and for what?"]
FOREIGN = ["Singapore's Personal Data Protection Act 2012", "the UK Companies Act 2006",
           "the United States Sherman Antitrust Act, 1890", "Australia's Privacy Act 1988",
           "the Canadian Criminal Code", "Germany's Federal Data Protection Act",
           "the UK Human Rights Act 1998", "New Zealand's Resource Management Act 1991",
           "the US Freedom of Information Act", "South Africa's Promotion of Access to Information Act, 2000",
           "the UK Bribery Act 2010", "the Sri Lankan Penal Code", "Nepal's Muluki Civil Code, 2017",
           "the Bangladesh Labour Act, 2006", "Pakistan's Prevention of Electronic Crimes Act, 2016",
           "the UK Equality Act 2010", "the US Clean Air Act", "Kenya's Data Protection Act, 2019",
           "the Malaysian Contracts Act 1950", "the European Union's Digital Services Act"]


def clean_title(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip().rstrip(".")
    t = re.sub(r",\s*(\d{4})$", r", \1", t)
    if t.isupper() or t.islower():
        words = t.lower().split(" ")
        t = " ".join(w if (i and w in _SMALL) else (w[:1].upper() + w[1:]) for i, w in enumerate(words))
        t = re.sub(r"\(([a-z])", lambda m: "(" + m.group(1).upper(), t)
    return re.sub(r"^the\s+", "", t, flags=re.I)


def probe(n: int = 400) -> dict:
    import pandas as pd
    from isr.pipeline import named_provisions, rule_abstain_reason
    from isr.query.rules import JURISDICTIONS, _strip_title, parse, title_index
    from isr.store import UnitStore

    ab = {"missing_provision": True, "unresolved_law": True}
    store = UnitStore()
    docs = store.docs
    rng = random.Random(20261117)

    def check(question):
        p = parse(question)
        nums, exact = named_provisions(p, store)
        return p, exact, rule_abstain_reason(p, nums, exact, ab)

    # dev in-sample reference (same rule code)
    dev = defaultdict(Counter)
    for q in questions_for("dev"):
        _, _, reason = check(q["question"])
        k = "unanswerable" if q["unanswerable"] else "answerable"
        dev[k]["n"] += 1
        dev[k]["abstain"] += reason is not None
        dev[k][f"reason:{reason}"] += 1

    # exclude every Act whose duplicate cluster contains a dev gold document
    dev_docs = {loc["document_id"] for q in questions_for("dev") for g in q["gold_evidence"] for loc in g["locations"]}
    # (most documents have no cluster: exclude by id, by non-null cluster, and by normalised title)
    gold = docs.loc[docs.index.intersection(list(dev_docs))]
    excl = {c for c in gold["cluster_id"] if c is not None and c == c}
    excl_titles = {_strip_title(str(t))[0] for t in gold["title"]}
    titles_norm = docs["title"].map(lambda t: _strip_title(str(t))[0])
    is_rep = docs["is_cluster_rep"].fillna(True).astype(bool) | docs["cluster_id"].isna()
    cand = docs[docs["document_type"].isin(["state_act", "central_act"]) & is_rep
                & ~docs.index.isin(list(dev_docs)) & ~(docs["cluster_id"].notna() & docs["cluster_id"].isin(excl))
                & ~titles_norm.isin(excl_titles) & (docs["n_provisions"] > 0)]
    print(f"probe pool: {len(cand)} Acts ({int(cand['cluster_id'].isna().sum())} singletons); "
          f"excluded {len(dev_docs)} dev gold docs, {len(excl)} clusters, {len(excl_titles)} titles", flush=True)
    units = store.units
    sec_re = re.compile(r"(?:^|/)section:(\d+[A-Z]?)$")
    tidx = title_index()
    states = [s for s in JURISDICTIONS if s not in ("India", "Central")]

    results = defaultdict(lambda: defaultdict(Counter))
    failures = defaultdict(list)
    dids = list(cand.index)
    rng.shuffle(dids)
    used = 0
    for did in dids:
        if used >= n:
            break
        du = units.iloc[store.doc_units(did)]
        prov = du[du["kind"] == "provision"]
        secs = [(int(u), sec_re.search(k).group(1)) for u, k in zip(prov["unit_id"], prov["key"]) if sec_re.search(k)]
        if not secs:
            continue
        used += 1
        raw = str(docs.at[did, "title"]).strip()
        variants = {"raw_title": raw, "clean_title": clean_title(raw)}
        uid, num = rng.choice(secs)
        tmpl = rng.choice(TEMPLATES)
        cid = docs.at[did, "cluster_id"]
        cluster_docs = {did} | (set(docs.index[docs["cluster_id"] == cid]) if cid is not None and cid == cid else set())
        maxnum = max(int(re.match(r"\d+", s).group()) for _, s in secs)
        fake = str(maxnum + rng.randint(5, 40))
        for vname, title in variants.items():
            # answerable
            q = tmpl.format(n=num, t=title)
            p, exact, reason = check(q)
            r = results[vname]["answerable"]
            r["n"] += 1
            r["act_resolved"] += bool(cluster_docs & set(p.act_document_ids))
            r["pinned_correct"] += int(store.class_id[uid]) in {int(store.class_id[e]) for e in exact}
            r["false_abstain"] += reason is not None
            r[f"reason:{reason}"] += 1
            if reason is not None or not (cluster_docs & set(p.act_document_ids)):
                failures[f"{vname}/answerable"].append({"q": q, "reason": reason, "resolved": p.act_document_ids[:3]})
            # nonexistent section of the same Act
            q = tmpl.format(n=fake, t=title)
            p, exact, reason = check(q)
            r = results[vname]["fake_section"]
            r["n"] += 1
            r["abstain"] += reason is not None
            r[f"reason:{reason}"] += 1
            if reason is None:
                failures[f"{vname}/fake_section"].append({"q": q, "resolved": p.act_document_ids[:3], "exact": len(exact)})
            # state-swapped title (only for titles that start with their own state's name)
            jur = str(docs.at[did, "jurisdiction"])
            if title.lower().startswith(jur.lower() + " "):
                other = rng.choice([s for s in states if s != jur])
                swapped = other + title[len(jur):]
                if _strip_title(swapped)[0] not in tidx:
                    q = tmpl.format(n=num, t=swapped)
                    p, exact, reason = check(q)
                    r = results[vname]["swapped_state"]
                    r["n"] += 1
                    r["abstain"] += reason is not None
                    r[f"reason:{reason}"] += 1
                    if reason is None:
                        failures[f"{vname}/swapped_state"].append({"q": q, "resolved": p.act_document_ids[:3],
                                                                   "exact": len(exact)})
    for f in FOREIGN:
        for tmpl in TEMPLATES[:2]:
            q = tmpl.format(n=rng.randint(3, 60), t=f).replace("the the ", "the ").replace("the Singapore", "Singapore")
            p, exact, reason = check(q)
            r = results["foreign"]["foreign"]
            r["n"] += 1
            r["abstain"] += reason is not None
            r[f"reason:{reason}"] += 1
            if reason is None:
                failures["foreign"].append({"q": q, "resolved": p.act_document_ids[:3], "exact": len(exact)})

    def rate(c, k):
        return round(c[k] / c["n"], 3) if c["n"] else None

    summary = {"dev_in_sample": {k: {"n": c["n"], "abstain_rate": rate(c, "abstain"),
                                     "reasons": {r: v for r, v in c.items() if r.startswith("reason:")}}
                                 for k, c in dev.items()},
               "probe_acts": used, "pool": int(len(cand)), "excluded_dev_docs": len(dev_docs), "excluded_dev_clusters": len(excl)}
    for vname, d in results.items():
        for kind, c in d.items():
            s = {"n": c["n"]}
            for k in ("act_resolved", "pinned_correct", "false_abstain", "abstain"):
                if k in c or (kind == "answerable" and k != "abstain") or (kind != "answerable" and k == "abstain"):
                    s[k] = rate(c, k)
            s["reasons"] = {r: v for r, v in c.items() if r.startswith("reason:")}
            summary[f"{vname}/{kind}"] = s
    print(json.dumps(summary, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe_failures.json").write_text(json.dumps({k: v[:40] for k, v in failures.items()}, indent=1,
                                                        ensure_ascii=False), encoding="utf-8")
    return summary


# ----------------------------------------------------------------------------- sensitivity
def _set(cfg, path, value):
    d = cfg
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value


VARIANTS = [
    ("fusion_k=30", [("fusion", "k")], 30), ("fusion_k=120", [("fusion", "k")], 120),
    ("bm25_w=0.5", [("retrieve", "bm25", "weight")], 0.5), ("bm25_w=0", [("retrieve", "bm25", "weight")], 0.0),
    ("dense_w=2", [("retrieve", "dense", "weight")], 2.0),
    ("exact_w=1", [("metadata", "exact_ref", "weight")], 1.0), ("exact_w=4", [("metadata", "exact_ref", "weight")], 4.0),
    ("act_w=0.5", [("metadata", "act_bm25", "weight")], 0.5), ("act_w=2", [("metadata", "act_bm25", "weight")], 2.0),
    ("jur_w=0.25", [("metadata", "jur_bm25", "weight")], 0.25), ("jur_w=1", [("metadata", "jur_bm25", "weight")], 1.0),
    ("depth=50", [("rerank", "depth")], 50), ("depth=200", [("rerank", "depth")], 200),
    ("min_piece=100", [("pack", "min_piece")], 100), ("min_piece=400", [("pack", "min_piece")], 400),
    ("list_k=100", [("retrieve", "bm25", "k"), ("retrieve", "dense", "k")], 100),
]


def sensitivity(config_path: Path) -> dict:
    import contextlib
    import io

    import yaml

    from isr.eval import evaluate as ev
    from isr.eval.predictions import write_predictions
    from isr.pipeline import Pipeline

    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pipe = Pipeline(base)
    qs = questions_for("dev")
    out = {}
    for name, paths, value in [("FROZEN", [], None)] + VARIANTS:
        cfg = copy.deepcopy(base)
        for path in paths:
            _set(cfg, path, value)
        cfg["name"] = f"audit-sens-{name}"
        pipe.config = cfg
        run = {q["question_id"]: pipe.run(q["question"]) for q in qs}
        pred = write_predictions(OUT / "sensitivity" / name / "pred_dev.json", system=cfg["name"], split="dev",
                                 run=run, meta={"config": cfg})
        with contextlib.redirect_stdout(io.StringIO()):
            ev.main([str(pred), "--split", "dev"])
        rep = json.loads(pred.with_suffix(".report.json").read_text(encoding="utf-8"))["official"]
        by = rep["retrieval"]["by_category"]
        row = {f"B{b}": {c: round(by[c][str(b)]["evidence_recall"], 3) for c in ANSWERABLE_CATS} for b in (2000, 8000)}
        for b in (2000, 8000):
            row[f"macro_B{b}"] = round(float(np.mean(list(row[f"B{b}"].values()))), 3)
        row["mrr_macro_B8000"] = round(float(np.mean([by[c]["8000"]["mrr"] for c in ANSWERABLE_CATS])), 3)
        abst = rep.get("abstention") or {}
        row["abstention"] = abst
        out[name] = row
        print(f"{name:>15}  macro@2k {row['macro_B2000']:.3f}  macro@8k {row['macro_B8000']:.3f}  "
              f"mrr {row['mrr_macro_B8000']:.3f}  " + " ".join(f"{c[:4]}={v:.2f}" for c, v in row["B8000"].items()),
              flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("threshold-cv")
    a.add_argument("trace", type=Path)
    b = sub.add_parser("probe")
    b.add_argument("--n", type=int, default=400)
    c = sub.add_parser("sensitivity")
    c.add_argument("config", type=Path)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "threshold-cv":
        res = threshold_cv(args.trace)
    elif args.cmd == "probe":
        res = probe(args.n)
    else:
        res = sensitivity(args.config)
    (OUT / f"{args.cmd}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
