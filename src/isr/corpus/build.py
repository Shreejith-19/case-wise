"""Phase 1: build the unit store (data/units/*.parquet) and document table (data/docs.parquet).

    python -m isr.corpus.build [--workers 6]

Pipeline
1. Workers turn batches of documents into units (isr.corpus.units) -> data/units/_shards/.
2. Dedup classes are computed corpus-wide: units are one class when they have the same
   kind and normalised text **and** come from the same duplicate cluster (copies of one
   instrument). Copies of the same text in *different* instruments (state re-enactments,
   boilerplate) stay separate classes — they are different law; Phase C relates them.
   One representative per class (prefer the cluster's representative document) is what
   gets indexed; the packer can swap in any member.
3. Shards are rewritten with global unit_id / parent_id / class_id / is_rep.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from isr import CORPUS_DIR, DATA_DIR
from isr.corpus.units import document_units

UNITS_DIR = DATA_DIR / "units"
SHARD_DIR = UNITS_DIR / "_shards"
DOCS_PATH = DATA_DIR / "docs.parquet"
STATS_PATH = DATA_DIR / "units_stats.json"
BATCH = 200
HEADINGS_MAX = 4000

UNIT_SCHEMA = pa.schema([
    ("document_id", pa.string()), ("kind", pa.string()), ("key", pa.string()),
    ("parent_row", pa.int32()), ("ordinal", pa.int16()),
    ("char_start", pa.int32()), ("char_end", pa.int32()),
    ("label", pa.string()), ("heading", pa.string()), ("region", pa.string()),
    ("evidence_confidence", pa.string()), ("key_ambiguous", pa.bool_()),
    ("norm_hash", pa.string()), ("text", pa.large_string()),
])


def _load(path: Path):
    return orjson.loads(path.read_bytes())


def _handle_of(document_id: str):
    return document_id.split("__handle-", 1)[1] if "__handle-" in document_id else None


def _doc_row(meta: dict, record: dict, structure: dict, units) -> dict:
    q = meta.get("quality") or {}
    parent = meta.get("parent") or {}
    heads = []
    for p in structure.get("provisions") or []:
        h = (p.get("heading") or "").strip()
        num = (p.get("unit_path") or [{}])[-1].get("number", "")
        heads.append(f"{num} {h}".strip())
    headings = "; ".join(heads)
    return {
        "document_id": record["document_id"],
        "title": meta.get("title") or record.get("title") or "",
        "short_title": meta.get("short_title") or "",
        "long_title": meta.get("long_title") or "",
        "hindi_title": meta.get("hindi_title") or "",
        "category": meta.get("category") or record.get("category"),
        "document_type": meta.get("document_type") or record.get("document_type"),
        "jurisdiction": meta.get("jurisdiction") or record.get("jurisdiction"),
        "year": meta.get("year") if meta.get("year") is not None else record.get("year"),
        "act_number": str(meta.get("act_number") or ""),
        "act_year": meta.get("act_year"),
        "enactment_date": str(meta.get("enactment_date") or ""),
        "ministry": meta.get("ministry") or "",
        "department": meta.get("department") or "",
        "parent_title": parent.get("title") or "",
        "parent_handle": (parent.get("handle") or "").rsplit("/", 1)[-1],
        "structure_confidence": record.get("structure_confidence"),
        "extraction_quality_score": q.get("extraction_quality_score"),
        "content_language": q.get("content_language"),
        "n_ocr_pages": len(q.get("ocr_pages") or []),
        "char_count": record.get("char_count"),
        "n_provisions": len(structure.get("provisions") or []),
        "n_units": len(units),
        "headings": headings[:HEADINGS_MAX],
    }


def _work(job):
    """Build one shard. Returns (shard_index, n_rows, doc_rows)."""
    shard_idx, records = job
    cols = {f.name: [] for f in UNIT_SCHEMA}
    doc_rows = []
    row = 0
    for rec in records:
        did = rec["document_id"]
        meta = _load(CORPUS_DIR / "meta" / f"{did}.json")
        structure = _load(CORPUS_DIR / "structure" / f"{did}.json")
        with open(CORPUS_DIR / meta["text"]["path"], encoding="utf-8", newline="") as fh:
            text = fh.read()
        units = document_units(text, structure)
        for u in units:
            cols["document_id"].append(did)
            cols["kind"].append(u.kind)
            cols["key"].append(u.key)
            cols["parent_row"].append(-1)  # filled in below once the document's base row is known
            cols["ordinal"].append(u.ordinal)
            cols["char_start"].append(u.char_start)
            cols["char_end"].append(u.char_end)
            cols["label"].append(u.label)
            cols["heading"].append(u.heading)
            cols["region"].append(u.region)
            cols["evidence_confidence"].append(u.evidence_confidence)
            cols["key_ambiguous"].append(u.key_ambiguous)
            cols["norm_hash"].append(u.norm_hash)
            cols["text"].append(text[u.char_start:u.char_end])
        # parent indices are per-document; convert to shard rows
        base = row
        for i, u in enumerate(units):
            cols["parent_row"][base + i] = -1 if u.parent is None else base + u.parent
        row += len(units)
        doc_rows.append(_doc_row(meta, rec, structure, units))
    pq.write_table(pa.table(cols, schema=UNIT_SCHEMA), SHARD_DIR / f"part-{shard_idx:04d}.parquet",
                   compression="zstd")
    return shard_idx, row, doc_rows


def _clusters() -> tuple[dict[str, str], set[str]]:
    dup = _load(CORPUS_DIR / "duplicates.json")
    cluster_of, reps = {}, set()
    for c in dup["clusters"]:
        reps.add(c["representative"])
        for d in c["documents"]:
            cluster_of[d["document_id"]] = c["cluster_id"]
    return cluster_of, reps


def build(workers: int = 6) -> dict:
    t0 = time.time()
    records = sorted((orjson.loads(l) for l in open(CORPUS_DIR / "documents.jsonl", "rb")),
                     key=lambda r: r["document_id"])
    if UNITS_DIR.exists():
        shutil.rmtree(UNITS_DIR)
    SHARD_DIR.mkdir(parents=True)
    jobs = [(i, records[s:s + BATCH]) for i, s in enumerate(range(0, len(records), BATCH))]

    counts: dict[int, int] = {}
    doc_rows: list[dict] = []
    with mp.Pool(workers) as pool:
        for n, (idx, rows, drows) in enumerate(pool.imap_unordered(_work, jobs), start=1):
            counts[idx] = rows
            doc_rows.extend(drows)
            if n % 10 == 0 or n == len(jobs):
                print(f"  shards {n}/{len(jobs)}  ({time.time() - t0:.0f}s)", flush=True)

    # --- global ids and dedup classes ---
    offsets, acc = {}, 0
    for i in sorted(counts):
        offsets[i] = acc
        acc += counts[i]
    cluster_of, cluster_reps = _clusters()
    keys = []
    for i in sorted(counts):
        t = pq.read_table(SHARD_DIR / f"part-{i:04d}.parquet",
                          columns=["document_id", "kind", "norm_hash"]).to_pandas()
        keys.append(t)
    k = pd.concat(keys, ignore_index=True)
    k["unit_id"] = range(len(k))
    k["group"] = k["document_id"].map(cluster_of).fillna(k["document_id"])
    k["pref"] = (~k["document_id"].isin(cluster_reps)).astype("int8")
    k["class_id"] = k.groupby(["group", "kind", "norm_hash"], sort=False).ngroup().astype("int32")
    order = k.sort_values(["class_id", "pref", "unit_id"])
    rep_ids = order.drop_duplicates("class_id")["unit_id"]
    k["is_rep"] = False
    k.loc[rep_ids.values, "is_rep"] = True
    k["class_size"] = k.groupby("class_id")["unit_id"].transform("size").astype("int32")

    for i in sorted(counts):
        src = SHARD_DIR / f"part-{i:04d}.parquet"
        t = pq.read_table(src)
        lo, hi = offsets[i], offsets[i] + counts[i]
        part = k.iloc[lo:hi]
        parent = t.column("parent_row").to_numpy()
        parent_id = [-1 if p < 0 else int(p) + lo for p in parent]
        t = t.drop(["parent_row"])
        t = t.add_column(0, "unit_id", pa.array(part["unit_id"].to_numpy(), pa.int32()))
        t = t.append_column("parent_id", pa.array(parent_id, pa.int32()))
        t = t.append_column("class_id", pa.array(part["class_id"].to_numpy(), pa.int32()))
        t = t.append_column("class_size", pa.array(part["class_size"].to_numpy(), pa.int32()))
        t = t.append_column("is_rep", pa.array(part["is_rep"].to_numpy(), pa.bool_()))
        pq.write_table(t, UNITS_DIR / f"part-{i:04d}.parquet", compression="zstd")
    shutil.rmtree(SHARD_DIR)

    # --- documents table ---
    docs = pd.DataFrame(doc_rows).sort_values("document_id").reset_index(drop=True)
    for col in ("year", "act_year", "extraction_quality_score"):     # metadata mixes str/int
        docs[col] = pd.to_numeric(docs[col], errors="coerce")
    docs[["year", "act_year"]] = docs[["year", "act_year"]].astype("Int32")
    docs["cluster_id"] = docs["document_id"].map(cluster_of)
    docs["is_cluster_rep"] = docs["document_id"].isin(cluster_reps) | docs["cluster_id"].isna()
    by_handle = {_handle_of(d): d for d in docs["document_id"] if _handle_of(d)}
    docs["parent_document_id"] = docs["parent_handle"].map(lambda h: by_handle.get(h) if h else None)
    docs.to_parquet(DOCS_PATH, compression="zstd", index=False)

    # --- stats ---
    units = pq.read_table(UNITS_DIR, columns=["kind", "char_start", "char_end", "is_rep"]).to_pandas()
    units["len"] = units["char_end"] - units["char_start"]
    stats = {
        "seconds": round(time.time() - t0, 1),
        "documents": len(docs),
        "units": int(len(units)),
        "by_kind": {kind: {"units": int(len(g)), "representatives": int(g["is_rep"].sum()),
                           "chars": int(g["len"].sum()), "rep_chars": int(g.loc[g["is_rep"], "len"].sum()),
                           "len_p50": int(g["len"].median()), "len_p90": int(g["len"].quantile(0.9)),
                           "len_max": int(g["len"].max())}
                    for kind, g in units.groupby("kind")},
        "representatives": int(units["is_rep"].sum()),
        "docs_with_parent_resolved": int(docs["parent_document_id"].notna().sum()),
        "docs_with_parent": int((docs["parent_title"] != "").sum()),
    }
    STATS_PATH.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    stats = build(args.workers)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
