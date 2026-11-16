# Project Plan: Structure-Aware, LLM-Assisted RAG for Indian Statutory Law (IndiaStatRAG)

> Status: **Phase 0 complete (2026-11-16)** — environment, pinned kit, eval wrapper, baselines reproduced
> exactly (see README.md). Next: Phase 1. This document is the single source of truth for
> what we build, in what order, and how we evaluate it. Update the "Decision log" when anything changes.

---

## 0. TL;DR

- **Benchmark:** IndiaStatRAG (`Advaith-Srinivas05/Indian-legal-benchmark`, commit `b45befa`).
  It is a **span-retrieval benchmark**: for each question the system returns ranked
  `(document_id, char_start, char_end)` spans within a character budget (B = 2,000 / **8,000** headline / 32,000).
  Answer metrics (fact coverage, citations, quote grounding) are optional, add-on.
- **Corpus:** `./corpus` — 15,325 India Code instruments (839 central Acts, 8,797 state Acts, 4,438 rules,
  1,251 regulations), 712.8M characters, 363,539 parsed provisions with character offsets.
- **Chosen approach (locked):** staged **A → B → C**, built as an ablation study.
  - **A** Structure-aware hybrid retrieval (provision chunks + BM25 + dense + cross-encoder + metadata + budget packer + abstention).
  - **B** Local-LLM query understanding (query analysis, HyDE, LLM rerank, abstention judge) + cited answer generation.
  - **C** Act-first routing + cross-reference graph + jurisdiction-equivalent mapping.
- **Constraints (locked):** local models only, no paid or free APIs, budget 0.
  Hardware: RTX 4060 Laptop (8 GB VRAM), Ryzen 7 7840HS, 15 GB RAM, ~102 GB free disk.
- **Primary target:** beat the shipped BM25 baselines in every category, with the largest gains on
  `situational`, `jurisdictional`, `cross_reference` and `unanswerable` (where BM25 is weakest), reported with
  confidence intervals.

---

## 1. Goal, research questions, and what we can / cannot claim

### 1.1 Goal
Build a RAG system over Indian statutory law that retrieves the *exact provision(s)* answering a natural-language
legal question, abstains when the law does not answer it, and generates a grounded, cited answer, while
running entirely on a consumer laptop with open models.

### 1.2 Research questions
- **RQ1.** How much does aligning retrieval units to legal structure (provisions) improve evidence recall *and
  precision* over structure-agnostic windows at fixed character budgets?
- **RQ2.** Does a small local LLM (≤8B, 4-bit) used for *query understanding* (Act/section/jurisdiction extraction,
  candidate-statute guessing, HyDE) close the gap on `situational` questions, which never name the Act?
- **RQ3.** Do Act-level routing and cross-reference expansion improve multi-hop (`all_groups`) recall on
  `cross_reference` questions and the correct-copy problem on `jurisdictional` questions?
- **RQ4.** Can calibrated abstention (deterministic existence checks + reranker thresholds + LLM judge) score on
  the 50 unanswerable questions without unacceptable false abstention on answerable ones?
- **RQ5.** How faithful are locally-generated answers (fact coverage, citation precision/recall, quote grounding)?

### 1.3 Claims we can make vs. cannot make (be honest in the paper)
- **CAN:** compare against BM25 baselines (shipped), and against standard retrievers we run ourselves on the same
  scorer (BM25, several dense embedders, hybrid, cross-encoder, InLegalBERT as a domain-encoder baseline).
- **CANNOT:** claim "comparable to current SOTA models in Indian legal AI" by citing external numbers. No other
  system has been evaluated on this benchmark, and existing Indian legal benchmarks (IL-TUR, AILA statute
  retrieval, LeSICiN, etc.) are on case law / different tasks with different metrics. Comparability is
  established **only by running those model families ourselves on this benchmark**.
- **CANNOT (yet):** claim the benchmark itself is human-validated. Questions were LLM-drafted and LLM-verified.
  Per decision D5, we assume validity for now; this must be listed as a limitation and revisited later.

---

## 2. Decision log

| # | Decision | Status | Notes |
|---|---|---|---|
| D1 | Approach = A → B → C staged, ablation framing | **Locked** | D (fine-tuning) and E (agentic) out of scope; E optional only if time remains. |
| D2 | LLMs: local only (Ollama), **no APIs at all**, budget 0 | **Locked** | Also avoids leaking the benchmark's test set to providers. |
| D3 | Scope = retrieval + generated cited answers | **Locked** | Both metric families reported. |
| D4 | Timeline: performance first; if a deadline is forced, 1 week | **Locked** | 1 week covers A + core B + minimal C only; see §10. |
| D5 | Benchmark validity: assume valid for now | **Locked (deferred)** | Revisit before submission; it is a reviewer risk. |
| D6 | Tune only on `dev` (150 Q); `test` (352 Q) touched only for frozen final runs | **Locked** | Every test run logged in `TEST_RUNS.md`. |
| D7 | Use `corpus/structure/` provision offsets for chunking | **Locked** | Legitimate (README: "optional aid"), but **disclosed** in the paper. |
| D8 | Primary local LLM | Open → pick on dev | Re-checked 2026-11-16: Qwen3-8B (Q4_K_M) primary, Llama-3.1-8B-Instruct (Q4_K_M) secondary; both pulled in Phase 0. Gemma dropped from the default shortlist. |
| D9 | Primary embedder / reranker | Open → pick on dev proxy | Re-checked 2026-11-16: add EmbeddingGemma-300M and Jina-v5-text-small (CC BY-NC 4.0: OK for research, disclose) to the §6 Phase 2 shortlist. |
| D11 | No Claude co-author / AI attribution in commits or PRs | **Locked** | User instruction. |
| D10 | LLM Act summaries for Phase C (≈15–20 GPU-hours for all 15k docs) | Open → decide after a 1k-doc trial | Cheap "card" version without LLM summaries is the default. |

---

## 3. What we are working with (verified facts)

### 3.1 Benchmark (IndiaStatRAG)
- 502 verified questions (8 rejected are excluded by the loader). Split `split-s20260923.json`: dev 150 / test 352.
- Per-category (dev / test):

  | category | dev | test | what it tests |
  |---|---:|---:|---|
  | situational | 40 | 93 | fact patterns that never name the Act |
  | provision_lookup | 23 | 53 | named Act + provision |
  | cross_reference | 21 | 51 | multi-hop; 144 `required` groups across the set |
  | jurisdictional | 21 | 50 | a specific state's version of the law |
  | numeric_threshold | 15 | 36 | penalties, limits, periods |
  | definitional | 15 | 34 | definitions clauses |
  | unanswerable | 15 | 35 | correct answer = return `[]` |

- Gold = groups of interchangeable locations (any copy of the same Act counts). `sufficient` = hit one group;
  `required` = must hit all groups (`all_groups` metric).
- Gold span length: median 751 chars, p90 2,642, max 5,865 → at B=2,000 even the oracle scores 0.83–1.00.
- **Scoring:** hit = coverage ≥ τ (0.5 primary; 0.8, 1.0 also) of a gold location over the union of returned
  spans in that document. Overlapping spans are merged before budgeting. Reported per category: evidence recall,
  evidence precision, all-groups, document recall, MRR. **Abstention** = returning an empty list at the headline
  budget (8,000); scored on unanswerable Qs beside false abstention on answerable Qs.
- **Answers** (optional) go in the same predictions file:
  `"answers": {qid: {"text", "abstained", "citations": [{"document_id","key"}]}}`.
  Quote grounding counts only quotes ≥ 12 chars in `"..."` or `“...”`.
- Baselines to beat (test, B=8,000, evidence recall):

  | category | oracle | bm25-windows | bm25-two-stage |
  |---|---:|---:|---:|
  | provision_lookup | 1.000 | 0.491 | 0.528 |
  | definitional | 1.000 | 0.382 | 0.294 |
  | numeric_threshold | 1.000 | 0.417 | 0.250 |
  | cross_reference | 1.000 | 0.235 | 0.196 |
  | jurisdictional | 1.000 | 0.100 | 0.120 |
  | situational | 1.000 | 0.075 | 0.065 |

  BM25 precision 0.009–0.055; BM25 never abstains (0.000 on unanswerable).

### 3.2 Corpus (`./corpus`)
- `documents.jsonl` (15,325 rows): `document_id, title, category, document_type, jurisdiction, year, char_count, pages, provisions, structure_confidence, ocr_pages, text_sha256`.
- `text/<document_id>.txt`: canonical UTF-8 text. **All offsets index this file** (Python `str` indices).
- `meta/<document_id>.json`: title, short/long title, jurisdiction, act number/year, dates, ministry/department, `parent`, India Code source URLs.
- `structure/<document_id>.json`: `preamble`, `units` (tree: part/chapter/section/subsection/clause/…, with offsets & headings), `provisions` (flat list with `key`, `unit_path`, `heading`, `char_start/end`, `evidence_confidence`, `text_sha256`, `key_ambiguous`).
  Confidence: 177k high / 172k medium / 14k low. 3,373 docs have OCR pages; 704 docs miss page fields.
- `duplicates.json`: 891 clusters covering 2,140 docs (identical / same_instrument / parallel / contains edges).
- `provision_equivalents.jsonl`: 30,969 classes, 82,893 provisions with identical normalised text (incl. cross-instrument, e.g. state re-enactments).
- `CHECKSUMS.txt`: hash that must be quoted in every reported result.

### 3.3 Environment
- Python 3.12.4, global env has torch 2.13+cu130 (CUDA OK), sentence-transformers 5.1.2, transformers 4.57.1,
  rank-bm25, chromadb, llama-index, langchain (we will **not** depend on llama-index/langchain; see §5).
  `torchaudio` is a CPU build (mismatch, harmless but a sign the global env is messy → use a fresh venv).
- No `faiss` installed. Ollama installed with only `mistral`, `nomic-embed-text`, `mxbai-embed-large` pulled.
- Project folder is **not a git repo** yet. Disk is 90% full (~102 GB free).

---

## 4. Risks, pitfalls, and things that are easy to get wrong

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Overfitting dev** | Dev has 15–40 Qs per category; 15 unanswerable. Tiny. | Keep rules simple, few tuned knobs; report bootstrap CIs; never pick configs by test. |
| **Test leakage** | Looking at test gold to debug = tuning on test. | Debug/error analysis on dev only; test runs only for frozen configs, logged. Scorer flag `--tuned-on-test` stays off honestly. |
| **Few-shot leakage** | Using dev/test questions as prompt examples inflates scores. | Hand-write few-shot examples from corpus provisions **not** in any gold location. |
| **Duplicate copies eat the budget** | Same section in 5 copies = 5× budget for one hit, precision collapses. | Dedup at index time via equivalence classes; pack one copy per class (jurisdiction-aware choice). |
| **Wrong copy for jurisdictional Qs** | Central Act hit when a state's version is gold. | Jurisdiction parsing + equivalence-class swap to the named state's copy (Phase C). |
| **Long provisions at B=2,000** | p90 gold is 2.6k chars. | Sub-unit splitting along subsections/clauses; pack the best sub-unit. Compare to oracle at B=2,000, not to 1.0. |
| **Abstention trade-off** | Abstain-all scores 1.0 abstention and 1.0 false abstention. | Optimise a joint objective on dev (see Phase 3.4); always report both numbers. |
| **Ollama silent context truncation** | Ollama's default `num_ctx` is small; long prompts get cut without error. | Always set `num_ctx` explicitly (≥ 8192) and assert prompt token counts. |
| **8 GB VRAM contention** | LLM (~5–6 GB) + reranker + embedder can't co-reside comfortably. | Stage-wise batch pipeline: each stage runs over all questions, caches to disk, frees GPU. |
| **15 GB RAM** | Dense index + BM25 + unit table can exceed RAM. | fp16 vectors, memory-mapped arrays, parquet with column projection; IVF/HNSW if flat index too large. |
| **Disk 90% full** | Embedding shards, indexes, model weights (~20–30 GB total). | Budget in §9; clean intermediate shards; consider moving `indexes/` to another drive if available. |
| **LLM hallucinated Acts** | 8B models invent statute names/sections. | Every LLM-proposed Act/section is resolved against the corpus title/structure gazetteer; unresolved ones are dropped. |
| **Reproducibility of LLM stages** | Sampling, model updates. | temperature 0, fixed seed, pin Ollama model digest, disk cache of every call keyed by (model digest, prompt version, input). |
| **Benchmark validity (deferred)** | LLM-drafted + LLM-verified questions; author = system builder. | D5: assume valid now; list as limitation; plan a human audit later. |
| **Generator bias** | If the question-drafting LLM's family is the same as our LLM, results may be optimistic. | Record which model drafted the questions (check benchmark authoring notes) and disclose. |

---

## 5. System architecture

Design principle: **stage-wise, cached, batch pipeline** (not a per-question online chain). Each stage reads the
previous stage's outputs from disk for *all* questions of a split, writes its own outputs, then releases the GPU.
This is what makes 8 GB VRAM workable and every ablation cheap to re-run.

```
                       ┌──────────────── offline (once) ────────────────┐
corpus/ ──► Phase 1 unit store (provisions + sub-units + gap windows, dedup classes, doc cards)
                 ├─► BM25 index (units)          ├─► dense index (units, fp16 + FAISS)
                 ├─► Act-card index (Phase C)    └─► cross-ref graph (Phase C)
                       └────────────────────────────────────────────────┘

question ─► [Q-parse rules] ─► [B1 LLM query analysis] ─► [B2 HyDE + rewrites]
        ─► candidate generation: BM25 ∪ dense ∪ title/section match ∪ (C) Act-routed ∪ (C) xref expansion
        ─► RRF fusion ─► cross-encoder rerank (top-100) ─► (B3) LLM rerank (top-20)
        ─► abstention decision (existence check + score threshold + B4 LLM judge)
        ─► budget packer (dedup classes, jurisdiction copy choice, sub-unit trimming) ─► spans @2k/8k/32k
        ─► (B5) answer generation with citations + verbatim quotes ─► predictions.json ─► official scorer
```

Libraries: plain Python + `sentence-transformers`, `faiss-cpu`, `bm25s` (or SQLite FTS5, as the baseline does),
`pyarrow`/`pandas`, `rapidfuzz`, `ollama` client, `pytest`. **No llama-index / langchain**: they add opaque
chunking/prompt layers that make offsets and ablations harder to control.

### 5.1 Repository layout (to create in Phase 0)
```
Claude/
  plan.md                 ← this file
  README.md               ← how to reproduce
  TEST_RUNS.md            ← log of every test-split evaluation
  corpus/                 ← existing (git-ignored)
  benchmark_kit/          ← output of `python -m benchmark export` (scorer, questions, split) — pinned commit
  configs/                ← YAML per system/ablation (A0, A1, …, ABC_full)
  prompts/                ← versioned prompt templates (query_analysis_v1.txt, hyde_v1.txt, …)
  src/isr/                ← our package
    corpus/  (load, units, dedup, cards, xref)
    index/   (bm25, dense, cards)
    query/   (rules parser, llm analysis, hyde)
    retrieve/(fusion, rerank, llm_rerank, routing, expansion)
    decide/  (abstention)
    pack/    (budget packer)
    answer/  (generation, citation validation)
    llm/     (ollama client, cache)
    eval/    (run split, score wrapper, bootstrap CIs, significance, error analysis)
  scripts/                ← CLI entry points (build_units, build_bm25, embed, run_stage, score_dev, …)
  tests/
  data/                   ← git-ignored: units.parquet, docs.parquet, graph, caches
  indexes/                ← git-ignored: bm25, faiss, card index
  runs/                   ← git-ignored except final reports: predictions + reports per config
```

---

## 6. Phased implementation plan

Each phase ends with a **complete, scorable system** on dev, so stopping early still yields results.

### Phase 0 — Setup & sanity (Day 1, ~half day)
1. `git init` in `Claude/`; `.gitignore`: `corpus/`, `data/`, `indexes/`, `runs/**/cache`, `*.npy`, model files.
2. Fresh venv (`python -m venv .venv`), pinned `requirements.txt`: torch (CUDA wheel matching driver), sentence-transformers,
   transformers, faiss-cpu, bm25s + PyStemmer, pyarrow, pandas, numpy, rapidfuzz, ollama, pyyaml, orjson, tqdm, pytest.
3. Clone benchmark at commit `b45befa`, run `python -m benchmark export --out ./benchmark_kit`; record commit + `CHECKSUMS.txt` sha256 in `README.md`.
4. Ollama: pull the LLM shortlist (D8) at Q4_K_M; record model digests; verify full GPU offload and measured tokens/s
   (prompt processing and generation) with `num_ctx` 8192.
5. **Sanity run:** score `oracle` (must be 1.0 at B=8k/32k), `bm25-windows`, `bm25-two-stage` on dev and test using
   the shipped baselines (`python -m benchmark index` ≈ 25 min, 693 MB). Must reproduce the README table.
   This proves our corpus copy matches the benchmark's.
6. Write `src/isr/eval/score_dev.py`: runs the official scorer on a predictions file, keeps per-question rows,
   computes **bootstrap 95% CIs (1,000 resamples) per category**, and a paired bootstrap test vs. a reference run.

**Done when:** README baseline numbers reproduced; eval wrapper prints per-category recall/precision ± CI.

### Phase 1 — Corpus processing → unit store (Day 1–2)
1. Load `documents.jsonl` + `meta/` → `docs.parquet` (title, short/long title, jurisdiction, category, type, year, parent, act ids, structure_confidence).
2. Provision units from `structure/*.json` `provisions` → `units.parquet` with `unit_id, document_id, key, unit_path, heading, char_start, char_end, evidence_confidence, key_ambiguous, text_sha256`.
3. **Sub-units:** provisions > ~2,000 chars are split along their `units` tree children (subsection/clause); keep `parent_unit_id`. Needed for B=2,000 packing.
4. **Gap windows:** text not covered by any provision (preamble, schedules, forms, tables, docs with `structure_confidence` ≠ structured, low-confidence regions) → 1,200-char windows / 900 stride, `unit_type = window`. Guarantees 100% text coverage.
5. **Embedding text** = context header + body: `"{short_title} [{jurisdiction}, {year}, {document_type}] — {unit path} {heading}\n{body}"`, truncated to the embedder's max length. Spans returned are always the raw body offsets.
6. **Dedup:** map every unit to an equivalence class (from `provision_equivalents.jsonl`, plus exact `text_sha256` / normalised-text hash). Index one representative per class; keep the full member list (doc, key, offsets, jurisdiction) for the packer.
7. **Doc cards** (cheap, for Phase C): title + short/long title + jurisdiction + year + type + parent + list of section headings (truncated to ~2k chars).
8. **Alignment check (dev only):** % of dev gold locations covered ≥ 0.5 by a single unit, and ≥ 0.5 by one unit **or** its sub-units. Target ≥ 95%; investigate misses (they reveal parsing problems).
9. Unit tests: offsets round-trip (`text[s:e]` sha256 equals `text_sha256` where `text_matches_span`), no negative/backwards spans, dedup members share normalised text.

**Done when:** `units.parquet` built, stats logged (#units, #classes, #windows, length distribution), alignment ≥ 95% on dev.

### Phase 2 — Plan A: structure-aware hybrid retrieval (Day 2–3)
1. **Sparse:** BM25 over unit text with stemming; legal-aware tokenisation (keep `2(1)(a)`, years, numbers, "Rs."); separate title/heading field fused or boosted. Also re-run as `A0 = BM25-provision-units` to isolate RQ1.
2. **Dense model selection (cheap proxy, dev only):** build a mini-corpus = all units of dev gold documents + their duplicate clusters + ~30k random distractor units; evaluate unit recall@10/50 on dev queries for the shortlist:
   - `BAAI/bge-m3` (dense), `Qwen/Qwen3-Embedding-0.6B`, `google/embeddinggemma-300m`, `jinaai/jina-embeddings-v5-text-small` (CC BY-NC), `intfloat/multilingual-e5-large-instruct`,
     `nomic-embed-text` / `mxbai-embed-large` (already in Ollama), `law-ai/InLegalBERT` (mean-pooled; domain baseline, expected weak as it's not a sentence encoder).
   Pick the winner (and keep one runner-up for the paper's baseline table if time allows).
3. **Full embedding run** of the winner over all representative units: fp16, length-sorted batches, checkpointed shards, resumable. Expect several hours on the 4060 → run overnight. FAISS flat IP (fp16/fp32, memory-mapped) or HNSW if RAM is tight.
4. **Rule-based query parsing (no LLM):** state/UT gazetteer (from `meta` jurisdictions + common abbreviations), Act-title fuzzy match (rapidfuzz on titles/short titles, year-aware), section/rule/schedule regex, doc-type hints ("rules", "regulations", "notification"). Used as **boosts**, hard filters only on very high-confidence matches (prevents cascade failures).
5. **Fusion:** weighted RRF (k≈60) over BM25, dense, and title/section matches; weights tuned on dev.
6. **Rerank:** cross-encoder on top-100 (`BAAI/bge-reranker-v2-m3` vs `Qwen/Qwen3-Reranker-0.6B`, choose on dev).
7. **Budget packer** (directly drives the metric):
   - walk ranked units; skip units whose equivalence class is already packed;
   - choose the class member whose jurisdiction matches the parsed jurisdiction, else the highest-scoring/central copy;
   - if a unit doesn't fit the remaining budget, use its best-scoring sub-unit (rerank children) else truncate from start;
   - merge adjacent/overlapping spans; output best-first (MRR).
   - Two variants tuned on dev: **fill** (use the whole budget, recall-oriented) and **cutoff** (stop below a score threshold, precision-oriented). Report both; pick one as primary.
8. **Abstention v0:**
   - deterministic **existence check**: if the question names Act X + provision N, X resolves with high confidence, and N exists in **no** copy of X → abstain;
   - else abstain if top reranker score < t (t tuned on dev).
9. **Answers v0 (no LLM):** none; Phase A is retrieval-only.

**Ablation rows produced:** A0 (BM25 on provisions), A1 (+dense/hybrid), A2 (+rerank), A3 (+metadata), A4 (+packer dedup/jurisdiction), A5 (+abstention).
**Done when:** A5 ≥ best BM25 baseline in every category on dev at B=8,000, with CIs logged.

### Phase 3 — Plan B: local LLM layer (Day 4–5)
0. **LLM infra:** `src/isr/llm/` Ollama client — temperature 0, fixed seed, explicit `num_ctx`, JSON-schema structured output with retry-on-invalid, SQLite cache keyed by (model digest, prompt version, input hash), token/latency logging. Qwen3: thinking **off** by default; thinking-on is a separate ablation on `situational` only (it's slow).
1. **B1 Query analysis** → JSON `{acts_mentioned[], sections[], jurisdiction, doc_type, legal_issue, candidate_acts[], keywords[], is_named_provision_lookup}`. `candidate_acts` = the LLM's guess of which Indian statutes govern the fact pattern (the key lever for `situational`). Every Act is resolved against the title gazetteer; unresolved → dropped.
2. **B2 Query expansion:** (a) HyDE: "write the statutory provision that would answer this, in Indian legislative drafting style" → dense query; (b) keyword rewrite → BM25; (c) resolved candidate Acts → document boosts. All fused via RRF into Phase A candidates.
3. **B3 LLM rerank** of top-20 after the cross-encoder: pointwise relevance (yes/no + logprob if available, else 0–3 grade), combined with CE score. Measure gain vs. latency; keep only if it helps on dev.
4. **B4 Abstention judge:** question + top-5 packed provisions → `{answerable, supporting_unit_ids}`. Final rule (tuned on dev, kept simple): abstain if existence-check fires, **or** (judge says unanswerable **and** CE top score < t_hi). Objective on dev: maximise `abstention_rate − λ·false_abstention` with λ fixed in advance (e.g. λ = 2, false abstention is worse than missed abstention because answerable Qs are 9× more numerous).
5. **B5 Answer generation:** prompt with the B=8,000 packed provisions (each tagged with `document_id` + `key` + citation string) → JSON `{text, citations[{document_id,key}], abstained}`. Instructions: answer only from provided text; quote decisive phrases verbatim in double quotes (≥ 12 chars, for quote grounding); cite every provision used. Post-processing: drop citations not in the retrieved set; verify quotes are substrings (normalised) of cited text, else remove the quote marks; if abstention decided → `abstained: true`, empty text.
6. **Few-shot examples:** 3–5 hand-written from corpus provisions that are not gold for any question.
7. **Throughput estimate:** ~5 LLM calls/question, ~20–40 s/question with an 8B Q4 model → dev ≈ 1–1.5 h, test ≈ 2.5–4 h per full config. Caching makes ablations that reuse stages near-free.

**Ablation rows:** B1 (+query analysis), B2 (+HyDE), B3 (+LLM rerank), B4 (+LLM abstention), B5 (+answers), and the D8 LLM comparison (2 models max on dev).
**Done when:** A+B > A on dev for situational / jurisdictional / unanswerable (CIs reported); answer metrics computed.

### Phase 4 — Plan C: Act routing + cross-reference graph + jurisdiction mapping (Day 6; extended later)
1. **Act index:** BM25 + dense over doc cards (Phase 1.7). Optional (D10): LLM 3-sentence "what this Act regulates" summaries; trial on ~1k docs (incl. dev gold docs + random) first; full run (~15–20 GPU-hours) only if the trial shows dev gains.
2. **Act router:** top-k Acts from card retrieval ∪ resolved B1 `candidate_acts` → provision retrieval restricted to those Acts → **merged** (RRF) with the global list. Never replace the global list (routing errors would cascade).
3. **Cross-reference graph:** regex extraction per unit: `section N`, `sub-section (x) of section N`, `clause`, `Schedule`, `rule N`, `the <Title> Act, YYYY`; resolve to units in the same document or to other documents via title match; link rules/regulations → parent Act via `meta.parent`. Store as an edge list.
4. **Expansion:** for the top-m units, add referenced units (and definitions sections of the same Act when the question has a defined term) with a decayed score; targets `all_groups` on `cross_reference`.
5. **Jurisdiction mapping:** when a state is named and the best hit is a central/other-state copy, swap to that state's member of the cross-instrument equivalence class (`provision_equivalents.jsonl`) or the same key in that state's Act with matching title.

**Ablation rows:** C1 (+Act routing), C2 (+LLM summaries, if D10 = yes), C3 (+xref expansion), C4 (+jurisdiction mapping) = **full system**.
**Done when:** gains on dev `cross_reference` all-groups and `jurisdictional` recall, no regression elsewhere beyond CI noise.

### Phase 5 — Evaluation, test runs, analysis (Day 7)
1. Freeze configs (`configs/final_*.yaml`), tag git commit.
2. Test runs (each logged in `TEST_RUNS.md` with date, commit, config, model digests): BM25 baselines (shipped), best dense-only, A-full, A+B-full, A+B+C-full. **No changes after seeing test numbers** except to fix crashes (and if that happens, disclose).
3. Report per category × budget (2k/8k/32k) × τ (0.5 primary, 0.8/1.0 appendix): recall, precision, all-groups, doc recall, MRR, abstention / false abstention, answer metrics.
4. Statistics: bootstrap 95% CIs; paired bootstrap (or approximate randomisation) vs. best BM25 and vs. previous stage.
5. Error analysis (dev for diagnosis; test only descriptively after final runs). Taxonomy: wrong Act; right Act wrong provision; right provision wrong copy/jurisdiction; partial coverage (<τ); budget wasted on duplicates; false abstention; missed abstention; hallucinated citation.
6. Efficiency table: index build time, disk, RAM/VRAM peak, per-question latency per stage.

### Phase 6 — Optional upper bound (only if time remains)
- **E (agentic, local):** 8B model with tools `search(query)`, `open_toc(doc)`, `read(doc,key)` on dev only, max N steps. Reported as an exploratory upper-bound attempt, not the main system.

### Phase 7 — Write-up & reproducibility
- Paper sections: task + benchmark summary, system, ablations (RQ1–RQ5), results with CIs, error analysis, efficiency, limitations (D5 validity, LLM-drafted questions, structure-aid disclosure, local-model ceiling, small dev set).
- Every result states the six required fields: corpus version + `CHECKSUMS.txt` hash; split; budget; τ; tuned-on-test (no); judge model/protocol (n/a unless answers are judged by an LLM; the built-in answer metrics are lexical).
- Release: code, configs, prompts, model digests, cached LLM outputs, predictions files, reports.

---

## 7. Experiment matrix (all on dev first; final rows on test)

| ID | System | Answers RQ |
|---|---|---|
| BL1 / BL2 | bm25-windows / bm25-two-stage (shipped) | reference |
| A0 | BM25 over provision units | RQ1 |
| A1-x | dense-only per embedder (incl. InLegalBERT) | baseline families |
| A1 | hybrid BM25 + best dense (RRF) | RQ1 |
| A2 | + cross-encoder rerank | — |
| A3 | + rule-based metadata (Act/section/state) | RQ3 (partial) |
| A4 | + dedup-aware, jurisdiction-aware packer | RQ1 |
| A5 | + abstention v0 | RQ4 |
| B1–B4 | + LLM query analysis / HyDE / LLM rerank / LLM abstention | RQ2, RQ4 |
| B5 | + cited answer generation | RQ5 |
| B-llm | B-full with LLM #2 | robustness |
| C1–C4 | + Act routing / summaries / xref expansion / jurisdiction mapping | RQ2, RQ3 |
| E | agentic upper bound (optional) | discussion |

---

## 8. Success criteria
- **Minimum:** full system beats the best BM25 baseline in every answerable category at B=8,000 on test, with
  non-overlapping or paired-significant CIs on situational, jurisdictional and cross_reference; abstention > 0
  with false abstention ≤ ~5%.
- **Good:** clear per-component gains in the ablation (each stage justified), precision improved by an order of
  magnitude over BM25 at equal recall, answer citation precision high (hallucinated citations ≈ 0 by construction).
- These are targets, not promises; small categories (dev 15 Qs) make some differences statistically unresolvable,
  and we will say so rather than over-claim.

---

## 9. Compute & storage budget (estimates; verify in Phase 0)

| Item | Estimate |
|---|---|
| Unit store (parquet) | ~1–2 GB |
| BM25 index | ~1 GB (shipped baseline index: 693 MB) |
| Dense vectors (~0.5–0.8M representative units × 1024-d fp16) | ~1–1.6 GB + FAISS overhead |
| Full embedding run (~180M tokens, truncated to 512/unit) | several hours on RTX 4060 Laptop (overnight) |
| Model weights (embedders, rerankers, 2 × 8B Q4 LLMs) | ~15–20 GB |
| LLM cache + runs | ~1–3 GB |
| Optional Act summaries (D10, all 15k docs) | ~15–20 GPU-hours |
| **Total disk** | **~25–30 GB** of the ~102 GB free |

---

## 10. Schedule

**Be clear-eyed:** the full A→B→C plan with proper ablations realistically needs **~3–4 weeks** of work on this laptop.
If forced into **1 week**, this is the critical path, and C is reduced to its cheapest parts:

| Day | Work | GPU running overnight |
|---|---|---|
| 1 | Phase 0 (setup, baselines reproduced, eval wrapper) + Phase 1 steps 1–6 | shipped BM25 index |
| 2 | Phase 1 checks; Phase 2.1–2.2 (BM25 units, embedder proxy selection) | **full dense embedding** |
| 3 | Phase 2.4–2.8 (parsing, fusion, rerank, packer, abstention v0) → A on dev | — |
| 4 | Phase 3.0–3.2 (LLM infra, query analysis, HyDE) on dev | dev LLM runs |
| 5 | Phase 3.3–3.5 (LLM rerank, abstention judge, answers) on dev | dev LLM runs |
| 6 | Phase 4 minimal: card-based Act routing (no LLM summaries) + same-document xref expansion + jurisdiction swap | **test runs for A and A+B** |
| 7 | Freeze; test run A+B+C; reports, CIs, error analysis | — |

**Dropped in the 1-week version:** LLM model comparison (D8 uses one model), LLM Act summaries (D10), second
embedder in the full index, thinking-mode ablation, cross-document xref, optional E. Results will be valid but
the ablation story thinner.

---

## 11. Pre-implementation checklist (everything to have ready before coding)
- [ ] Confirm D8/D9 shortlists are still the best available ≤8B / ≤1B open models at implementation time.
- [ ] Confirm free disk space (≥ 35 GB with margin); decide where `indexes/` lives.
- [ ] `git init`, fresh venv, pinned requirements.
- [ ] Benchmark kit exported at pinned commit; `CHECKSUMS.txt` hash recorded.
- [ ] Shipped baselines reproduced (sanity).
- [ ] `TEST_RUNS.md` created (empty log) — the test split is not touched until Phase 5.
- [ ] Find out which LLM drafted/verified the questions (for the generator-bias disclosure).
- [ ] Revisit D5 (human validation of the benchmark) before any submission.
