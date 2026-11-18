# isr — Indian Statutory RAG on IndiaStatRAG

Structure-aware, fully local retrieval-augmented generation over Indian statutory law, evaluated on the
IndiaStatRAG benchmark. Runs on one laptop GPU (RTX 4060, 8 GB) with open models only.

- Design, decision log (D1–D16) and every phase's results: [plan.md](plan.md)
- Final test results (frozen, run once): [runs/final/REPORT.md](runs/final/REPORT.md)
- Post-test exploration (not part of the headline): [runs/final/POST_TEST.md](runs/final/POST_TEST.md)
- Evaluation logs: [TEST_RUNS.md](TEST_RUNS.md) (test split), [HOLDOUT_RUNS.md](HOLDOUT_RUNS.md) (dev_holdout)

## Results (test split, 352 questions, B = 8,000 characters, τ = 0.5)

| system | config | macro evidence recall | abstention / false abstention | answers: fact coverage |
|---|---|---:|---|---:|
| bm25-two-stage (shipped baseline) | — | 0.242 | 0 / 0 | — |
| dense only (Qwen3-Embedding-0.6B) | `configs/A1d.yaml` | 0.649 | 0 / 0 | — |
| **Plan A** — structure-aware hybrid retrieval | `configs/A_full.yaml` | 0.747 | 0.771 / 0.032 | — |
| **Plan B** — Plan A + qwen3:8b cited answers | `configs/B_full.yaml` | 0.747 | 0.771 / 0.032 | 0.709 |
| **Plan C** — Plan B + cross-reference expansion (final) | `configs/C_full.yaml` | **0.757** | 0.771 / 0.032 | **0.722** |

All pre-registered minimum success criteria were met (plan.md §8). Frozen commit: tag `phase5-frozen` (7ad684c).

## Pinned evaluation setup

| item | value |
|---|---|
| Benchmark | IndiaStatRAG — `Advaith-Srinivas05/Indian-legal-benchmark` @ `b45befa5fa7f6242dbe9b15cf7b496a0be80921a` |
| Scoring kit | `benchmark_kit/` (from `python -m benchmark export`), scorer version 1 |
| Split | `split-s20260923.json` — dev 150 / test 352; dev further split into dev_tune 100 / dev_holdout 50 (`configs/dev_split.json`) |
| Corpus | `corpus/`, schema v1, 15,325 documents |
| `CHECKSUMS.txt` sha256 | `33b624048b536b14f807178276d61e8efda758d04e9e0406740ca65bd83b69f2` |
| Budgets / τ | 2,000 / **8,000** (headline) / 32,000 chars; τ = 0.5 (0.8, 1.0 reported) |
| Tuned on test | no (all tuning on dev_tune; test run once per frozen system) |

## Environment

| item | value |
|---|---|
| GPU / CPU / RAM | RTX 4060 Laptop 8 GB · Ryzen 7 7840HS · 15 GB · Windows 11 |
| Python / torch | 3.12.4 · torch 2.13.0+cu130 · sentence-transformers 5.7.0 · transformers 4.57.6 (exact: `requirements.lock.txt`) |
| Embedder / reranker | `Qwen/Qwen3-Embedding-0.6B` @ 97b0c614be4d… · `BAAI/bge-reranker-v2-m3` @ 953dc6f6f85a… |
| LLM (Ollama 0.17.7) | `qwen3:8b` Q4_K_M, digest `500a1f067a9f…` (temperature 0, seed 20261117, thinking off) |

## Setup (Windows, Python 3.12)

```bash
python -m venv .venv
.venv/Scripts/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/pip install -r requirements.txt      # exact versions: requirements.lock.txt
.venv/Scripts/pip install -e .
ollama pull qwen3:8b

# only for the shipped BM25 baselines: the full benchmark repo at the pinned commit
git clone https://github.com/Advaith-Srinivas05/Indian-legal-benchmark external/Indian-legal-benchmark
git -C external/Indian-legal-benchmark checkout b45befa5fa7f6242dbe9b15cf7b496a0be80921a
```

## Reproducing everything

Run from the repository root with the venv's Python. Long jobs are **resumable and pausable**: create a file named
`PAUSE` in the repository root (`New-Item PAUSE`) and the job stops at its next checkpoint (exit code 3); delete
it and re-run the same command to continue. LLM replies and reranker scores are cached in `data/cache/`, so
re-runs are exact and fast.

```bash
python -m isr.corpus.build                          # unit store: 884k units + dedup classes      (~1 min)
python -m isr.index.bm25 build                      # BM25 over 784k unit representatives          (~0.5 min)
python -m isr.index.dense build --model qwen3-0.6b  # dense index, fp16                           (~7.7 h, resumable)
python scripts/phase4.py build                      # Act cards + cross-reference graph            (~7 min, pausable)

git checkout phase5-frozen
python scripts/phase5.py status
python scripts/phase5.py run                        # each frozen system once on test, logged      (~45 min, pausable)
python scripts/phase5_analysis.py                   # -> runs/final/REPORT.md
python scripts/post_test_report.py                  # -> runs/final/POST_TEST.md (post-test C-LS)
```

## Development workflow (dev only)

```bash
# retrieval-only systems
python scripts/run_system.py configs/A_full.yaml --split dev_tune --compare runs/A1d-dense-qwen3/pred_dev.json
# systems with LLM stages (staged: LLM -> GPU retrieval -> LLM, so the 8 GB GPU is never shared)
python scripts/run_b.py configs/B_full.yaml --split dev_tune --compare runs/B0-Afull-extractive/pred_dev_tune.json
# answer metrics, paired bootstrap between systems
python scripts/answer_bootstrap.py runs/B0-Afull-extractive runs/B-full --split dev_tune

# official scorer + bootstrap CIs on any predictions file
python -m isr.eval.evaluate runs/X/pred_dev_tune.json --split dev_tune --compare runs/Y/pred_dev_tune.json
```

`dev_holdout` and `test` require `--test-note "<why>"` and are logged to `HOLDOUT_RUNS.md` / `TEST_RUNS.md`.
The test split is spent: any new system can only be reported as post-test.

## Layout

```
plan.md                design, decision log, per-phase results
configs/               one YAML per system (frozen: A_full, B_full, C_full; post-test: C_LS; exploratory: E)
src/isr/
  corpus/              unit store build (provisions, sub-units, gap windows, dedup classes)
  index/               BM25 and dense indexes
  query/               rule-based query parsing (Act titles, sections, States)
  retrieve/            cross-encoder reranker with score cache
  pipeline.py          fusion, metadata lists, rerank, abstention, cross-reference expansion
  pack.py              budget packer
  acts.py, xref.py     Plan C: Act-card index, citation graph
  llm/                 Ollama client with content-addressed cache; LLM stages (analysis, HyDE, rerank, judge, answer)
  agent.py             Phase 6 exploratory agent (dev only)
  pause.py             PAUSE-file checkpoints
  eval/                official-scorer wrapper, bootstrap CIs, logs
scripts/               CLI entry points (see above)
tests/                 pytest (37 tests)
benchmark_kit/         official scorer, questions, split (exported, pinned)
runs/final/            final test predictions + reports (tracked)
corpus/ data/ indexes/ runs/ external/   large or derived — git-ignored
```
