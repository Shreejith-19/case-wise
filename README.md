# isr — Indian Statutory RAG on IndiaStatRAG

Structure-aware, local-LLM-assisted retrieval-augmented generation over Indian statutory law.
Design, decisions and phases: [plan.md](plan.md). Test-split evaluation log: [TEST_RUNS.md](TEST_RUNS.md).

## Pinned evaluation setup

| item | value |
|---|---|
| Benchmark | IndiaStatRAG — `Advaith-Srinivas05/Indian-legal-benchmark` @ `b45befa5fa7f6242dbe9b15cf7b496a0be80921a` |
| Scoring kit | `benchmark_kit/` (from `python -m benchmark export`), scorer version 1 |
| Split | `split-s20260923.json` — dev 150 / test 352 |
| Corpus | `corpus/`, schema v1, 15,325 documents |
| `CHECKSUMS.txt` sha256 | `33b624048b536b14f807178276d61e8efda758d04e9e0406740ca65bd83b69f2` |
| Budgets / τ | 2,000 / **8,000** (headline) / 32,000 chars; τ = 0.5 (0.8, 1.0 reported) |
| Tuned on test | no (all tuning on dev; see plan.md D6) |

## Environment (verified 2026-11-16)

| item | value |
|---|---|
| GPU / CPU / RAM | RTX 4060 Laptop 8 GB · Ryzen 7 7840HS · 15 GB |
| Python / torch | 3.12.4 · torch 2.13.0+cu130 (exact packages: `requirements.lock.txt`) |
| Ollama | 0.17.7 |
| `qwen3:8b` | Q4_K_M, digest `500a1f067a9f…`, 100% VRAM, ~1.7k prompt tok/s, ~43 gen tok/s @ num_ctx 8192 |
| `llama3.1:8b` | Q4_K_M, digest `46e0c10c039e…`, 100% VRAM, ~1.7k prompt tok/s, ~47 gen tok/s @ num_ctx 8192 |

## Phase 0 sanity results

- `oracle`: recall 1.000 at B=8k/32k on dev and test (B=2k ceiling 0.81–1.00) — scorer and corpus consistent.
- Shipped BM25 baselines on **test** reproduce the benchmark README table exactly (B=8k evidence recall):

  | category | bm25-windows | bm25-two-stage |
  |---|---:|---:|
  | provision_lookup | 0.491 | 0.528 |
  | definitional | 0.382 | 0.294 |
  | numeric_threshold | 0.417 | 0.250 |
  | cross_reference | 0.235 | 0.196 |
  | jurisdictional | 0.100 | 0.120 |
  | situational | 0.075 | 0.065 |

  Both abstain 0.000 on unanswerable. BM25 index: 794,541 windows, 693 MB, built in ~6 min; querying ≈ 0.8–1.8 s/question.

## Setup (Windows, Python 3.12)

```bash
python -m venv .venv
.venv/Scripts/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/pip install -r requirements.txt      # exact versions: requirements.lock.txt
.venv/Scripts/pip install -e .

# full benchmark repo (only needed to build the shipped BM25 baseline index)
git clone https://github.com/Advaith-Srinivas05/Indian-legal-benchmark external/Indian-legal-benchmark
git -C external/Indian-legal-benchmark checkout b45befa5fa7f6242dbe9b15cf7b496a0be80921a
cd external/Indian-legal-benchmark
python -m benchmark --out ../../corpus index --path ../../indexes/bm25_baseline.sqlite
cd ../..

# local LLMs (Ollama)
ollama pull qwen3:8b
ollama pull llama3.1:8b
```

## Evaluating

```bash
# shipped baselines -> predictions files
.venv/Scripts/python scripts/run_baseline.py bm25-windows --split dev

# official scorer + per-category bootstrap 95% CIs (+ paired test against a reference run)
.venv/Scripts/python -m isr.eval.evaluate runs/baselines/bm25-windows_dev.json --split dev
.venv/Scripts/python -m isr.eval.evaluate runs/X/pred_dev.json --split dev --compare runs/baselines/bm25-windows_dev.json

# test split: only for frozen systems; requires a note and is logged to TEST_RUNS.md
.venv/Scripts/python -m isr.eval.evaluate runs/X/pred_test.json --split test --test-note "frozen A-full @ <commit>"
```

## Layout

```
plan.md            design + decision log
benchmark_kit/     official scorer, questions, split (exported, pinned)
src/isr/           our package (eval/ now; corpus/, index/, query/, retrieve/, … per plan.md §5.1)
scripts/           CLI entry points
tests/             pytest
configs/ prompts/  system configs and versioned prompt templates
corpus/ data/ indexes/ runs/ external/   large or derived — git-ignored
```
