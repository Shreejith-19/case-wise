"""isr — Indian Statutory RAG. See plan.md for the design."""

from pathlib import Path

#: Repository root (…/Claude). Everything else is resolved from here.
ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "corpus"
KIT_DIR = ROOT / "benchmark_kit"
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "indexes"
RUNS_DIR = ROOT / "runs"
