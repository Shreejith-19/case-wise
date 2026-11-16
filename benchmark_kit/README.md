# IndiaStatRAG — scoring kit

Everything needed to score a retrieval system against the IndiaStatRAG question
set. Copy `benchmark/` into your project; it needs Python 3.9+ and nothing else.

## What is here

| path | what it is |
| --- | --- |
| `benchmark/data/questions/` | the questions, gold answers and gold evidence (one JSON per question) |
| `benchmark/data/splits/` | the public dev/test assignment — a result must say which it used |
| `benchmark/score.py` | the scorer |
| `benchmark/baselines/` | oracle, bm25-windows, bm25-two-stage |
| `example_system.py` | a runnable stub of the system contract |

You also need the **published corpus** (the `text/` and `meta/` directories,
`documents.jsonl` and `CHECKSUMS.txt`). Pass it with `--corpus`.

## The contract

```python
def retrieve(self, question: str, budget_chars: int) -> list[tuple[str, int, int]]:
    """Ranked (document_id, char_start, char_end), best first. [] means abstain."""
```

Offsets are character offsets into `corpus/text/<document_id>.txt`. Chunk
however you like — the benchmark never asks about your chunks.

## Score a run

Write a predictions file and score it, importing nothing:

```json
{"system": "my-rag", "split": "test",
 "predictions": {"IN-STAT-0001": {"2000": [["act__handle-1", 120, 1500]],
                                  "8000": [], "32000": []}}}
```

```
python -m benchmark score --predictions run.json --corpus ./corpus --split test
```

Or call the scorer directly — see `example_system.py`.

## Reading the report

Recall and precision are always reported together, per category, at three
budgets. There is deliberately no single averaged number: the categories
measure different things. `oracle` is the ceiling; at B=2,000 the ceiling is
below 1.0 because a gold provision can be longer than the budget.
