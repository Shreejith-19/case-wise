"""A runnable stub of the system contract. Replace retrieve() with your pipeline.

    python example_system.py --corpus /path/to/corpus --split test
"""

import argparse
import json
from pathlib import Path

from benchmark import score, splits
from benchmark.questions import load_questions


class MySystem:
    name = "my-system"

    def retrieve(self, question: str, budget_chars: int):
        """Ranked (document_id, char_start, char_end), best first, within the budget.

        Returning [] is a real answer: it is how you abstain, and the
        unanswerable questions are where abstaining earns credit.
        """
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    questions = splits.select(load_questions(verified_only=True),
                              sorted(splits.SPLITS_DIR.glob("split-s*.json"))[0], args.split)
    run = score.run_system(MySystem(), questions, score.DEFAULT_BUDGETS)
    report = score.score_run(questions, run, system=MySystem.name,
                             corpus_dir=args.corpus, split=args.split)
    print(score.format_report(report))
    Path("report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Or write a predictions file and score it without importing this package:
    #   {"system": "my-system", "split": "test",
    #    "predictions": {"IN-STAT-0001": {"2000": [["doc-id", 0, 1500]], ...}}}
    #   python -m benchmark score --predictions run.json --corpus /path/to/corpus
