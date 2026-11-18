"""Phase 5 driver: the single frozen test run of every system (plan.md Phase 5 protocol). Resumable and pausable.

    python scripts/phase5.py status
    python scripts/phase5.py run

Runs, in order, each system once on the TEST split (each evaluation logged to TEST_RUNS.md by the evaluator),
skipping systems whose test predictions already exist. Refuses to run unless the working tree is clean and
HEAD carries the tag ``phase5-frozen`` (so every test number maps to one frozen commit).

Pause:  New-Item PAUSE  -> stops at the next question / between systems (exit 3); LLM replies are cached.
Resume: Remove-Item PAUSE; python scripts/phase5.py run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from isr import ROOT, RUNS_DIR, pause

PY = sys.executable
REF = RUNS_DIR / "baselines" / "bm25-two-stage_test.json"      # best shipped BM25 on dev (macro R@8k .211)
TAG = "phase5-frozen"
SYSTEMS = [   # (config, runner)
    ("configs/A1d.yaml", "scripts/run_system.py"),     # best dense-only
    ("configs/A_full.yaml", "scripts/run_system.py"),  # Plan A
    ("configs/B0.yaml", "scripts/run_b.py"),           # Plan A + extractive answer (answer baseline, no LLM)
    ("configs/B_full.yaml", "scripts/run_b.py"),       # Plan B
    ("configs/C_full.yaml", "scripts/run_b.py"),       # Plan C
]


def _pred(cfg: str) -> Path:
    name = yaml.safe_load((ROOT / cfg).read_text(encoding="utf-8"))["name"]
    return RUNS_DIR / name / "pred_test.json"


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def status() -> int:
    for cfg, _ in SYSTEMS:
        print(f"{cfg:<22} {'done' if _pred(cfg).exists() else 'pending'}")
    print(f"HEAD tagged {TAG}: {TAG in _git('tag', '--points-at', 'HEAD').split()}")
    print(f"PAUSE file: {'present' if pause.PAUSE_FILE.exists() else 'absent'}")
    return 0


def run() -> int:
    if TAG not in _git("tag", "--points-at", "HEAD").split():
        print(f"refusing: HEAD is not tagged {TAG}")
        return 2
    dirty = [l for l in _git("status", "--porcelain").splitlines()       # the evaluator appends to TEST_RUNS.md
             if not l.endswith(("PAUSE", "TEST_RUNS.md"))]
    if dirty:
        print("refusing: working tree not clean:\n" + "\n".join(dirty))
        return 2
    commit = _git("rev-parse", "--short", "HEAD")
    for cfg, runner in SYSTEMS:
        if _pred(cfg).exists():
            print(f"{cfg}: done (skipping)")
            continue
        pause.check(f"before {cfg}")
        note = f"Phase 5 final test run, frozen commit {commit} ({TAG}): {Path(cfg).stem}"
        print(f"\n=== {cfg}", flush=True)
        rc = subprocess.run([PY, runner, cfg, "--split", "test", "--compare", str(REF), "--test-note", note],
                            cwd=ROOT).returncode
        if rc:
            return rc
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "run"])
    args = ap.parse_args()
    try:
        return status() if args.cmd == "status" else run()
    except pause.Paused as e:
        print(e)
        return pause.EXIT_PAUSED


if __name__ == "__main__":
    raise SystemExit(main())
