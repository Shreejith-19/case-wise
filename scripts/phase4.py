"""Phase 4 (Plan C) driver -- resumable and pausable.

    python scripts/phase4.py status
    python scripts/phase4.py build                      # Act-card index + cross-reference graph
    python scripts/phase4.py run configs/C1.yaml ...    # retrieval ablations on dev_tune (skips finished runs)

Pause:  create a file named PAUSE in the repository root (PowerShell: New-Item PAUSE). The current chunk or
        run finishes and is saved, then the driver exits with code 3.
Resume: delete PAUSE and run the same command again; finished work is skipped.
Ctrl+C is safe too (checkpoints are written atomically); at worst the run in progress is redone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from isr import INDEX_DIR, RUNS_DIR, pause

HERE = Path(__file__).resolve().parent
COMPARE = RUNS_DIR / "A-full" / "pred_dev.json"


def _done(cfg_path: Path, split: str) -> bool:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    pred = RUNS_DIR / cfg["name"] / f"pred_{split}.json"
    return pred.exists() and pred.stat().st_mtime > cfg_path.stat().st_mtime


def status() -> int:
    acts = INDEX_DIR / "acts"
    parts = sorted(acts.glob("dense_*.part*.npy"))
    print(f"Act-card index : {'done' if (acts / 'info.json').exists() else f'{len(parts)} chunk(s) saved'}")
    xr = INDEX_DIR / "xref"
    print(f"xref graph     : {'done' if (xr / 'info.json').exists() else f'{len(list(xr.glob(chr(101) + 'dges.part*.npy')))} chunk(s) saved'}")
    for c in sorted((HERE.parent / "configs").glob("C*.yaml")):
        print(f"run {c.name:<16}: {'done' if _done(c, 'dev_tune') else 'pending'}")
    print(f"PAUSE file     : {'present (jobs will stop at the next safe point)' if pause.PAUSE_FILE.exists() else 'absent'}")
    return 0


def build() -> int:
    from isr import acts, xref
    for name, fn, marker in (("Act-card index", acts.build, INDEX_DIR / "acts" / "info.json"),
                             ("xref graph", xref.build, INDEX_DIR / "xref" / "info.json")):
        if marker.exists():
            print(f"{name}: already built")
            continue
        print(f"{name}: building ...", flush=True)
        print(json.dumps(fn(), indent=1))
    return 0


def run(configs: list[Path], split: str) -> int:
    sys.path.insert(0, str(HERE))
    import run_system
    for c in configs:
        if _done(c, split):
            print(f"{c.name}: done (skipping)")
            continue
        pause.check(f"before {c.name}")
        print(f"\n=== {c.name} on {split}", flush=True)
        sys.argv = ["run_system.py", str(c), "--split", split, "--compare", str(COMPARE)]
        rc = run_system.main()
        if rc:
            return rc
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "build", "run"])
    ap.add_argument("configs", nargs="*", type=Path)
    ap.add_argument("--split", default="dev_tune", choices=["dev_tune"])
    args = ap.parse_args()
    try:
        if args.cmd == "status":
            return status()
        if args.cmd == "build":
            return build()
        return run(args.configs, args.split)
    except pause.Paused as e:
        print(e)
        return pause.EXIT_PAUSED


if __name__ == "__main__":
    raise SystemExit(main())
