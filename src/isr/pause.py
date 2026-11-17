"""Cooperative pause + atomic checkpoints for long jobs (Phase 4 onward).

Pause:   create a file named ``PAUSE`` in the repository root (e.g. ``New-Item PAUSE`` in PowerShell).
         The running job finishes its current chunk, saves it, and exits with code 3.
Resume:  delete ``PAUSE`` and run the same command again; finished chunks are skipped.
Ctrl+C is also safe: every checkpoint is written to a temporary file and renamed into place, so a
partially written file never looks finished.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from isr import ROOT

PAUSE_FILE = ROOT / "PAUSE"
EXIT_PAUSED = 3


class Paused(Exception):
    """Raised at a safe point when the PAUSE file exists."""


def check(where: str = "") -> None:
    if PAUSE_FILE.exists():
        raise Paused(f"paused{' at ' + where if where else ''} (delete {PAUSE_FILE.name} and re-run to resume)")


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def save_npy(path: Path, arr: np.ndarray) -> None:
    def w(tmp: Path) -> None:
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
    _atomic(path, w)


def save_json(path: Path, obj: Any) -> None:
    _atomic(path, lambda tmp: tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8"))
