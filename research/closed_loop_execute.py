#!/usr/bin/env python3
"""Execute the evidence-based Baseball research lifecycle.

This runner consumes only already-produced chronological OOS predictions. It
never invents missing market/PIT evidence. Calibration is selected on a
chronological development-validation split, locked, then evaluated once on an
unseen holdout. The holdout is never used to fit a parameter.

Stages:
  PIT evidence -> calibration -> development OOS -> independent holdout ->
  result audit -> weakness discovery -> candidate validation/adoption gate.

If a required artifact is missing, the stage is BLOCKED rather than fabricated.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

# When invoked as `python research/closed_loop_execute.py`, Python places the
# `research/` directory first on sys.path, not the repository root. Explicitly
# add the root before importing sibling top-level packages such as `evaluation`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from evaluation.calibration import TemperatureCalibration, fit_temperature
from research.adoption_gate import GatePolicy, candidate_lock, evaluate_locked_holdout

RESULTS = ROOT / "results"
PIT_DIR = ROOT / "data" / "pit"


def _write(name: str, obj: Any) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonl_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    n = 0
