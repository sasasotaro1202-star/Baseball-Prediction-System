#!/usr/bin/env python3
"""Cheap, deterministic preflight checks for the Baseball Actions pipeline.

Keep this gate dependency-light: it must catch broken production imports before
expensive research starts, without masking failures from the real test suite.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_MODULES = (
    "baseball_backtest",
    "research_runner",
    "research_runner_v6",
    "data.npb_pbp_adapter",
    "data.availability",
    "research.validation_pipeline",
    "research.adoption_gate",
    "research.closed_loop_governance",
    "prediction.runner",
    "prediction.prediction_log",
    "prediction.score_distribution",
    "evaluation.calibration",
)

REQUIRED_FILES = (
    Path("requirements.txt"),
    Path("requirements-ci.txt"),
    Path("requirements-pit.txt"),
    Path("research_runner.py"),
    Path("research_runner_v6.py"),
    Path("baseball_backtest.py"),
    Path("data/pit_acquisition.py"),
    Path("data/availability.py"),
    Path("core/pit_snapshot.py"),
    Path("core/pit_replay.py"),
    Path("prediction/runner.py"),
    Path("prediction/prediction_log.py"),
    Path("prediction/score_distribution.py"),
    Path("evaluation/calibration.py"),
    Path("research/closed_loop_governance.py"),
)


def main() -> int:
    missing = [str(p) for p in REQUIRED_FILES if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise SystemExit("Missing required files: " + ", ".join(missing))
    for name in REQUIRED_MODULES:
        importlib.import_module(name)
        print(f"OK: {name}")
    print("Baseball Action preflight OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
