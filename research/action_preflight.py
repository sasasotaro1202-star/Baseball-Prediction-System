#!/usr/bin/env python3
"""Cheap, deterministic preflight checks for the Baseball Actions pipeline."""
from __future__ import annotations

import importlib
from pathlib import Path

REQUIRED_MODULES = (
    "baseball_backtest",
    "research_runner",
    "data.npb_pbp_adapter",
    "research.validation_pipeline",
    "research.adoption_gate",
    "prediction.runner",
    "prediction.prediction_log",
    "evaluation.calibration",
)

REQUIRED_FILES = (
    Path("requirements.txt"),
    Path("requirements-ci.txt"),
    Path("requirements-pit.txt"),
    Path("research_runner.py"),
    Path("baseball_backtest.py"),
    Path("data/pit_acquisition.py"),
    Path("core/pit_snapshot.py"),
    Path("prediction/runner.py"),
    Path("evaluation/calibration.py"),
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
