#!/usr/bin/env python3
"""Cheap, deterministic preflight checks for the Baseball Actions pipeline.

Keep this gate dependency-light: it must catch broken production imports and
unsafe workflow wiring before expensive research starts, without masking
failures from the real test suite.
"""
from __future__ import annotations

import importlib
import re
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
    "core.atomic_io",
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
    Path("core/atomic_io.py"),
    Path("research/closed_loop_governance.py"),
)

WORKFLOW_DIR = ROOT / ".github" / "workflows"
SHA_ACTION_RE = re.compile(r"^\s*uses:\s*[^@\s]+@([0-9a-fA-F]{40})\s*$", re.MULTILINE)
USES_RE = re.compile(r"^\s*uses:\s*([^@\s]+)@([^\s#]+)", re.MULTILINE)


def _workflow_reliability_checks() -> None:
    """Reject workflow wiring that would make CI non-reproducible or fragile."""
    if not WORKFLOW_DIR.is_dir():
        raise SystemExit("Missing .github/workflows directory")
    workflows = sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))
    if not workflows:
        raise SystemExit("No GitHub Actions workflow files found")

    checked = 0
    for path in workflows:
        text = path.read_text(encoding="utf-8")
        uses = list(USES_RE.finditer(text))
        for match in uses:
            ref = match.group(2)
            if not re.fullmatch(r"[0-9a-fA-F]{40}", ref):
                raise SystemExit(
                    f"Workflow action is not pinned to an immutable full SHA: {path}: {match.group(1)}@{ref}"
                )
        if uses and len(SHA_ACTION_RE.findall(text)) != len(uses):
            raise SystemExit(f"Workflow action SHA parsing mismatch: {path}")
        if "runs-on:" not in text:
            raise SystemExit(f"Workflow has no runner declaration: {path}")
        checked += 1
    print(f"OK: {checked} workflow files use immutable action refs")


def main() -> int:
    missing = [str(p) for p in REQUIRED_FILES if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise SystemExit("Missing required files: " + ", ".join(missing))
    for name in REQUIRED_MODULES:
        importlib.import_module(name)
        print(f"OK: {name}")
    _workflow_reliability_checks()
    print("Baseball Action preflight OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
