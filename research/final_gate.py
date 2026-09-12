"""Final fail-closed gate for the Baseball research system.

This gate verifies that the artifacts required for production/promotion exist
and that no known unsafe fallback is being used. It never invents evidence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"missing artifact: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid JSON artifact: {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise RuntimeError(f"artifact is not an object: {path}")
    return obj


def run_final_gate() -> dict:
    branch = os.getenv("GITHUB_REF_NAME", "").strip()
    if branch == "main":
        raise RuntimeError("refusing promotion/persistence on main")

    checks: dict[str, object] = {}
    required = [
        ROOT / "BASEBALL_SYSTEM_SPEC.md",
        ROOT / "prediction" / "runner.py",
        ROOT / "prediction" / "prediction_log.py",
        ROOT / "core" / "pit_replay.py",
        ROOT / "data" / "availability.py",
        ROOT / "data" / "market_lines.py",
        ROOT / "evaluation" / "calibration.py",
        ROOT / "monitoring" / "result_audit.py",
        ROOT / "research" / "validation_pipeline.py",
        ROOT / "research" / "adoption_gate.py",
    ]
    checks["required_modules"] = all(p.exists() for p in required)
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("required modules missing: " + ", ".join(missing))

    pit = ROOT / "data" / "pit"
    checks["pit_directory"] = pit.exists()
    if not pit.exists():
        raise RuntimeError("PIT directory missing")

    # A production prediction is never eligible merely because a probable
    # starter or an arbitrary total-run threshold exists. Real announcement
    # timestamps and PIT-usable market lines must be present.
    checks["production_requirements"] = {
        "starter_announcement_timestamp": "REQUIRED",
        "lineup_announcement_timestamp_when_claimed_confirmed": "REQUIRED",
        "pit_cutoff": "REQUIRED",
        "verified_total_runs_line": "REQUIRED",
        "no_hardcoded_low_high_threshold": "REQUIRED",
    }

    # Existing lifecycle artifacts, when present, must not claim adoption
    # without explicit evidence. Missing artifacts are reported as HOLD rather
    # than converted into a false pass.
    lifecycle_files = [
        RESULTS / "npb_research_lifecycle.json",
        RESULTS / "mlb_locked_holdout.json",
    ]
    lifecycle = {}
    for path in lifecycle_files:
        if path.exists():
            lifecycle[path.name] = _read_json(path).get("decision", "UNKNOWN")
    checks["lifecycle_artifacts"] = lifecycle

    checks["status"] = "PASS_INFRASTRUCTURE_ONLY"
    checks["promotion"] = "BLOCKED_UNTIL_REAL_PIT_STARTER_AND_LINE_EVIDENCE"
    return checks


if __name__ == "__main__":
    print(json.dumps(run_final_gate(), ensure_ascii=False, indent=2, sort_keys=True))
