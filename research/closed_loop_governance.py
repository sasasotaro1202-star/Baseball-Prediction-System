"""Fail-closed governance for the Baseball research lifecycle.

This module does not manufacture evidence. It verifies that each stage has the
artifacts needed to move to the next stage and records explicit blockers when
those artifacts are absent.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Stage:
    name: str
    status: str
    blockers: tuple[str, ...] = ()


REQUIRED_PIT = (
    "data/pit/source_snapshots.jsonl",
    "data/pit/event_observations.jsonl",
    "data/pit/availability_observations.jsonl",
    "data/pit/acquisition_runs.jsonl",
)


def _nonempty(path: str | Path) -> bool:
    p = Path(path)
    return p.exists() and p.is_file() and p.stat().st_size > 0


def _jsonl_valid(path: str | Path) -> tuple[bool, int, str | None]:
    p = Path(path)
    if not _nonempty(p):
        return False, 0, "missing_or_empty"
    rows = 0
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                json.loads(line)
                rows += 1
    except Exception as exc:
        return False, rows, f"invalid_jsonl:{type(exc).__name__}"
    return rows > 0, rows, None if rows else "no_rows"


def pit_stage() -> Stage:
    blockers: list[str] = []
    for path in REQUIRED_PIT:
        ok, rows, err = _jsonl_valid(path)
        if not ok:
            blockers.append(f"{path}:{err or 'invalid'}")
    return Stage("PIT", "READY" if not blockers else "BLOCKED", tuple(blockers))


def calibration_stage(calibration_artifact: str | Path = "results/calibration.json") -> Stage:
    if not _nonempty(calibration_artifact):
        return Stage("Calibration", "BLOCKED", ("calibration_artifact_missing",))
    try:
        obj = json.loads(Path(calibration_artifact).read_text(encoding="utf-8"))
        temperature = float(obj["temperature"])
        if not temperature > 0:
            raise ValueError("temperature must be positive")
    except Exception as exc:
        return Stage("Calibration", "BLOCKED", (f"invalid_calibration:{type(exc).__name__}",))
    return Stage("Calibration", "READY")


def oos_stage(oos_artifact: str | Path = "results/development_oos.json") -> Stage:
    if not _nonempty(oos_artifact):
        return Stage("Development OOS", "BLOCKED", ("development_oos_missing",))
    return Stage("Development OOS", "READY")


def holdout_stage(holdout_artifact: str | Path = "results/independent_holdout.json") -> Stage:
    if not _nonempty(holdout_artifact):
        return Stage("Independent Holdout", "BLOCKED", ("independent_holdout_missing",))
    return Stage("Independent Holdout", "READY")


def result_stage(result_artifact: str | Path = "results/result_audit.json") -> Stage:
    if not _nonempty(result_artifact):
        return Stage("Result Collection", "BLOCKED", ("result_audit_missing",))
    return Stage("Result Collection", "READY")


def weakness_stage(weakness_artifact: str | Path = "results/weakness_report.json") -> Stage:
    if not _nonempty(weakness_artifact):
        return Stage("Weakness Discovery", "BLOCKED", ("weakness_report_missing",))
    return Stage("Weakness Discovery", "READY")


def candidate_stage(candidate_artifact: str | Path = "results/candidate_validation.json") -> Stage:
    if not _nonempty(candidate_artifact):
        return Stage("Candidate Validation", "BLOCKED", ("candidate_validation_missing",))
    return Stage("Candidate Validation", "READY")


def lifecycle_report() -> dict[str, Any]:
    stages = [
        pit_stage(),
        calibration_stage(),
        oos_stage(),
        holdout_stage(),
        result_stage(),
        weakness_stage(),
        candidate_stage(),
    ]
    ready = not any(s.status == "BLOCKED" for s in stages)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": "READY" if ready else "BLOCKED",
        "stages": [asdict(s) for s in stages],
    }
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True).encode("utf-8")
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    return report


def main() -> int:
    report = lifecycle_report()
    path = Path("results/lifecycle_governance.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # Governance is intentionally non-destructive: it reports blockers but does
    # not pretend that missing evidence is success.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
