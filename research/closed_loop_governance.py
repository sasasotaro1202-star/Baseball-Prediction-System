"""Fail-closed governance for the Baseball research lifecycle.

This module does not manufacture evidence. It verifies that each stage has the
artifacts needed to move to the next stage and records explicit blockers when
those artifacts are absent. PIT revision/backfill findings are treated as a
research blocker when a source claims historical availability earlier than an
already observed availability timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from core.pit_revision import detect_revisions


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


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not _nonempty(p):
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"PIT JSONL row must be an object: {p}")
        rows.append(obj)
    return rows


def pit_revision_stage(snapshot_artifact: str | Path = "data/pit/source_snapshots.jsonl") -> Stage:
    """Block research when historical availability has been backfilled.

    Ordinary payload revisions are recorded evidence but do not automatically
    block a run. A BACKFILL changes what was knowable at an earlier cutoff and
    therefore requires replay/recomputation before any OOS result can be used.
    """
    if not _nonempty(snapshot_artifact):
        return Stage("PIT Revision/Backfill", "BLOCKED", ("pit_snapshot_artifact_missing",))
    try:
        findings = detect_revisions(_read_jsonl(snapshot_artifact))
    except Exception as exc:
        return Stage("PIT Revision/Backfill", "BLOCKED", (f"pit_revision_audit_error:{type(exc).__name__}",))
    backfills = [f for f in findings if f.kind == "BACKFILL"]
    if backfills:
        return Stage(
            "PIT Revision/Backfill",
            "BLOCKED",
            (f"pit_backfill_detected:{len(backfills)}",),
        )
    return Stage("PIT Revision/Backfill", "READY")


def pit_stage() -> Stage:
    blockers: list[str] = []
    for path in REQUIRED_PIT:
        ok, rows, err = _jsonl_valid(path)
        if not ok:
            blockers.append(f"{path}:{err or 'invalid'}")
    if not blockers:
        revision = pit_revision_stage()
        blockers.extend(revision.blockers)
    return Stage("PIT", "READY" if not blockers else "BLOCKED", tuple(blockers))


def calibration_stage(calibration_artifact: str | Path = "results/calibration.json") -> Stage:
    """Validate the current calibration schema without assuming a global scalar.

    Production calibration is stored per league because NPB and MLB have
    different class structures and probability sharpness. Older artifacts may
    contain a single top-level ``temperature``; those remain accepted for
    backward compatibility, but the current per-league schema is preferred.
    """
    if not _nonempty(calibration_artifact):
        return Stage("Calibration", "BLOCKED", ("calibration_artifact_missing",))
    try:
        obj = json.loads(Path(calibration_artifact).read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("calibration artifact must be an object")
        if "temperature" in obj:
            temperatures = {"global": obj["temperature"]}
        else:
            leagues = obj.get("leagues")
            if not isinstance(leagues, dict) or not leagues:
                raise ValueError("calibration artifact has no per-league calibrations")
            temperatures = {}
            for league, payload in leagues.items():
                if not isinstance(payload, dict) or "temperature" not in payload:
                    raise ValueError(f"missing temperature for league {league}")
                temperatures[str(league)] = payload["temperature"]
        for league, value in temperatures.items():
            temperature = float(value)
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError(f"temperature must be positive and finite for {league}")
    except Exception as exc:
        return Stage("Calibration", "BLOCKED", (f"invalid_calibration:{type(exc).__name__}",))
    return Stage("Calibration", "READY")


def _validate_league_artifact(path: str | Path, required_leagues: set[str]) -> tuple[bool, tuple[str, ...]]:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return False, ("artifact_not_object",)
        missing = sorted(required_leagues - set(obj))
        return (not missing), tuple(f"missing_league:{x}" for x in missing)
    except Exception as exc:
        return False, (f"invalid_json:{type(exc).__name__}",)


def oos_stage(oos_artifact: str | Path = "results/development_oos.json") -> Stage:
    if not _nonempty(oos_artifact):
        return Stage("Development OOS", "BLOCKED", ("development_oos_missing",))
    ok, blockers = _validate_league_artifact(oos_artifact, {"NPB", "MLB"})
    return Stage("Development OOS", "READY" if ok else "BLOCKED", blockers)


def holdout_stage(holdout_artifact: str | Path = "results/independent_holdout.json") -> Stage:
    if not _nonempty(holdout_artifact):
        return Stage("Independent Holdout", "BLOCKED", ("independent_holdout_missing",))
    try:
        obj = json.loads(Path(holdout_artifact).read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("holdout artifact must be an object")
        blockers: list[str] = []
        for league in ("NPB", "MLB"):
            payload = obj.get(league)
            if not isinstance(payload, dict):
                blockers.append(f"missing_league:{league}")
                continue
            if payload.get("used_for_candidate_selection") is not False:
                blockers.append(f"holdout_selection_contamination:{league}")
            for side in ("baseline", "candidate"):
                metrics = payload.get(side)
                if not isinstance(metrics, dict):
                    blockers.append(f"missing_holdout_metrics:{league}:{side}")
                    continue
                for key in ("rows", "LogLoss", "Brier", "Accuracy"):
                    try:
                        value = float(metrics[key])
                    except (KeyError, TypeError, ValueError):
                        blockers.append(f"invalid_holdout_metric:{league}:{side}:{key}")
                        continue
                    if not math.isfinite(value) or (key == "rows" and (value < 0 or value % 1 != 0)):
                        blockers.append(f"invalid_holdout_metric:{league}:{side}:{key}")
            if isinstance(payload.get("baseline"), dict) and isinstance(payload.get("candidate"), dict):
                try:
                    if int(payload["baseline"]["rows"]) != int(payload["candidate"]["rows"]):
                        blockers.append(f"holdout_row_mismatch:{league}")
                except (KeyError, TypeError, ValueError):
                    pass
        return Stage("Independent Holdout", "READY" if not blockers else "BLOCKED", tuple(blockers))
    except Exception as exc:
        return Stage("Independent Holdout", "BLOCKED", (f"invalid_holdout:{type(exc).__name__}",))


def result_stage(result_artifact: str | Path = "results/result_audit.json") -> Stage:
    if not _nonempty(result_artifact):
        return Stage("Result Collection", "BLOCKED", ("result_audit_missing",))
    ok, blockers = _validate_league_artifact(result_artifact, {"NPB", "MLB"})
    return Stage("Result Collection", "READY" if ok else "BLOCKED", blockers)


def weakness_stage(weakness_artifact: str | Path = "results/weakness_report.json") -> Stage:
    if not _nonempty(weakness_artifact):
        return Stage("Weakness Discovery", "BLOCKED", ("weakness_report_missing",))
    return Stage("Weakness Discovery", "READY")


def candidate_stage(candidate_artifact: str | Path = "results/candidate_validation.json") -> Stage:
    """Validate candidate decisions before lifecycle promotion.

    A present file is not evidence of a completed candidate evaluation. Both
    leagues must have an explicit ADOPT/REJECT decision and finite primary
    metrics; malformed or incomplete artifacts fail closed.
    """
    if not _nonempty(candidate_artifact):
        return Stage("Candidate Validation", "BLOCKED", ("candidate_validation_missing",))
    try:
        obj = json.loads(Path(candidate_artifact).read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("candidate validation artifact must be an object")
        blockers: list[str] = []
        for league in ("NPB", "MLB"):
            payload = obj.get(league)
            if not isinstance(payload, dict):
                blockers.append(f"missing_league:{league}")
                continue
            decision = payload.get("decision")
            if decision not in {"ADOPT", "REJECT"}:
                blockers.append(f"invalid_candidate_decision:{league}")
            for key in ("baseline", "candidate"):
                metrics = payload.get(key)
                if not isinstance(metrics, dict):
                    blockers.append(f"missing_candidate_metrics:{league}:{key}")
                    continue
                for metric in ("LogLoss", "Brier", "Accuracy", "rows"):
                    try:
                        value = float(metrics[metric])
                    except (KeyError, TypeError, ValueError):
                        blockers.append(f"invalid_candidate_metric:{league}:{key}:{metric}")
                        continue
                    if not math.isfinite(value):
                        blockers.append(f"nonfinite_candidate_metric:{league}:{key}:{metric}")
        return Stage("Candidate Validation", "READY" if not blockers else "BLOCKED", tuple(blockers))
    except Exception as exc:
        return Stage("Candidate Validation", "BLOCKED", (f"invalid_candidate_validation:{type(exc).__name__}",))


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
        "schema_version": 2,
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
