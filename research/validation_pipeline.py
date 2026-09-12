"""Executable Development OOS -> Candidate Lock -> Holdout -> ADOPT/REJECT pipeline."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping

from research.adoption_gate import GatePolicy, candidate_lock, evaluate_locked_holdout


@dataclass(frozen=True)
class ValidationRecord:
    candidate_id: str
    stage: str
    decision: str
    development: dict[str, Any]
    locked_holdout: dict[str, Any] | None = None


def run_validation_pipeline(
    *,
    candidate_id: str,
    development_metrics: Mapping[str, float],
    holdout_baseline: Mapping[str, float],
    holdout_candidate: Mapping[str, float],
    validation_windows: int,
    calibration_ok: bool,
    no_future_target_data: bool,
    reproducible: bool,
    holdout_score_baseline: Mapping[str, float] | None = None,
    holdout_score_candidate: Mapping[str, float] | None = None,
    holdout_hilo_baseline: Mapping[str, float] | None = None,
    holdout_hilo_candidate: Mapping[str, float] | None = None,
    league: str | None = None,
    policy: GatePolicy = GatePolicy(),
) -> ValidationRecord:
    """Run the full promotion state machine.

    Development metrics are used only to lock the already-selected candidate.
    The locked holdout is evaluated exactly once here and is never used for
    candidate selection. Promotion requires the independent target checks
    configured in GatePolicy.
    """
    locked = candidate_lock(development_metrics=development_metrics, candidate_id=candidate_id)
    result = evaluate_locked_holdout(
        holdout_baseline,
        holdout_candidate,
        policy=policy,
        validation_windows=validation_windows,
        calibration_ok=calibration_ok,
        no_future_target_data=no_future_target_data,
        reproducible=reproducible,
        baseline_score=holdout_score_baseline,
        candidate_score=holdout_score_candidate,
        baseline_hilo=holdout_hilo_baseline,
        candidate_hilo=holdout_hilo_candidate,
        league=league,
    )
    return ValidationRecord(
        candidate_id=candidate_id,
        stage=result["stage"],
        decision=result["decision"],
        development=locked,
        locked_holdout=result,
    )


def can_promote(record: ValidationRecord) -> bool:
    return record.decision == "ADOPT"


def serialize(record: ValidationRecord) -> dict[str, Any]:
    return asdict(record)
