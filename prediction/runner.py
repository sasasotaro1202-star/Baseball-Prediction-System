"""Production prediction eligibility gate and canonical logging adapter."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Callable, Mapping

from core.pit import _ts
from data.availability import AvailabilityRecord, prediction_eligible
from prediction.prediction_log import PredictionRecord, append_prediction, make_prediction_id


def eligibility_gate(*, availability: AvailabilityRecord, required_data_ok: bool,
                     feature_complete: bool, model_available: bool,
                     calibration_available: bool) -> tuple[bool, list[str]]:
    """Fail closed unless every required production condition is satisfied."""
    ok, reasons = prediction_eligible(availability)
    if not required_data_ok:
        reasons.append("required_data_unavailable")
    if not feature_complete:
        reasons.append("feature_incomplete")
    if not model_available:
        reasons.append("model_unavailable")
    if not calibration_available:
        reasons.append("calibration_unavailable")
    cutoff = _ts(availability.prediction_cutoff)
    retrieved = _ts(availability.retrieved_at)
    if retrieved > cutoff:
        reasons.append("retrieval_after_cutoff")
    return (not reasons, reasons)


def _validate_final_probabilities(probabilities: Mapping[str, float], league: str) -> dict[str, float]:
    if not isinstance(probabilities, Mapping):
        raise ValueError("probability callback must return a mapping")
    expected = {"home", "away"} | ({"draw"} if league == "NPB" else set())
    if set(probabilities) != expected:
        raise ValueError("probability callback returned the wrong league contract")
    values: dict[str, float] = {}
    for key, raw in probabilities.items():
        value = float(raw)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("final probabilities must be finite and in [0,1]")
        values[key] = value
    if abs(sum(values.values()) - 1.0) > 1e-8:
        raise ValueError("final probabilities must sum to 1")
    return values


def _optional_probability(value: Any, name: str) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return value


def _validate_score_candidates(candidates: Any) -> list[dict[str, Any]]:
    if candidates is None:
        return []
    if not isinstance(candidates, (list, tuple)):
        raise ValueError("score_candidates must be a list or tuple")
    out = []
    for item in candidates:
        if not isinstance(item, Mapping) or "score" not in item or "probability" not in item:
            raise ValueError("each score candidate requires score and probability")
        p = float(item["probability"])
        if not math.isfinite(p) or p < 0 or p > 1:
            raise ValueError("score candidate probability must be finite and in [0,1]")
        out.append(dict(item))
    return out


def run_prediction(*, row: Mapping[str, Any], availability: AvailabilityRecord,
                   probability_fn: Callable[[Mapping[str, Any]], Mapping[str, float]],
                   model_version: str, feature_version: str, calibration_version: str,
                   git_commit: str, data_snapshot_id: str, log_path: str,
                   calibrate_fn: Callable[[Mapping[str, float]], Mapping[str, float]] | None = None) -> dict[str, Any]:
    eligible, reasons = eligibility_gate(
        availability=availability,
        required_data_ok=bool(row.get("required_data_ok", True)),
        feature_complete=bool(row.get("feature_complete", True)),
        model_available=bool(row.get("model_available", True)),
        calibration_available=bool(row.get("calibration_available", True)),
    )
    if not eligible:
        return {"eligible": False, "reasons": reasons, "event_id": availability.event_id}

    raw = _validate_final_probabilities(probability_fn(row), availability.league)
    calibrated = calibrate_fn(raw) if calibrate_fn is not None else raw
    probabilities = _validate_final_probabilities(calibrated, availability.league)

    now = datetime.now(timezone.utc)
    if now < _ts(availability.prediction_cutoff):
        raise ValueError("prediction cannot be created before its declared cutoff")

    total_line = row.get("total_runs_line")
    if total_line is not None:
        total_line = float(total_line)
        if not math.isfinite(total_line) or total_line < 0 or abs(total_line * 2 - round(total_line * 2)) > 1e-9:
            raise ValueError("total_runs_line must be finite, non-negative, integer or half-point")

    low = _optional_probability(row.get("low_probability"), "low_probability")
    high = _optional_probability(row.get("high_probability"), "high_probability")
    if (low is None) != (high is None):
        raise ValueError("low_probability and high_probability must be supplied together")
    if low is not None and abs(low + high - 1.0) > 1e-8:
        raise ValueError("low/high probabilities must sum to 1")

    pid = make_prediction_id(availability.event_id, availability.prediction_cutoff, model_version, git_commit)
    record = PredictionRecord(
        prediction_id=pid, event_id=availability.event_id, league=availability.league,
        prediction_cutoff=availability.prediction_cutoff, prediction_created_at=now.isoformat(),
        home_team=availability.home_team, away_team=availability.away_team,
        home_starter=availability.home_starter, away_starter=availability.away_starter,
        probabilities=probabilities, score_candidates=_validate_score_candidates(row.get("score_candidates", [])),
        low_probability=low, high_probability=high, total_runs_line=total_line,
        confidence=_optional_probability(row.get("confidence"), "confidence"),
        volatility=_optional_probability(row.get("volatility"), "volatility"),
        model_version=model_version, feature_version=feature_version,
        calibration_version=calibration_version, git_commit=git_commit,
        data_snapshot_id=data_snapshot_id,
    )
    append_prediction(record, log_path)
    return {"eligible": True, "prediction": record}
