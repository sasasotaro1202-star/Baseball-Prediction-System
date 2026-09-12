"""S-rank orchestration contracts for the Baseball system.

This module connects the new S-rank components without automatically launching
heavy backtests. Numerical model training remains delegated to the existing
BaseballBacktest engine; this layer enforces eligibility, PIT metadata,
calibration, immutable logging, research lifecycle and post-game auditing.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from core.pit_snapshot import SourceSnapshot
from data.availability import AvailabilityRecord, prediction_eligible
from prediction.prediction_log import PredictionRecord, append_prediction


S_RANK_VERSION = "baseball-s-rank-v1"


def pregame_gate(*, availability: AvailabilityRecord, snapshots: list[SourceSnapshot],
                 required_data_ok: bool, feature_complete: bool,
                 model_available: bool, calibration_available: bool) -> dict[str, Any]:
    """Fail closed before production prediction."""
    ok, reasons = prediction_eligible(availability)
    if not required_data_ok:
        reasons.append("required_data_unavailable")
    if not feature_complete:
        reasons.append("feature_incomplete")
    if not model_available:
        reasons.append("model_unavailable")
    if not calibration_available:
        reasons.append("calibration_unavailable")
    if not snapshots:
        reasons.append("no_pit_snapshot")
    for snapshot in snapshots:
        snapshot.validate()
    return {"eligible": not reasons, "reasons": reasons, "version": S_RANK_VERSION}


def persist_prediction(record: PredictionRecord, *, log_path: str | Path) -> dict[str, Any]:
    """Persist one already-calibrated, eligibility-approved production row."""
    append_prediction(record, log_path)
    return {"prediction_id": record.prediction_id, "event_id": record.event_id,
            "league": record.league, "status": "LOGGED", "version": S_RANK_VERSION}


def lifecycle_contract(league: str) -> dict[str, Any]:
    """Return the exact research lifecycle expected for the selected league."""
    if league not in {"NPB", "MLB"}:
        raise ValueError("league must be NPB or MLB")
    return {
        "league": league,
        "development_oos": True,
        "candidate_selection": "development_only",
        "candidate_lock_before_holdout": True,
        "independent_holdout": True,
        "baseline_vs_candidate": True,
        "decision_states": ["ADOPT", "REJECT", "HOLD", "NO_CHANGE"],
        "npb_three_way": league == "NPB",
        "score_and_hilo_required_for_promotion": True,
    }
