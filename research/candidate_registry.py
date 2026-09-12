"""Candidate lifecycle registry for leakage-safe Baseball research."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from research.validation_pipeline import run_validation_pipeline

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "results" / "candidate_registry.json"


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    selected_at: str
    git_commit: str
    feature_version: str
    model_version: str
    development: dict[str, Any]
    lock: dict[str, Any]
    holdout: dict[str, Any] | None
    decision: str


def _load() -> list[dict[str, Any]]:
    if not REGISTRY.exists():
        return []
    try:
        value = json.loads(REGISTRY.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except Exception:
        return []


def record_candidate(
    *,
    candidate_id: str,
    git_commit: str,
    feature_version: str,
    model_version: str,
    development_metrics: Mapping[str, float],
    holdout_baseline: Mapping[str, float],
    holdout_candidate: Mapping[str, float],
    validation_windows: int,
    calibration_ok: bool,
    no_future_target_data: bool,
    reproducible: bool,
    holdout_score_baseline: Mapping[str, float] | None,
    holdout_score_candidate: Mapping[str, float] | None,
    holdout_hilo_baseline: Mapping[str, float] | None,
    holdout_hilo_candidate: Mapping[str, float] | None,
    league: str | None = None,
) -> CandidateRecord:
    """Lock on development metrics, then evaluate the independent holdout."""
    validation = run_validation_pipeline(
        candidate_id=candidate_id,
        development_metrics=development_metrics,
        holdout_baseline=holdout_baseline,
        holdout_candidate=holdout_candidate,
        validation_windows=validation_windows,
        calibration_ok=calibration_ok,
        no_future_target_data=no_future_target_data,
        reproducible=reproducible,
        holdout_score_baseline=holdout_score_baseline,
        holdout_score_candidate=holdout_score_candidate,
        holdout_hilo_baseline=holdout_hilo_baseline,
        holdout_hilo_candidate=holdout_hilo_candidate,
        league=league,
    )
    record = CandidateRecord(
        candidate_id=candidate_id,
        selected_at=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit,
        feature_version=feature_version,
        model_version=model_version,
        development=validation.development,
        lock={"stage": "candidate_locked", "holdout_evaluated": False},
        holdout=validation.locked_holdout,
        decision=validation.decision,
    )
    history = _load()
    history.append(asdict(record))
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(history[-500:], ensure_ascii=False, indent=2), encoding="utf-8")
    return record
