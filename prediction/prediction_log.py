"""Immutable prediction ledger for Baseball production outputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _dt(value: str) -> datetime:
    ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        raise ValueError("prediction timestamps must be timezone-aware")
    return ts


def _finite_probability(value: Any) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise ValueError("probabilities must be finite and in [0,1]")
    return v


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    event_id: str
    league: str
    prediction_cutoff: str
    prediction_created_at: str
    home_team: str
    away_team: str
    home_starter: str | None
    away_starter: str | None
    probabilities: dict[str, float]
    score_candidates: list[dict[str, Any]]
    low_probability: float | None
    high_probability: float | None
    total_runs_line: float | None
    confidence: float | None
    volatility: float | None
    model_version: str
    feature_version: str
    calibration_version: str
    git_commit: str
    data_snapshot_id: str
    eligibility: str = "ELIGIBLE"


def make_prediction_id(event_id: str, cutoff: str, model_version: str, git_commit: str) -> str:
    return hashlib.sha256(f"{event_id}|{cutoff}|{model_version}|{git_commit}".encode()).hexdigest()[:24]


def validate_prediction(record: PredictionRecord) -> None:
    if record.league not in {"NPB", "MLB"}:
        raise ValueError("league must be NPB or MLB")
    if not record.event_id or not record.home_team or not record.away_team:
        raise ValueError("event and team identifiers are required")
    if record.eligibility != "ELIGIBLE":
        raise ValueError("only eligible predictions may enter the production ledger")
    if record.home_starter is None or record.away_starter is None:
        raise ValueError("both starting pitchers must be confirmed")

    cutoff = _dt(record.prediction_cutoff)
    created = _dt(record.prediction_created_at)
    if created < cutoff:
        raise ValueError("prediction_created_at cannot precede prediction_cutoff")

    required = {"home", "away"} | ({"draw"} if record.league == "NPB" else set())
    if set(record.probabilities) != required:
        raise ValueError("probability contract does not match league")
    probabilities = {k: _finite_probability(v) for k, v in record.probabilities.items()}
    if abs(sum(probabilities.values()) - 1.0) > 1e-8:
        raise ValueError("prediction probabilities must sum to 1")

    for name, value in (("low_probability", record.low_probability),
                        ("high_probability", record.high_probability),
                        ("confidence", record.confidence),
                        ("volatility", record.volatility)):
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0 or float(value) > 1.0):
            raise ValueError(f"{name} must be finite and in [0,1]")
    if (record.low_probability is None) != (record.high_probability is None):
        raise ValueError("low_probability and high_probability must be supplied together")
    if record.low_probability is not None and abs(float(record.low_probability) + float(record.high_probability) - 1.0) > 1e-8:
        raise ValueError("Low/High probabilities must sum to 1")

    if record.total_runs_line is not None:
        line = float(record.total_runs_line)
        if not math.isfinite(line) or line < 0 or abs(line * 2 - round(line * 2)) > 1e-9:
            raise ValueError("total_runs_line must be a finite non-negative integer or half-point")

    if not record.model_version or not record.feature_version or not record.calibration_version:
        raise ValueError("model, feature, and calibration versions are required")
    if not record.git_commit or not record.data_snapshot_id:
        raise ValueError("git_commit and data_snapshot_id are required for auditability")


def append_prediction(record: PredictionRecord, path: str | Path) -> None:
    """Append exactly once by prediction_id; conflicting duplicate IDs fail closed."""
    validate_prediction(record)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)
    if p.exists():
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                existing = json.loads(line)
                if existing.get("prediction_id") == record.prediction_id:
                    if line.rstrip("\n") != serialized:
                        raise ValueError("prediction_id collision with different record")
                    return
    with p.open("a", encoding="utf-8") as fh:
        fh.write(serialized + "\n")


def record_from_mapping(row: Mapping[str, Any]) -> PredictionRecord:
    record = PredictionRecord(**dict(row))
    validate_prediction(record)
    return record
