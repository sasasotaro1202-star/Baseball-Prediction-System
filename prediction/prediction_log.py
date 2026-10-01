"""Immutable prediction ledger for Baseball production outputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
from contextlib import contextmanager

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows/local fallback
    fcntl = None


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


def _validate_score_candidates(candidates: Any) -> None:
    """Enforce the immutable production contract for displayed exact scores."""
    if not isinstance(candidates, list) or len(candidates) != 4:
        raise ValueError("production score contract requires exactly four candidates")
    seen: set[str] = set()
    for item in candidates:
        if not isinstance(item, Mapping) or "score" not in item or "probability" not in item:
            raise ValueError("each score candidate requires score and probability")
        score = str(item["score"])
        if score in seen:
            raise ValueError("score candidates must be unique")
        seen.add(score)
        _finite_probability(item["probability"])


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
    competition_key: str | None = None
    competition_stage: str | None = None
    prediction_set: list[str] | None = None
    prediction_set_alpha: float | None = None
    prediction_set_method: str | None = None
    prediction_set_action: str | None = None
    prediction_intelligence: dict[str, Any] | None = None


def _validate_prediction_set(record: PredictionRecord, expected: set[str]) -> None:
    values = record.prediction_set
    related = (
        record.prediction_set_alpha,
        record.prediction_set_method,
        record.prediction_set_action,
    )
    if values is None:
        if any(v is not None for v in related):
            raise ValueError("prediction-set metadata requires prediction_set")
        return
    if not values or len(values) != len(set(values)):
        raise ValueError("prediction_set must be a non-empty list of unique labels")
    if not set(values).issubset(expected):
        raise ValueError("prediction_set contains an unknown class")
    if record.prediction_set_alpha is None or not 0.0 < float(record.prediction_set_alpha) < 1.0:
        raise ValueError("prediction_set_alpha must be in (0,1)")
    if record.prediction_set_method not in {"split_conformal", "group_split_conformal"}:
        raise ValueError("unsupported prediction_set_method")
    if record.prediction_set_action not in {"SINGLE", "SET", "ABSTAIN"}:
        raise ValueError("invalid prediction_set_action")
    expected_action = "SINGLE" if len(values) == 1 else "SET"
    if record.prediction_set_action != expected_action:
        raise ValueError("prediction_set_action does not match prediction_set size")


def _validate_prediction_intelligence(record: PredictionRecord) -> None:
    value = record.prediction_intelligence
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ValueError("prediction_intelligence must be a mapping")
    required = {"prediction_time", "pit_status", "provenance"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError("prediction_intelligence missing required keys: " + ",".join(missing))
    if str(value["prediction_time"]) != str(record.prediction_cutoff):
        raise ValueError("prediction_intelligence prediction_time must equal prediction_cutoff")
    if str(value["pit_status"]) != "PASS":
        raise ValueError("prediction_intelligence pit_status must be PASS")
    if not isinstance(value["provenance"], Mapping) or not value["provenance"]:
        raise ValueError("prediction_intelligence provenance must be a non-empty mapping")
    bounded = (
        "confidence",
        "predictability",
        "uncertainty",
        "disagreement",
        "ood",
        "failure_risk",
        "freshness",
        "information_value",
    )
    for name in bounded:
        if name in value:
            x = float(value[name])
            if not math.isfinite(x) or x < 0.0 or x > 1.0:
                raise ValueError(f"prediction_intelligence {name} must be in [0,1]")
    if "forecast_lifetime" in value:
        x = float(value["forecast_lifetime"])
        if not math.isfinite(x) or x < 0.0:
            raise ValueError("prediction_intelligence forecast_lifetime must be finite and non-negative")
    for name in ("action", "update_need"):
        if name in value and not str(value[name]).strip():
            raise ValueError(f"prediction_intelligence {name} must be non-empty")
    if "valid_until" in value:
        valid_until = _dt(str(value["valid_until"]))
        if valid_until < _dt(record.prediction_cutoff):
            raise ValueError("prediction_intelligence valid_until precedes prediction_cutoff")


def make_prediction_id(event_id: str, cutoff: str, model_version: str, git_commit: str) -> str:
    return hashlib.sha256(f"{event_id}|{cutoff}|{model_version}|{git_commit}".encode()).hexdigest()[:24]


def validate_prediction(record: PredictionRecord) -> None:
    if record.league not in {"NPB", "MLB"} and not record.competition_key:
        raise ValueError("league/competition identity is required")
    if record.competition_key is not None and not str(record.competition_key).strip():
        raise ValueError("competition_key must be non-empty when supplied")
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

    _validate_score_candidates(record.score_candidates)
    _validate_prediction_set(record, required)
    _validate_prediction_intelligence(record)

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


def _ledger_lock(fh):
    @contextmanager
    def _ctx():
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return _ctx()


def append_prediction(record: PredictionRecord, path: str | Path) -> None:
    """Append exactly once by prediction_id; serialize concurrent writers."""
    validate_prediction(record)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)
    with p.open("a+", encoding="utf-8") as fh:
        with _ledger_lock(fh):
            fh.seek(0)
            for line in fh:
                if not line.strip():
                    continue
                existing = json.loads(line)
                if existing.get("prediction_id") == record.prediction_id:
                    if line.rstrip("\n") != serialized:
                        raise ValueError("prediction_id collision with different record")
                    return
            fh.seek(0, 2)
            fh.write(serialized + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def record_from_mapping(row: Mapping[str, Any]) -> PredictionRecord:
    record = PredictionRecord(**dict(row))
    validate_prediction(record)
    return record
