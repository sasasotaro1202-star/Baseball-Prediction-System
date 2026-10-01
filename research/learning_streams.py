"""Audit the two learning streams without unsafe target mixing.

Historical chronological OOS is the model/strategy learning stream.
Matured production Experience is the postgame feedback/calibration stream.
Both remain independently auditable and PIT-constrained; this module only
builds a manifest and never mutates production models.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXPERIENCE = ROOT / "data" / "experience" / "experience_ledger.csv"
OOS_FILES = {
    "NPB": RESULTS / "checkpoints" / "npb_walkforward.csv",
    "MLB": RESULTS / "checkpoints" / "mlb_walkforward.csv",
}


def _load_experience() -> dict[str, Any]:
    if not EXPERIENCE.exists() or EXPERIENCE.stat().st_size == 0:
        return {"status": "NO_DATA", "rows": 0}
    frame = pd.read_csv(EXPERIENCE)
    required = {"prediction_id", "prediction_cutoff_utc", "experience_available_at_utc", "actual_outcome"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"experience ledger missing required fields: {missing}")
    cutoff = pd.to_datetime(frame["prediction_cutoff_utc"], errors="coerce", utc=True)
    available = pd.to_datetime(frame["experience_available_at_utc"], errors="coerce", utc=True)
    if cutoff.isna().any() or available.isna().any():
        raise RuntimeError("experience ledger contains invalid timestamps")
    if (available < cutoff).any():
        raise RuntimeError("experience ledger contains impossible availability timestamps")
    return {
        "status": "READY",
        "rows": int(len(frame)),
        "earliest_prediction_cutoff_utc": cutoff.min().isoformat(),
        "latest_prediction_cutoff_utc": cutoff.max().isoformat(),
        "latest_experience_available_at_utc": available.max().isoformat(),
        "pit_rows": int((available >= cutoff).sum()),
    }


def _load_oos(league: str) -> dict[str, Any]:
    path = OOS_FILES[league]
    if not path.exists() or path.stat().st_size == 0:
        return {"status": "NO_DATA", "rows": 0, "path": str(path)}
    frame = pd.read_csv(path)
    required = {"game_id", "datetime", "correct", "logloss", "brier"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{league} OOS artifact missing required fields: {missing}")
    dt = pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
    if dt.isna().any():
        raise RuntimeError(f"{league} OOS artifact contains invalid chronology")
    if frame["game_id"].astype(str).duplicated().any():
        raise RuntimeError(f"{league} OOS artifact contains duplicate game IDs")
    return {
        "status": "READY",
        "rows": int(len(frame)),
        "earliest_game_utc": dt.min().isoformat(),
        "latest_game_utc": dt.max().isoformat(),
        "accuracy": float(pd.to_numeric(frame["correct"], errors="coerce").mean()),
        "logloss": float(pd.to_numeric(frame["logloss"], errors="coerce").mean()),
        "brier": float(pd.to_numeric(frame["brier"], errors="coerce").mean()),
        "path": str(path),
    }


def build_learning_manifest() -> dict[str, Any]:
    historical = {league: _load_oos(league) for league in OOS_FILES}
    experience = _load_experience()
    return {
        "schema_version": "baseball-learning-streams-v1",
        "status": "EXECUTED",
        "streams": {
            "historical_chronological_oos": historical,
            "matured_production_experience": experience,
        },
        "separation_contract": {
            "historical_oos_is_model_strategy_learning": True,
            "experience_is_postgame_feedback_calibration_learning": True,
            "future_outcomes_into_pregame_features": False,
            "current_prediction_outcome_reuse": False,
            "production_model_modified": False,
            "auto_promotion": False,
        },
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = build_learning_manifest()
    path = RESULTS / "learning_streams_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
