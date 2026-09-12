"""Candidate data contract and Development-OOS-only lock artifact."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    league: str
    objective: str
    model_version: str
    feature_version: str
    development_metrics: dict[str, float]
    selection_reason: str
    git_commit: str
    dataset_hash: str


def lock_candidate(spec: CandidateSpec) -> dict[str, Any]:
    """Persist the Candidate Lock before any independent holdout is opened."""
    if not spec.candidate_id or not spec.model_version:
        raise ValueError("candidate_id and model_version are required")
    locked_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": 1,
        "stage": "candidate_locked",
        "locked_at": locked_at,
        "candidate": asdict(spec),
        "holdout_evaluated": False,
        "holdout_access": "forbidden_during_selection",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{spec.league.lower()}_candidate_lock.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload
