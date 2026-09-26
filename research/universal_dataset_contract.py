"""Universal baseball research dataset contract.

Every competition-specific collector should normalize into this contract before
features reach OOS or model selection. This is deliberately research-first:
unknown provenance/PIT is rejected instead of guessed.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from data.source_registry import SOURCES
from research.competition_catalog import get_scope
from research.universal_source_matrix import application_plan, pit_safe_rows, source_capabilities


IDENTITY_COLUMNS = (
    "event_id",
    "scope_id",
    "source_id",
    "event_time",
    "prediction_time",
    "available_at",
)

PROVENANCE_COLUMNS = ("source_id", "available_at", "prediction_time")
FEATURE_COLUMNS = (
    "schedule_identity", "starting_pitchers", "lineups", "roster", "batting",
    "pitching", "fielding", "bullpen", "play_by_play", "tracking", "weather",
    "market", "news_context", "tournament_rules",
)


class UniversalContractError(ValueError):
    """Raised when a source cannot enter the universal research contract."""


def validate_source_id(source_id: str) -> None:
    if source_id not in {s.source_id for s in SOURCES}:
        raise UniversalContractError(f"unknown source_id: {source_id}")


def normalize_research_frame(
    frame: pd.DataFrame,
    *,
    scope_id: str,
    source_id: str,
    status_col: str | None = "status",
) -> pd.DataFrame:
    """Normalize and fail-closed a collected table for one competition scope."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise UniversalContractError("frame must be a non-empty DataFrame")
    scope = get_scope(scope_id)
    validate_source_id(source_id)
    applicable = {row["source_id"] for row in application_plan(scope_id)}
    if source_id not in applicable:
        raise UniversalContractError(
            f"source {source_id} is not registered as applicable to scope {scope_id}"
        )

    missing = {"event_id", "event_time", "prediction_time", "available_at"} - set(frame.columns)
    if missing:
        raise UniversalContractError(f"missing universal columns: {sorted(missing)}")

    out = frame.copy()
    out["scope_id"] = scope.scope_id
    out["source_id"] = source_id
    for col in ("event_time", "prediction_time", "available_at"):
        out[col] = pd.to_datetime(out[col], errors="coerce", utc=True)

    if out[["event_id", "event_time", "prediction_time", "available_at"]].isna().any().any():
        raise UniversalContractError("identity/PIT timestamps contain missing values")

    if status_col and status_col not in out.columns:
        raise UniversalContractError(f"{status_col} is required for fail-closed research ingestion")

    if status_col:
        out = pit_safe_rows(
            out,
            available_at_col="available_at",
            prediction_time_col="prediction_time",
            status_col=status_col,
        )

    if out.empty:
        raise UniversalContractError("no PIT-safe rows remain after fail-closed filtering")

    out["competition_level"] = scope.level
    out["competition_gender"] = scope.gender
    out["outcome_contract"] = scope.outcome_contract
    out["rule_family"] = scope.rule_family
    out["source_capability_contract"] = [",".join(sorted(source_capabilities(source_id)))] * len(out)
    return out


def research_manifest(frame: pd.DataFrame, *, scope_id: str, source_id: str) -> dict[str, Any]:
    """Return an auditable manifest after normalization."""
    normalized = normalize_research_frame(frame, scope_id=scope_id, source_id=source_id)
    scope = get_scope(scope_id)
    return {
        "scope_id": scope.scope_id,
        "scope_label": scope.label,
        "level": scope.level,
        "gender": scope.gender,
        "outcome_contract": scope.outcome_contract,
        "rule_family": scope.rule_family,
        "source_id": source_id,
        "rows": int(len(normalized)),
        "events": int(normalized["event_id"].nunique()),
        "pit_status": "PASS",
        "required_pit_rule": "available_at <= prediction_time",
        "feature_columns_present": sorted(set(FEATURE_COLUMNS) & set(normalized.columns)),
    }
