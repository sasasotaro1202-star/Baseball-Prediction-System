"""Minimal data contract for PIT-aware Baseball research tables.

The contract is additive and deliberately does not replace the existing
NPB/MLB schemas. It gives the research layer stable names for provenance,
event identity and target fields while preserving domain-specific columns.
"""
from __future__ import annotations

BASE_COLUMNS = (
    "event_id",
    "event_time",
    "source",
    "source_timestamp",
    "available_ts",
)

BASEBALL_EVENT_COLUMNS = (
    "team_a",
    "team_b",
    "home_team",
    "away_team",
)

TARGET_COLUMNS = (
    "target",
    "result",
    "winner",
    "home_win",
    "away_win",
    "runs_home",
    "runs_away",
)


def missing_columns(df, columns=BASE_COLUMNS) -> list[str]:
    return [c for c in columns if c not in df.columns]


def validate_base_contract(df, *, require_source=True) -> None:
    required = ["event_id", "event_time"]
    if require_source:
        required += ["source"]
    missing = missing_columns(df, required)
    if missing:
        raise ValueError(f"Baseball data contract missing columns: {missing}")


def attach_event_ids(df, *, id_candidates=("event_id", "game_id", "gamePk", "match_id")):
    """Create event_id only from an existing stable identifier; never fabricate one."""
    out = df.copy()
    if "event_id" in out.columns:
        return out
    for candidate in id_candidates:
        if candidate in out.columns:
            out["event_id"] = out[candidate].astype("string")
            return out
    raise ValueError("No stable existing event identifier found; event_id was not fabricated")
