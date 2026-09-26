"""Universal source-to-scope application and PIT gate.

The purpose is to make the collected research sources applicable across every
known competition scope without pretending that every source is available for
every competition. Features are additive: missing/unknown information is kept
missing, never converted to zero.

This module is research-safe and does not alter production model selection.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

import pandas as pd

from data.source_registry import SOURCES
from research.competition_catalog import get_scope, scopes


CANONICAL_FEATURES = (
    "schedule_identity",
    "starting_pitchers",
    "lineups",
    "roster",
    "batting",
    "pitching",
    "fielding",
    "bullpen",
    "play_by_play",
    "tracking",
    "weather",
    "market",
    "news_context",
    "tournament_rules",
)

# Source-level capabilities are intentionally conservative. A capability means
# "the source can potentially provide this signal", not "the source has been
# verified PIT-safe for historical OOS".
SOURCE_CAPABILITIES: dict[str, frozenset[str]] = {
    "npb_schedule": frozenset({"schedule_identity"}),
    "npb_starters": frozenset({"starting_pitchers"}),
    "npb_lineups": frozenset({"lineups"}),
    "npb_hawkeye_npbplus": frozenset({"tracking", "pitching", "batting", "fielding"}),
    "npb_public_spaia_pbp": frozenset({"play_by_play", "pitching", "batting", "lineups"}),
    "armstjc_npb_repository": frozenset({"schedule_identity", "play_by_play", "batting", "pitching"}),
    "wocchi09_npb_data": frozenset({"play_by_play", "tracking", "pitching", "batting"}),
    "mlb_statsapi": frozenset({"schedule_identity", "roster", "batting", "pitching", "fielding"}),
    "mlb_starters": frozenset({"starting_pitchers"}),
    "mlb_milb_statcast": frozenset({"tracking", "pitching", "batting"}),
    "ncaa_baseball_sportsdataverse": frozenset({"schedule_identity", "play_by_play", "batting", "pitching"}),
    "retrosheet": frozenset({"schedule_identity", "play_by_play", "batting", "pitching", "fielding"}),
    "kbo_official_tracking": frozenset({"tracking", "pitching", "batting"}),
    "kbo_official_stats": frozenset({"schedule_identity", "batting", "pitching", "fielding"}),
    "kbo_naver_pbp_public": frozenset({"play_by_play", "tracking", "pitching", "batting"}),
    "cpbl_rebas": frozenset({"schedule_identity", "play_by_play", "batting", "pitching", "fielding"}),
    "lmb_official": frozenset({"schedule_identity", "batting", "pitching"}),
    "abl_official": frozenset({"schedule_identity", "batting", "pitching"}),
    "wbsc_mywbsc": frozenset({"schedule_identity", "play_by_play", "batting", "pitching", "roster"}),
    "omyu_high_school": frozenset({"schedule_identity", "play_by_play", "batting", "pitching", "lineups"}),
    "omyu_junior": frozenset({"schedule_identity", "play_by_play", "batting", "pitching", "lineups"}),
    "omyu_little_senior": frozenset({"schedule_identity", "play_by_play", "batting", "pitching"}),
    "omyu_boys": frozenset({"schedule_identity", "play_by_play", "batting", "pitching"}),
    "omyu_young": frozenset({"schedule_identity", "play_by_play", "batting", "pitching"}),
    "omyu_pony": frozenset({"schedule_identity", "play_by_play", "batting", "pitching"}),
    "omyu_youth_girls": frozenset({"schedule_identity", "play_by_play", "batting", "pitching", "lineups"}),
    "omyu_university": frozenset({"schedule_identity", "play_by_play", "batting", "pitching", "lineups"}),
    "omyu_jaba": frozenset({"schedule_identity", "play_by_play", "batting", "pitching", "lineups"}),
    "jaba_official": frozenset({"schedule_identity", "batting", "pitching", "tournament_rules"}),
    "big6_scorebook": frozenset({"schedule_identity", "batting", "pitching"}),

    "omyu_elementary": frozenset({"schedule_identity", "play_by_play", "batting", "pitching"}),
    "omyu_womens_high_school": frozenset({"schedule_identity", "play_by_play", "batting", "pitching", "lineups"}),
    "omyu_womens_junior": frozenset({"schedule_identity", "play_by_play", "batting", "pitching", "lineups"}),
    "omyu_independent": frozenset({"schedule_identity", "play_by_play", "batting", "pitching", "lineups"}),
    "iblj_official_stats": frozenset({"schedule_identity", "batting", "pitching"}),
    "bcl_official_stats": frozenset({"schedule_identity", "batting", "pitching"}),
    "yahoo_ipbl_stats": frozenset({"schedule_identity", "batting", "pitching"}),
    "lidom_mlb_winter": frozenset({"schedule_identity", "batting", "pitching"}),
    "lvbp_official": frozenset({"schedule_identity", "batting", "pitching"}),
    "lbprc_official": frozenset({"schedule_identity", "batting", "pitching"}),
    "lmp_mlb_winter": frozenset({"schedule_identity", "batting", "pitching"}),
    "wbc_official_stats": frozenset({"schedule_identity", "batting", "pitching", "roster"}),
    "wbsc_international_events": frozenset({"schedule_identity", "batting", "pitching", "roster", "tournament_rules"}),
    "wbsc_womens_baseball": frozenset({"schedule_identity", "batting", "pitching", "roster"}),
    "little_league_world_series": frozenset({"schedule_identity", "batting", "pitching", "roster", "tournament_rules"}),
    "cape_cod_league": frozenset({"schedule_identity", "batting", "pitching", "roster"}),
    "wbsc_europe_baseball": frozenset({"schedule_identity", "batting", "pitching", "roster", "tournament_rules"}),
    "knbsb_baseball": frozenset({"schedule_identity", "batting", "pitching", "tournament_rules"}),
    "openbiomechanics": frozenset({"tracking", "pitching", "batting"}),

    "jhbf_official": frozenset({"schedule_identity", "tournament_rules"}),
    "samurai_youth": frozenset({"schedule_identity", "roster", "batting", "pitching", "tournament_rules"}),
    "samurai_u23": frozenset({"schedule_identity", "roster", "batting", "pitching", "tournament_rules"}),
    "wbsc_age_group_reports": frozenset({"schedule_identity", "batting", "pitching", "tournament_rules"}),
    "statcast": frozenset({"tracking", "pitching", "batting", "fielding"}),
    "fangraphs": frozenset({"batting", "pitching", "fielding"}),
    "weather": frozenset({"weather"}),
    "x_api_recent_search": frozenset({"news_context"}),
    "lahman": frozenset({"schedule_identity", "batting", "pitching", "fielding"}),
}

_SHARED_SOURCES = {
    "weather",
}


def source_capabilities(source_id: str) -> frozenset[str]:
    return SOURCE_CAPABILITIES.get(source_id, frozenset())


def application_plan(scope_id: str) -> list[dict[str, Any]]:
    """Return all registered sources intentionally applicable to one scope."""
    scope = get_scope(scope_id)
    allowed = set(scope.source_ids) | _SHARED_SOURCES
    registry_ids = {s.source_id for s in SOURCES}
    rows: list[dict[str, Any]] = []
    for source_id in sorted(allowed):
        if source_id not in registry_ids:
            continue
        features = sorted(source_capabilities(source_id) & set(CANONICAL_FEATURES))
        if not features:
            continue
        rows.append({
            "scope_id": scope.scope_id,
            "source_id": source_id,
            "features": features,
            "outcome_contract": scope.outcome_contract,
            "rule_family": scope.rule_family,
            "pit_requirement": "explicit available_at <= prediction_time",
        })
    return rows


def application_matrix() -> list[dict[str, Any]]:
    """Build the complete scope × source application matrix."""
    rows: list[dict[str, Any]] = []
    for scope in scopes():
        rows.extend(application_plan(scope.scope_id))
    return rows


def pit_safe_rows(
    frame: pd.DataFrame,
    *,
    available_at_col: str = "available_at",
    prediction_time_col: str = "prediction_time",
    status_col: str | None = "status",
) -> pd.DataFrame:
    """Fail-closed filter for any collected feature table.

    Rows without both explicit timestamps are rejected. Unknown status is also
    rejected when a status column is supplied. No missing feature is imputed.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    required = {available_at_col, prediction_time_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"PIT columns missing: {sorted(missing)}")

    out = frame.copy()
    out[available_at_col] = pd.to_datetime(out[available_at_col], errors="coerce", utc=True)
    out[prediction_time_col] = pd.to_datetime(out[prediction_time_col], errors="coerce", utc=True)
    mask = out[available_at_col].notna() & out[prediction_time_col].notna()
    mask &= out[available_at_col] <= out[prediction_time_col]
    if status_col and status_col in out.columns:
        mask &= out[status_col].astype(str).str.upper().eq("KNOWN")
    return out.loc[mask].copy()


def source_snapshot(source_id: str, *, observed_at: datetime | str | None = None) -> Mapping[str, Any]:
    """Return a provenance record template for a collected source."""
    if source_id not in {s.source_id for s in SOURCES}:
        raise KeyError(f"unknown source_id: {source_id}")
    observed = pd.Timestamp(observed_at, tz="UTC") if observed_at is not None else pd.Timestamp.utcnow()
    return {
        "source_id": source_id,
        "observed_at": observed.isoformat(),
        "features": sorted(source_capabilities(source_id)),
        "pit_policy": "available_at <= prediction_time; missing timestamps => FAIL CLOSED",
    }
