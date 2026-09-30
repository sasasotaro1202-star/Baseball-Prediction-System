"""Competition-aware prediction policy for baseball.

This layer is metadata and policy only. It never grants production eligibility.
Competition-specific calibration/routing is enabled only when chronological
validation evidence reaches the configured minimum sample size; otherwise the
prediction path falls back to the league-level policy.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompetitionStrategy:
    competition_key: str
    competition_type: str
    stage: str
    strategy_id: str
    calibration_id: str
    specialist_min_validation_rows: int
    status: str = "RESEARCH"


def strategy_for(
    competition_key: str,
    *,
    competition_type: str = "unknown",
    stage: str = "unknown",
) -> CompetitionStrategy:
    key = str(competition_key or "baseball:unknown")
    ctype = str(competition_type or "unknown")
    stg = str(stage or "unknown")

    if stg in {"regular_season", "interleague"}:
        strategy_id = "league_adaptive_ensemble"
        calibration_id = "competition_temperature" if stg == "interleague" else "competition_temperature"
        minimum = 40
    elif stg in {"climax_series", "division_series", "league_championship_series", "wild_card", "japan_series", "postseason"}:
        strategy_id = "postseason_shrunk_ensemble"
        calibration_id = "competition_temperature"
        minimum = 50
    elif ctype in {"tournament", "qualifier", "international", "international_club"} or stg in {"tournament", "qualifier", "championship"}:
        strategy_id = "tournament_shrunk_ensemble"
        calibration_id = "competition_temperature"
        minimum = 60
    elif stg in {"all_star", "spring"} or ctype in {"exhibition", "friendly"}:
        strategy_id = "exhibition_fallback"
        calibration_id = "league_temperature_only"
        minimum = 999999
    elif key.endswith(":unknown") or stg == "unknown":
        strategy_id = "unknown_fail_closed"
        calibration_id = "league_temperature_only"
        minimum = 999999
    else:
        strategy_id = "competition_shrunk_ensemble"
        calibration_id = "competition_temperature"
        minimum = 60

    return CompetitionStrategy(
        competition_key=key,
        competition_type=ctype,
        stage=stg,
        strategy_id=strategy_id,
        calibration_id=calibration_id,
        specialist_min_validation_rows=minimum,
    )


def eligible_for_competition_calibration(
    strategy: CompetitionStrategy,
    validation_rows: int,
) -> bool:
    rows = int(validation_rows)
    return rows >= strategy.specialist_min_validation_rows and strategy.calibration_id == "competition_temperature"
