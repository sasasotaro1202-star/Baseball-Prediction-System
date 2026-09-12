"""Stage registry for incremental feature addition.

Each stage is a pure function: (df) -> df_with_added_columns. New stages
should ONLY use data available strictly before the game being predicted
(no target-game result, no future lineup/odds/weather), per the
leakage-prevention rules in prediction_model_specification.md section 29.
"""
import pandas as pd

from parallel.feature_engineering import (
    add_rest_days, add_recent_form, add_h2h, add_shot_efficiency, apply_all,
)


def stage_elo_only(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline stage -- no-op placeholder documenting Stage-0 scope."""
    return df


def stage_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    return add_rest_days(df)


def stage_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    return add_recent_form(df, windows=(5, 10))


def stage_h2h(df: pd.DataFrame) -> pd.DataFrame:
    return add_h2h(df, max_meetings=8)


def stage_shot_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    return add_shot_efficiency(df)


def stage_all_available(df: pd.DataFrame) -> pd.DataFrame:
    """Applies every feature stage that is implementable with currently
    acquired data (rest days, recent form, H2H, shot efficiency)."""
    return apply_all(df, windows=(5, 10), max_h2h=8)


# Future stages (require data not yet acquired -- see
# prediction_model_specification.md for the full list):
#   stage_bullpen(df)            -> bullpen workload/fatigue (MLB)
#   stage_starting_pitcher(df)   -> starter history & matchup (MLB)
#   stage_market_odds(df)        -> closing odds / implied probability
#   stage_xg(df)                 -> Understat xG/xGA (soccer)
#   stage_weather(df)            -> park/venue weather adjustment (MLB)
#   stage_sofascore_players(df)  -> player-level attack/defense/passing (soccer)

STAGE_REGISTRY = {
    "elo_only": stage_elo_only,
    "rest_days": stage_rest_days,
    "recent_form": stage_recent_form,
    "h2h": stage_h2h,
    "shot_efficiency": stage_shot_efficiency,
    "all_available": stage_all_available,
}


def apply_stages(df: pd.DataFrame, stage_names: list) -> pd.DataFrame:
    for name in stage_names:
        if name not in STAGE_REGISTRY:
            raise KeyError(f"Unknown feature stage: {name}")
        df = STAGE_REGISTRY[name](df)
    return df
