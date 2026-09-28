import pandas as pd
import math
import pytest

from data.mlb_statcast import (
    build_game_level_features,
    build_lagged_team_features,
)


def _sample():
    return pd.DataFrame([
        {
            "game_date": "2026-09-20",
            "game_pk": 1,
            "home_team": "AAA",
            "away_team": "BBB",
            "pitcher": 10,
            "batter": 20,
            "release_speed": 95,
            "release_spin": 2400,
            "pfx_x": 8,
            "pfx_z": 14,
            "launch_speed": 101,
            "launch_angle": 20,
            "hit_distance": 390,
            "launch_speed_angle": 6,
        },
        {
            "game_date": "2026-09-22",
            "game_pk": 2,
            "home_team": "BBB",
            "away_team": "AAA",
            "pitcher": 11,
            "batter": 21,
            "release_speed": 92,
            "release_spin": 2300,
            "pfx_x": 7,
            "pfx_z": 13,
            "launch_speed": 80,
            "launch_angle": 5,
            "hit_distance": 200,
            "launch_speed_angle": 1,
        },
    ])


def test_statcast_game_aggregation():
    out = build_game_level_features(_sample())
    assert len(out) == 2
    assert out.loc[0, "home_pitch_velocity_mean"] == 95
    assert out.loc[0, "home_hard_hit_rate"] == 1
    assert out.loc[0, "home_barrel_rate"] == 1
    assert out.loc[0, "home_pitch_type_count"] != 0


def test_statcast_lagged_features_use_prior_games_only():
    games = build_game_level_features(_sample())
    out = build_lagged_team_features(games)
    assert set(out["tracking_pit_status"]) == {"UNVERIFIED"}
    assert set(out["historical_backtest_eligible"]) == {False}
    # AAA appears in game 1 before game 2, so game 2 away lag can use game 1.
    row2 = out[out["game_pk"] == "2"].iloc[0]
    assert row2["away_lag_pitch_velocity_mean"] == 95


def test_lagged_features_differ_by_team_side():
    games = build_game_level_features(_sample())
    out = build_lagged_team_features(games)
    row1 = out[out["game_pk"] == "1"].iloc[0]
    assert math.isnan(row1["home_lag_pitch_velocity_mean"])
    assert math.isnan(row1["away_lag_pitch_velocity_mean"])

def test_missing_statcast_measurements_are_not_encoded_as_zero():
    frame = _sample().copy()
    frame.loc[0, "release_spin"] = None
    frame.loc[0, "launch_speed"] = None
    out = build_game_level_features(frame)
    assert pd.isna(out.loc[0, "home_spin_rate_mean"])
    assert pd.isna(out.loc[0, "home_exit_velocity_mean"])
