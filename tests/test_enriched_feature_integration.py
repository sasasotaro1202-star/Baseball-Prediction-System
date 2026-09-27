import json
from pathlib import Path

import pandas as pd
import pytest

from baseball_backtest import BaseballBacktest
from data.npb_enrichment_contract import ENRICHED_GAME_FIELDS
from data.npb_pbp_adapter import normalize_pbp_frame


def test_enriched_collector_fields_survive_normalization():
    raw = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "game_date": "2026-09-01T09:00:00Z",
                "home_team_name": "A",
                "away_team_name": "B",
                "home_total_runs": 5,
                "away_total_runs": 2,
                "home_bat_pa": 38,
                "home_bat_ab": 34,
                "home_bat_h": 11,
                "home_bat_hr": 2,
                "away_bat_pa": 35,
                "away_bat_ab": 32,
                "away_bat_h": 7,
                "away_bat_hr": 1,
                "home_starter_era": 2.75,
                "away_starter_era": 4.25,
                "home_starter_fip": 3.05,
                "away_starter_fip": 4.55,
                "home_bullpen_ip": 3.0,
                "away_bullpen_ip": 2.0,
                "home_lineup_json": json.dumps([{"player_id": "p1"}]),
                "away_lineup_json": json.dumps([{"player_id": "p2"}]),
                "weather_temp_c": 23.0,
            }
        ]
    )
    normalized = normalize_pbp_frame(raw)

    for column in (
        "home_bat_pa",
        "away_bat_h",
        "home_starter_era",
        "away_starter_fip",
        "home_bullpen_ip",
        "home_lineup_json",
        "weather_temp_c",
    ):
        assert column in normalized.columns
    assert normalized.loc[0, "home_bat_h"] == pytest.approx(11)
    assert normalized.loc[0, "home_starter_era"] == pytest.approx(2.75)

    aggregated = BaseballBacktest(Path("data")).aggregate_npb_games(normalized)
    assert aggregated.loc[0, "home_bullpen_ip"] == pytest.approx(3.0)
    assert aggregated.loc[0, "away_bullpen_ip"] == pytest.approx(2.0)


def test_all_declared_enriched_fields_survive_normalization():
    raw = pd.DataFrame([{
        "game_id": "g_contract",
        "game_date": "2026-09-01T09:00:00Z",
        "home_team_name": "A",
        "away_team_name": "B",
        "home_total_runs": 4,
        "away_total_runs": 3,
        **{
            field: (
                json.dumps([{"player_id": "p1"}])
                if field.endswith("_lineup_json")
                else 1.0
            )
            for field in ENRICHED_GAME_FIELDS
            if field not in {"league"}
        },
    }])
    normalized = normalize_pbp_frame(raw)
    missing = [field for field in ENRICHED_GAME_FIELDS if field not in normalized.columns]
    assert not missing, f"declared enriched fields dropped: {missing}"


def test_realized_starter_identity_never_becomes_target_time_identity():
    raw = pd.DataFrame([{
        "game_id": "g_hindsight",
        "game_date": "2026-09-01T09:00:00Z",
        "home_team_name": "A",
        "away_team_name": "B",
        "home_total_runs": 2,
        "away_total_runs": 1,
        "home_starter": "HINDSIGHT_HOME",
        "away_starter": "HINDSIGHT_AWAY",
    }])
    normalized = normalize_pbp_frame(raw)
    aggregated = BaseballBacktest(Path("data")).aggregate_npb_games(normalized)
    assert aggregated.loc[0, "home_starter"] == ""
    assert aggregated.loc[0, "away_starter"] == ""
    assert not bool(aggregated.loc[0, "confirmed_starters"])


def test_lagged_historical_batting_and_starter_quality_reach_feature_matrix():
    bt = BaseballBacktest(Path("data"))
    games = pd.DataFrame(
        [
            {
                "league": "NPB",
                "game_id": "g1",
                "datetime": pd.Timestamp("2026-09-01T09:00:00Z"),
                "home": "A",
                "away": "B",
                "home_score": 5,
                "away_score": 2,
                "home_starter": "",
                "away_starter": "",
                "home_bat_pa": 40,
                "home_bat_ab": 36,
                "home_bat_h": 18,
                "home_bat_hr": 2,
                "home_bat_bb": 3,
                "home_bat_so": 8,
                "home_bat_2b": 4,
                "home_bat_3b": 0,
                "home_bat_sb": 1,
                "home_bat_cs": 0,
                "away_bat_pa": 36,
                "away_bat_ab": 33,
                "away_bat_h": 9,
                "away_bat_hr": 1,
                "away_bat_bb": 2,
                "away_bat_so": 9,
                "away_bat_2b": 2,
                "away_bat_3b": 0,
                "away_bat_sb": 0,
                "away_bat_cs": 0,
                "home_starter_era": 2.80,
                "away_starter_era": 4.60,
                "home_starter_fip": 3.00,
                "away_starter_fip": 4.80,
                "home_starter_k9": 9.0,
                "away_starter_k9": 6.2,
                "home_starter_bb9": 2.1,
                "away_starter_bb9": 3.7,
                "home_starter_hr9": 0.8,
                "away_starter_hr9": 1.4,
                "home_starter_whip": 1.05,
                "away_starter_whip": 1.42,
                "home_starter_ip": 6.2,
                "away_starter_ip": 5.1,
                "home_starter_pitches": 94,
                "away_starter_pitches": 91,
                "home_bullpen_er": 1,
                "home_bullpen_ip": 2.2,
                "home_bullpen_h": 2,
                "home_bullpen_bb": 1,
                "home_bullpen_so": 3,
                "home_bullpen_hr": 0,
                "away_bullpen_er": 3,
                "away_bullpen_ip": 3.2,
                "away_bullpen_h": 4,
                "away_bullpen_bb": 2,
                "away_bullpen_so": 2,
                "away_bullpen_hr": 1,
            },
            {
                "league": "NPB",
                "game_id": "g2",
                "datetime": pd.Timestamp("2026-09-02T09:00:00Z"),
                "home": "A",
                "away": "B",
                "home_score": 3,
                "away_score": 4,
                "home_starter": "",
                "away_starter": "",
                "home_bat_pa": 39,
                "home_bat_ab": 35,
                "home_bat_h": 12,
                "home_bat_hr": 1,
                "home_bat_bb": 4,
                "home_bat_so": 7,
                "home_bat_2b": 1,
                "home_bat_3b": 0,
                "home_bat_sb": 0,
                "home_bat_cs": 0,
                "away_bat_pa": 37,
                "away_bat_ab": 34,
                "away_bat_h": 10,
                "away_bat_hr": 2,
                "away_bat_bb": 2,
                "away_bat_so": 10,
                "away_bat_2b": 3,
                "away_bat_3b": 0,
                "away_bat_sb": 1,
                "away_bat_cs": 0,
                "home_starter_era": 3.20,
                "away_starter_era": 3.90,
                "home_starter_fip": 3.40,
                "away_starter_fip": 4.10,
                "home_starter_k9": 8.2,
                "away_starter_k9": 7.1,
                "home_starter_bb9": 2.6,
                "away_starter_bb9": 3.2,
                "home_starter_hr9": 1.0,
                "away_starter_hr9": 1.1,
                "home_starter_whip": 1.15,
                "away_starter_whip": 1.29,
                "home_starter_ip": 5.2,
                "away_starter_ip": 5.8,
                "home_starter_pitches": 87,
                "away_starter_pitches": 95,
            },
        ]
    )

    X, y, meta = bt.build_features(games)
    assert len(X) == 2
    assert len(y) == 2
    assert X.loc[1, "h_bat_avg_3"] == pytest.approx(18 / 36)
    assert X.loc[1, "h_bat_hr_3"] == pytest.approx(2.0)
    assert X.loc[1, "h_starter_team_era_3"] == pytest.approx(2.80)
    assert X.loc[1, "h_starter_team_fip_3"] == pytest.approx(3.00)
    assert X.loc[1, "h_bp_ip_10"] == pytest.approx(2.2)
    assert X.loc[1, "d_starter_team_era_3"] == pytest.approx(2.80 - 4.60)
    assert meta.loc[0, "game_id"] == "g1"
    assert meta.loc[1, "game_id"] == "g2"
