import pandas as pd

from baseball_backtest import BaseballBacktest


def test_pregame_interactions_are_deterministic_and_target_free(monkeypatch):
    bt = BaseballBacktest.__new__(BaseballBacktest)
    monkeypatch.setattr(
        bt,
        "_team_features",
        lambda league, team, venue, dt: {
            "gf_10": 4.0 if venue == "home" else 3.0,
            "ga_10": 2.0 if venue == "home" else 3.0,
            "gd_10": 2.0 if venue == "home" else 0.0,
            "rest_days": 2.0 if venue == "home" else 1.0,
            "elo": 1510.0 if venue == "home" else 1490.0,
            "bp_app_10": 1.0 if venue == "home" else 3.0,
            "bp_era_10": 3.5 if venue == "home" else 4.5,
            "bp_whip_10": 1.2 if venue == "home" else 1.4,
            "bp_k9_10": 8.5 if venue == "home" else 7.5,
            "bp_bb9_10": 2.5 if venue == "home" else 3.0,
            "bp_hr9_10": 0.8 if venue == "home" else 1.1,
            "bp_actual_coverage_10": 1.0,
            "gf_sd_20": 1.1 if venue == "home" else 1.4,
            "gf_slope_20": 0.05 if venue == "home" else -0.02,
            "bat_hr_rate_10": 0.04 if venue == "home" else 0.03,
            "bat_bb_rate_10": 0.09 if venue == "home" else 0.08,
            "bat_avg_10": 0.270 if venue == "home" else 0.250,
            "bat_so_rate_10": 0.19 if venue == "home" else 0.21,
        },
    )
    monkeypatch.setattr(
        bt,
        "starter_features",
        lambda league, pitcher, dt, prefix: {
            prefix + "k9": 9.0 if prefix == "hs_" else 7.0,
            prefix + "bb9": 2.0 if prefix == "hs_" else 3.0,
            prefix + "hr9": 0.8 if prefix == "hs_" else 1.1,
            prefix + "recent_era": 3.2 if prefix == "hs_" else 4.1,
            prefix + "starts": 20.0 if prefix == "hs_" else 5.0,
            prefix + "fip": 3.2 if prefix == "hs_" else 4.1,
            prefix + "recent_k9": 9.0 if prefix == "hs_" else 7.0,
            prefix + "recent_bb_rate": 0.07,
            prefix + "recent_k_rate": 0.24,
            prefix + "recent_pitches": 90.0,
            prefix + "recent_ip": 6.0,
            prefix + "era": 3.5 if prefix == "hs_" else 4.2,
            prefix + "whip": 1.15 if prefix == "hs_" else 1.35,
            prefix + "recent_era_sd": 0.2,
            prefix + "recent_k9_sd": 0.4,
            prefix + "recent_pitches_sd": 4.0,
            prefix + "era_slope": 0.0,
            prefix + "k9_slope": 0.0,
            prefix + "fip_slope": 0.0,
        },
    )
    monkeypatch.setattr(bt, "_context_pit_safe", lambda row: False)

    row = pd.Series({
        "league": "MLB",
        "home": "HOME",
        "away": "AWAY",
        "datetime": "2026-08-01T18:00:00Z",
        "home_starter": "H",
        "away_starter": "A",
    })
    out = bt.match_features(row)

    assert out["elo_x_starter_quality_gap"] == 20.0 * ((9.0 - 2.0 - 0.8) - (7.0 - 3.0 - 1.1))
    assert out["starter_quality_reliability_gap"] != 0.0
    assert out["bullpen_fatigue_x_rest_gap"] == -2.0 * (1.0 + 1.0)
    assert out["environment_x_volatility_gap"] != 0.0
    assert all(v == v for v in out.values())
