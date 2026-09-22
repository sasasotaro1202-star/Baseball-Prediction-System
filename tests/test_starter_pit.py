import pandas as pd

from baseball_backtest import BaseballBacktest


def test_npb_pbp_first_pitcher_is_not_prediction_time_starter(tmp_path):
    bt = BaseballBacktest(tmp_path)
    pbp = pd.DataFrame([
        {
            "game_id": "npb-1",
            "row_order": 1,
            "date": "2025-06-01T18:00:00+09:00",
            "game_type": "公式戦",
            "home": "巨人",
            "away": "阪神",
            "home_score": 0,
            "away_score": 0,
            "home_pitcher": "実際の先発A",
            "away_pitcher": "実際の先発B",
        },
        {
            "game_id": "npb-1",
            "row_order": 2,
            "date": "2025-06-01T18:01:00+09:00",
            "game_type": "公式戦",
            "home": "巨人",
            "away": "阪神",
            "home_score": 3,
            "away_score": 2,
            "home_pitcher": "実際の先発A",
            "away_pitcher": "実際の先発B",
        },
    ])
    out = bt.aggregate_npb_games(pbp)
    assert out.loc[0, "home_starter"] == ""
    assert out.loc[0, "away_starter"] == ""
    assert bool(out.loc[0, "confirmed_starters"]) is False
    assert out.loc[0, "starter_evidence_status"] == "unknown"
