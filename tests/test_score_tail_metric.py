import pandas as pd

from baseball_backtest import BaseballBacktest


def test_evaluate_does_not_count_tail_bucket_as_exact_top4_hit():
    bt = BaseballBacktest()
    df = pd.DataFrame([
        {
            "correct": 1, "logloss": 0.2, "brier": 0.1,
            "actual_home_score": 7, "actual_away_score": 0,
            "lambda_home": 6.0, "lambda_away": 2.0,
            "high": 0.9,
            "score1": "3-2", "score2": "2-1", "score3": "1-0", "score4": "その他",
            "actual": 1, "pred_home": 0.7,
        },
        {
            "correct": 1, "logloss": 0.2, "brier": 0.1,
            "actual_home_score": 3, "actual_away_score": 2,
            "lambda_home": 3.0, "lambda_away": 2.0,
            "high": 0.3,
            "score1": "3-2", "score2": "2-1", "score3": "1-0", "score4": "その他",
            "actual": 1, "pred_home": 0.7,
        },
    ])
    result = bt.evaluate(df, "MLB")
    assert result["Top4ScoreHitRate"] == 0.5
