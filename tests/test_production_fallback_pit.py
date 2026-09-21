import pandas as pd
import pytest

from production_npb import direct_pit_safe_lambdas


class FakeBacktest:
    def starter_features(self, league, pitcher, dt, prefix):
        return {prefix + "fip": 4.0}


def test_direct_fallback_excludes_future_games_from_prediction_cutoff():
    hist = pd.DataFrame(
        [
            {"datetime": "2026-09-20T00:00:00Z", "game_id": "old", "home": "A", "away": "B", "home_score": 3, "away_score": 2},
            {"datetime": "2026-09-21T00:00:00Z", "game_id": "future", "home": "A", "away": "B", "home_score": 30, "away_score": 30},
        ]
    )
    row = pd.Series({
        "datetime": pd.Timestamp("2026-09-21T12:00:00Z"),
        "home": "A",
        "away": "B",
        "home_starter": "H",
        "away_starter": "A",
    })
    lh, la, _ = direct_pit_safe_lambdas(hist, row, FakeBacktest())
    assert lh < 10.0
    assert la < 10.0


def test_direct_fallback_rejects_malformed_historical_datetime():
    hist = pd.DataFrame(
        [{"datetime": "not-a-date", "game_id": "bad", "home": "A", "away": "B", "home_score": 3, "away_score": 2}]
    )
    row = pd.Series({
        "datetime": pd.Timestamp("2026-09-21T12:00:00Z"),
        "home": "A",
        "away": "B",
        "home_starter": "H",
        "away_starter": "A",
    })
    with pytest.raises(RuntimeError, match="historical datetime is malformed"):
        direct_pit_safe_lambdas(hist, row, FakeBacktest())
