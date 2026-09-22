import numpy as np
import pandas as pd
import pytest

from baseball_backtest import BaseballBacktest


def test_build_features_rejects_unexpected_nan_in_feature_matrix(tmp_path, monkeypatch):
    games = pd.DataFrame({
        "league": ["MLB"] * 2,
        "game_id": ["g1", "g2"],
        "datetime": pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC"),
        "home": ["Home", "Home"],
        "away": ["Away", "Away"],
        "home_score": [1, 2],
        "away_score": [0, 1],
        "home_starter": ["", ""],
        "away_starter": ["", ""],
    })
    bt = BaseballBacktest(tmp_path)

    def poisoned_match_features(row):
        return {"safe_feature": 1.0, "unexpected_missing": np.nan}

    monkeypatch.setattr(bt, "match_features", poisoned_match_features)

    with pytest.raises(RuntimeError, match="non-finite feature value"):
        bt.build_features(games)
