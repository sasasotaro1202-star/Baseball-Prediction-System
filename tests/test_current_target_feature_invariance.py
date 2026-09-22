import numpy as np
import pandas as pd

from baseball_backtest import BaseballBacktest


def test_current_game_target_cannot_change_its_pregame_features(tmp_path):
    n = 121
    games = pd.DataFrame({
        "league": ["MLB"] * n,
        "game_id": [f"g{i}" for i in range(n)],
        "datetime": pd.date_range("2026-04-01", periods=n, freq="h", tz="UTC"),
        "home": ["Home"] * n,
        "away": ["Away"] * n,
        "home_score": [1 + (i % 4) for i in range(n)],
        "away_score": [i % 3 for i in range(n)],
        "home_starter": [""] * n,
        "away_starter": [""] * n,
    })
    mutated = games.copy()
    target_idx = n - 1
    mutated.loc[target_idx, "home_score"] = 11
    mutated.loc[target_idx, "away_score"] = 0

    bt_a = BaseballBacktest(tmp_path)
    x_a, y_a, meta_a = bt_a.build_features(games)

    bt_b = BaseballBacktest(tmp_path)
    x_b, y_b, meta_b = bt_b.build_features(mutated)

    feature_a = x_a.iloc[target_idx].to_numpy(dtype=float)
    feature_b = x_b.iloc[target_idx].to_numpy(dtype=float)

    assert np.allclose(feature_a, feature_b, equal_nan=True), (
        "current-game realized score changed the pregame feature vector"
    )
    assert int(y_a[target_idx]) != int(y_b[target_idx])
