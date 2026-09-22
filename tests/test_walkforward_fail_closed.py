import numpy as np
import pandas as pd
import pytest

from baseball_backtest import BaseballBacktest


def test_walkforward_refuses_partial_block_results(tmp_path, monkeypatch):
    bt = BaseballBacktest(tmp_path)
    n = 120
    games = pd.DataFrame({
        "league": ["MLB"] * n,
        "game_id": [f"g{i}" for i in range(n)],
        "datetime": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
        "home": ["Home"] * n,
        "away": ["Away"] * n,
        "home_score": np.ones(n),
        "away_score": np.zeros(n),
        "home_starter": [""] * n,
        "away_starter": [""] * n,
    })
    X = pd.DataFrame({"x": np.arange(n, dtype=float)})
    y = np.zeros(n, dtype=int)
    meta = games.copy()

    monkeypatch.setattr(bt, "build_features", lambda frame: (X, y, meta))
    def fail_fit(*args, **kwargs):
        raise RuntimeError("deliberate block failure")
    monkeypatch.setattr(bt, "fit_ensemble", fail_fit)
    bt.time_budget_sec = 10_000

    with pytest.raises(RuntimeError, match="walk-forward incomplete"):
        bt.run_walkforward(games, "MLB")

    assert any(row.get("type") == "walkforward_incomplete" for row in bt.audit)
