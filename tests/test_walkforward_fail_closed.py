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


def test_walkforward_invalidates_checkpoint_on_input_fingerprint_mismatch(tmp_path, monkeypatch):
    bt = BaseballBacktest(tmp_path)
    n = 120
    games = pd.DataFrame({
        "league": ["MLB"] * n,
        "game_id": [f"g{i}" for i in range(n)],
        "datetime": pd.date_range("2026-02-01", periods=n, freq="h", tz="UTC"),
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
    bt.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = bt.checkpoint_dir / "mlb_walkforward.csv"
    version = checkpoint.with_suffix(".version")
    pd.DataFrame({
        "game_id": [f"g{i}" for i in range(100, 120)],
        "model": ["stale-model"] * 20,
        "input_fingerprint": ["stale-fingerprint"] * 20,
    }).to_csv(checkpoint, index=False)
    version.write_text(bt.checkpoint_version, encoding="utf-8")

    def fail_fit(*args, **kwargs):
        raise RuntimeError("deliberate stale-checkpoint rejection")
    monkeypatch.setattr(bt, "fit_ensemble", fail_fit)
    bt.time_budget_sec = 10_000

    with pytest.raises(RuntimeError, match="walk-forward incomplete"):
        bt.run_walkforward(games, "MLB")

    assert any(row.get("type") == "walkforward_incomplete" for row in bt.audit)


def test_build_features_freezes_state_within_same_timestamp(tmp_path, monkeypatch):
    bt = BaseballBacktest(tmp_path)
    t0 = pd.Timestamp("2026-03-01T12:00:00Z")
    games = pd.DataFrame({
        "league": ["MLB", "MLB", "MLB"],
        "game_id": ["g0", "g1", "g2"],
        "datetime": [t0, t0, t0 + pd.Timedelta(hours=1)],
        "home": ["H0", "H1", "H2"],
        "away": ["A0", "A1", "A2"],
        "home_score": [1.0, 2.0, 3.0],
        "away_score": [0.0, 1.0, 0.0],
    })
    seen = []

    monkeypatch.setattr(bt, "load_npb_player_features", lambda: pd.DataFrame())

    def fake_match_features(row):
        seen.append((str(row["game_id"]), len(bt.states)))
        return {"state_count": float(len(bt.states))}

    def fake_update_after_game(row):
        bt.states[(str(row["league"]), str(row["game_id"]))] = object()

    monkeypatch.setattr(bt, "match_features", fake_match_features)
    monkeypatch.setattr(bt, "update_after_game", fake_update_after_game)

    X, y, meta = bt.build_features(games)

    assert list(X["state_count"]) == [0.0, 0.0, 2.0]
    assert seen == [("g0", 0), ("g1", 0), ("g2", 2)]
    assert any(row.get("type") == "same_timestamp_state_freeze" for row in bt.audit)
