"""Smoke tests for the parallel matrix pipeline."""
import pandas as pd
from parallel.elo_model import run_elo_walk_forward
from parallel.soccer_elo_model import run_soccer_elo_walk_forward
from parallel.metrics import accuracy, base_rate_home_win
from parallel.feature_store import apply_stages


def _toy_mlb_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04"]),
        "home_team": ["A", "B", "A", "B"],
        "away_team": ["B", "A", "B", "A"],
        "home_score": [5, 2, 4, 3],
        "away_score": [3, 4, 1, 2],
        "league": ["MLB"] * 4,
    })


def test_elo_walk_forward_runs():
    df = _toy_mlb_df()
    result = run_elo_walk_forward(df)
    assert len(result.predictions) == 4
    assert set(result.final_ratings.keys()) == {"A", "B"}
    acc = accuracy(result.predictions["correct"])
    assert 0.0 <= acc <= 1.0


def test_base_rate_home_win():
    df = _toy_mlb_df()
    result = run_elo_walk_forward(df)
    rate = base_rate_home_win(result.predictions, "actual_home_win")
    assert 0.0 <= rate <= 1.0


def test_soccer_elo_walk_forward_runs():
    df = _toy_mlb_df()
    result = run_soccer_elo_walk_forward(df)
    assert len(result.predictions) == 4
    assert set(result.predictions["prediction"].unique()) <= {"H", "D", "A"}


def test_feature_store_noop_stage():
    df = _toy_mlb_df()
    out = apply_stages(df, ["elo_only"])
    assert list(out.columns) == list(df.columns)
