"""Regression tests for MLB candidate replay runtime contract."""

from research.mlb_candidate_replay import MLBReplayConfig


def test_mlb_candidate_retraining_default_is_budget_bounded():
    """MLB research replay defaults to the bounded 720-game cadence."""
    assert MLBReplayConfig().retrain_every == 720
