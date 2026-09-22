import pandas as pd

from baseball_backtest import BaseballBacktest


def _frame(**extra):
    base = {
        "game_id": ["1"],
        "datetime": ["2025-06-01T18:00:00Z"],
        "home": ["A"],
        "away": ["B"],
        "home_score": [3],
        "away_score": [2],
        "home_starter": ["Starter A"],
        "away_starter": ["Starter B"],
    }
    base.update(extra)
    return pd.DataFrame(base)


def test_mlb_probable_starters_without_timestamp_are_not_pit_safe(tmp_path):
    bt = BaseballBacktest(tmp_path)
    out = bt._normalize_mlb_games(_frame())
    assert out.loc[0, "home_starter"] == ""
    assert out.loc[0, "away_starter"] == ""
    assert bool(out.loc[0, "confirmed_starters"]) is False
    assert out.loc[0, "starter_evidence_status"] == "unknown"


def test_mlb_starters_are_retained_only_when_announced_before_cutoff(tmp_path):
    bt = BaseballBacktest(tmp_path)
    out = bt._normalize_mlb_games(_frame(
        prediction_cutoff=["2025-06-01T12:00:00Z"],
        home_starter_announced_at=["2025-06-01T08:00:00Z"],
        away_starter_announced_at=["2025-06-01T09:30:00Z"],
    ))
    assert out.loc[0, "home_starter"] == "Starter A"
    assert out.loc[0, "away_starter"] == "Starter B"
    assert bool(out.loc[0, "confirmed_starters"]) is True
    assert out.loc[0, "starter_evidence_status"] == "pit_safe"


def test_mlb_starter_after_cutoff_is_rejected(tmp_path):
    bt = BaseballBacktest(tmp_path)
    out = bt._normalize_mlb_games(_frame(
        prediction_cutoff=["2025-06-01T12:00:00Z"],
        home_starter_announced_at=["2025-06-01T13:00:00Z"],
        away_starter_announced_at=["2025-06-01T09:30:00Z"],
    ))
    assert out.loc[0, "home_starter"] == ""
    assert out.loc[0, "away_starter"] == "Starter B"
    assert bool(out.loc[0, "confirmed_starters"]) is False
    assert out.loc[0, "starter_evidence_status"] == "partial"
