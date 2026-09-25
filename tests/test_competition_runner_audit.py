import pandas as pd

from research_runner_v6 import _competition_audit


def test_runner_audits_competition_partitions_without_outcome_information():
    games = pd.DataFrame([
        {"league": "MLB", "game_type": "R", "series_description": "Regular Season"},
        {"league": "MLB", "game_type": "F", "series_description": "Wild Card Series"},
        {"league": "MLB", "game_type": "", "series_description": "World Series Game 1"},
    ])
    audited = _competition_audit(games, "MLB")
    out = audited["games"]
    inventory = audited["inventory"]

    assert out.loc[0, "competition_key"] == "MLB:mlb_regular:regular_season"
    assert out.loc[1, "competition_stage"] == "wild_card"
    assert out.loc[2, "competition_classification_status"] == "unknown"
    assert inventory["rows"] == 3
    assert inventory["unknown_rows"] == 1
    assert inventory["competition_counts"]["MLB:mlb_regular:regular_season"] == 1


def test_runner_audits_npb_interleague_separately_from_regular():
    games = pd.DataFrame([
        {"league": "NPB", "game_type": "公式戦"},
        {"league": "NPB", "game_type": "交流戦"},
    ])
    audited = _competition_audit(games, "NPB")
    keys = set(audited["games"]["competition_key"].tolist())
    assert "NPB:npb_regular:regular_season" in keys
    assert "NPB:npb_interleague:interleague" in keys
