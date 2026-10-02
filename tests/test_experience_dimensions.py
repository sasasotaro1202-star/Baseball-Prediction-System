import pandas as pd

from research.experience_dimensions import add_dimensions


def test_dimensions_keep_league_when_competition_metadata_is_unknown():
    frame = pd.DataFrame([{
        "league": "NPB",
        "competition_id": "NPB",
        "game_type": "",
        "series_description": "",
    }])
    out = add_dimensions(frame)
    assert out.loc[0, "league"] == "NPB"
    assert out.loc[0, "competition"].endswith("UNKNOWN")
    assert out.loc[0, "competition_stage"] == "UNKNOWN"
    assert out.loc[0, "competition_classification_status"] == "UNKNOWN"


def test_dimensions_classify_npb_interleague_without_outcome_data():
    frame = pd.DataFrame([{
        "league": "NPB",
        "competition_id": "NPB",
        "game_type": "交流戦",
    }])
    out = add_dimensions(frame)
    assert out.loc[0, "league"] == "NPB"
    assert out.loc[0, "competition"] == "NPB_INTERLEAGUE"
    assert out.loc[0, "competition_stage"] == "INTERLEAGUE"
    assert out.loc[0, "season_type"] == "REGULAR_SEASON"
    assert out.loc[0, "competition_classification_status"] == "CLASSIFIED"


def test_dimensions_never_silently_map_unknown_phase():
    frame = pd.DataFrame([{
        "league": "MLB",
        "competition_id": "MLB",
        "game_type": "",
        "series_description": "Unknown Series",
    }])
    out = add_dimensions(frame)
    assert out.loc[0, "league"] == "MLB"
    assert out.loc[0, "competition"] == "MLB_UNKNOWN"
    assert out.loc[0, "competition_stage"] == "UNKNOWN"
    assert out.loc[0, "competition_classification_status"] == "UNKNOWN"
