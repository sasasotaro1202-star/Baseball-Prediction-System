import pandas as pd

from data.npb_pbp_adapter import normalize_pbp_frame


def test_explicit_pregame_japanese_starters_are_extracted_without_pitcher_column():
    raw = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g1"],
            "PlayInfo_SeqNo": [0, 1, 2],
            "game_date": ["2025-04-01T09:00:00Z"] * 3,
            "home_team_name": ["西武"] * 3,
            "away_team_name": ["オリックス"] * 3,
            "home_total_runs": [None, None, 3],
            "away_total_runs": [None, None, 2],
            "game_type_name": ["公式戦"] * 3,
            "game_state_name": ["試合前情報", "試合前情報", "試合終了"],
            "description_jap": [
                "先発ピッチャーは西武が隅田、オリックスがエスピノーザ",
                "試合開始",
                "試合終了",
            ],
            "pitcher": [None, None, "別の投手"],
        }
    )

    out = normalize_pbp_frame(raw)

    assert len(out) == 3
    assert set(out["home_pitcher"]) == {"隅田"}
    assert set(out["away_pitcher"]) == {"エスピノーザ"}


def test_ambiguous_starter_text_does_not_guess():
    raw = pd.DataFrame(
        {
            "game_id": ["g2"],
            "PlayInfo_SeqNo": [0],
            "game_date": ["2025-04-01T09:00:00Z"],
            "home_team_name": ["A"],
            "away_team_name": ["B"],
            "home_total_runs": [1],
            "away_total_runs": [0],
            "game_type_name": ["公式戦"],
            "description_jap": ["先発ピッチャーはAが投手"] ,
            "pitcher": [None],
        }
    )

    out = normalize_pbp_frame(raw)

    assert out.iloc[0]["home_pitcher"] == ""
    assert out.iloc[0]["away_pitcher"] == ""


def test_compact_pitch_metadata_is_preserved_for_safe_postgame_usage():
    raw = pd.DataFrame(
        {
            "game_id": ["g3", "g3", "g3", "g3"],
            "PlayInfo_SeqNo": [1, 2, 3, 4],
            "game_date": ["2025-04-01T09:00:00Z"] * 4,
            "home_team_name": ["西武"] * 4,
            "away_team_name": ["オリックス"] * 4,
            "home_total_runs": [1, 1, 1, 1],
            "away_total_runs": [0, 0, 0, 0],
            "game_type_name": ["公式戦"] * 4,
            "description_jap": [""] * 4,
            "pitcher": [101, 101, 202, 303],
            "TB": ["T", "T", "B", "B"],
        }
    )
    out = normalize_pbp_frame(raw)
    assert {"pitcher_id", "half_inning"}.issubset(out.columns)
    assert out["pitcher_id"].tolist() == ["101", "101", "202", "303"]
    assert out["half_inning"].tolist() == ["T", "T", "B", "B"]
