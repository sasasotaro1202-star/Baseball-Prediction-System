import pytest

from prediction.game_ranking import rank_games


def game(event_id, home, draw, away):
    return {
        "event_id": event_id,
        "home_probability": home,
        "draw_probability": draw,
        "away_probability": away,
    }


def test_highest_draw_game_is_separate_from_winner_ranking():
    result = rank_games(
        [game("g1", 0.55, 0.10, 0.35), game("g2", 0.41, 0.21, 0.38), game("g3", 0.46, 0.18, 0.36)],
        draw_capable=True,
    )
    assert result["draw_candidate"]["event_id"] == "g2"
    assert result["draw_candidate"]["draw_probability"] == pytest.approx(0.21)
    assert result["draw_candidate"]["draw_probability_pct"] == pytest.approx(21.0)
    assert result["draw_candidate"]["ranking_type"] == "highest_draw_probability"
    assert result["winner_ranking"][0]["event_id"] == "g1"


def test_draw_candidate_is_not_forced_to_be_normal_winner():
    result = rank_games(
        [game("g1", 0.60, 0.05, 0.35), game("g2", 0.31, 0.34, 0.35)],
        draw_capable=True,
    )
    assert result["draw_candidate"]["event_id"] == "g2"
    assert result["draw_candidate"]["winner"] == "away"


def test_every_draw_capable_game_requires_a_real_draw_probability():
    with pytest.raises(ValueError):
        rank_games(
            [{"event_id": "g1", "home_probability": 0.6, "away_probability": 0.4}],
            draw_capable=True,
        )


def test_non_draw_competition_has_no_draw_candidate():
    result = rank_games(
        [
            {"event_id": "g1", "home_probability": 0.6, "away_probability": 0.4},
            {"event_id": "g2", "home_probability": 0.3, "away_probability": 0.7},
        ],
        draw_capable=False,
    )
    assert result["draw_candidate"] is None
    assert result["draw_capable"] is False


def test_low_confidence_games_are_explicitly_separated():
    result = rank_games(
        [game("strong", 0.80, 0.05, 0.15), game("uncertain", 0.36, 0.33, 0.31)],
        draw_capable=True,
    )
    assert result["low_confidence_count"] == 1
    assert result["low_confidence_games"][0]["event_id"] == "uncertain"
    assert result["winner_ranking"][0]["event_id"] == "strong"
