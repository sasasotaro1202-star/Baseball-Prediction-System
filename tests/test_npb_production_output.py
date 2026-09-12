import pytest

from prediction.npb_output import build_npb_prediction, validate_npb_prediction


def test_production_output_is_three_way():
    row = build_npb_prediction({
        "home": "読売ジャイアンツ",
        "away": "阪神タイガース",
        "pred_home": 0.48,
        "pred_draw": 0.12,
        "pred_away": 0.40,
    })
    assert row["prediction_contract"] == "NPB_HOME_DRAW_AWAY_V1"
    assert row["predicted_outcome"] == "HOME_WIN"
    assert row["draw_probability"] == pytest.approx(0.12)
    assert sum(row[x] for x in ("home_win_probability", "draw_probability", "away_win_probability")) == pytest.approx(1.0)


def test_production_output_fails_closed_without_draw():
    with pytest.raises(ValueError):
        build_npb_prediction({
            "home": "読売ジャイアンツ",
            "away": "阪神タイガース",
            "pred_home": 0.60,
            "pred_away": 0.40,
        })
