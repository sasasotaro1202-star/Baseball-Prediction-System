import pytest

from prediction.npb_draw_rank import select_top_draw_prediction


def test_selects_highest_draw_probability_separately():
    result = select_top_draw_prediction([
        {"home": "A", "away": "B", "draw_probability": 0.11},
        {"home": "C", "away": "D", "draw_probability": 0.24},
        {"home": "E", "away": "F", "draw_probability": 0.18},
    ])
    assert result["home"] == "C"
    assert result["draw_probability"] == pytest.approx(0.24)
    assert result["prediction_type"] == "NPB_TOP_DRAW_PROBABILITY_V1"
    assert result["draw_prediction"] is True


def test_rejects_missing_draw_probability():
    with pytest.raises(ValueError):
        select_top_draw_prediction([{ "home": "A", "away": "B" }])
