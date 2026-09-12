import numpy as np
import pytest

from evaluation.npb_outcome import (
    NPB_OUTCOME_LABELS,
    add_npb_outcome_columns,
    npb_result_label,
    score_matrix_to_outcomes,
    validate_npb_probabilities,
)


def test_npb_result_has_three_outcomes():
    assert npb_result_label(5, 3) == "HOME_WIN"
    assert npb_result_label(3, 3) == "DRAW"
    assert npb_result_label(2, 4) == "AWAY_WIN"
    assert NPB_OUTCOME_LABELS == ("HOME_WIN", "DRAW", "AWAY_WIN")


def test_npb_probabilities_keep_draw_explicit():
    row = add_npb_outcome_columns({
        "pred_home": 0.47,
        "pred_draw": 0.14,
        "pred_away": 0.39,
    })
    assert row["home_win_probability"] == pytest.approx(0.47)
    assert row["draw_probability"] == pytest.approx(0.14)
    assert row["away_win_probability"] == pytest.approx(0.39)
    assert row["outcome_labels"] == ["HOME_WIN", "DRAW", "AWAY_WIN"]


def test_score_matrix_produces_draw_probability():
    matrix = np.array([
        [0.10, 0.05, 0.02],
        [0.04, 0.20, 0.06],
        [0.01, 0.03, 0.49],
    ])
    p = score_matrix_to_outcomes(matrix)
    assert p.shape == (3,)
    assert p.sum() == pytest.approx(1.0)
    assert p[1] == pytest.approx(0.20 / matrix.sum())


def test_npb_probability_shape_is_strict():
    with pytest.raises(ValueError):
        validate_npb_probabilities([0.6, 0.4])
