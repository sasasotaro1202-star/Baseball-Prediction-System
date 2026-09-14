import math

import pytest

from prediction.score_distribution import (
    build_score_outputs,
    low_high_probabilities,
    score_distribution,
    top_score_candidates,
)


def test_top_four_are_exact_highest_probability_scorelines():
    got = top_score_candidates(4.2, 3.1)
    assert len(got) == 4
    probs = [x["probability"] for x in got]
    assert probs == sorted(probs, reverse=True)
    assert all(x["score"] != "その他" for x in got)

    matrix = score_distribution(4.2, 3.1)
    expected = sorted(
        ((float(matrix[h, a]), f"{h}-{a}") for h in range(matrix.shape[0]) for a in range(matrix.shape[1])),
        reverse=True,
    )[:4]
    assert [(x["score"], x["probability"]) for x in got] == [
        (score, probability) for probability, score in expected
    ]


def test_low_high_is_total_runs_boundary_not_teamwise_boundary():
    low, high = low_high_probabilities(6.0, 6.0)
    assert 0.0 < low < 1.0
    assert math.isclose(low + high, 1.0, abs_tol=1e-12)

    # 6-6 is HIGH (12 total), so the implementation must not use
    # P(home<=6) * P(away<=6) as its Low definition.
    matrix = score_distribution(6.0, 6.0)
    wrong_teamwise = sum(float(matrix[h, a]) for h in range(7) for a in range(7))
    assert abs(low - wrong_teamwise) > 0.05


def test_game_specific_lambdas_change_the_top_four():
    first = [x["score"] for x in top_score_candidates(2.0, 1.2)]
    second = [x["score"] for x in top_score_candidates(7.0, 5.0)]
    assert first != second


def test_low_high_uses_full_distribution_independent_of_top_four():
    output = build_score_outputs(5.0, 4.5)
    assert len(output["score_candidates"]) == 4
    assert output["low_definition"] == "total runs <= 6"
    assert output["high_definition"] == "total runs >= 7"
    assert output["low_high_boundary"] == 6.5
    assert math.isclose(output["low_probability"] + output["high_probability"], 1.0, abs_tol=1e-12)


def test_invalid_run_means_fail_closed():
    with pytest.raises(ValueError):
        score_distribution(0, 3)
    with pytest.raises(ValueError):
        score_distribution(float("nan"), 3)
    with pytest.raises(ValueError):
        top_score_candidates(3, 4, n=3)
