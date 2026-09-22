import numpy as np
from research.correlated_score import (
    estimate_shared_lambda,
    grid,
    low_high,
    npb_final_outcomes,
    top_scores,
)


def test_grid_normalizes_and_preserves_nonnegative_mass():
    m = grid(4.2, 3.8, 0.35, 14)
    assert abs(m.sum() - 1.0) < 1e-9
    assert np.isfinite(m).all()
    assert (m >= 0).all()


def test_shared_lambda_is_nonnegative_and_bounded():
    assert estimate_shared_lambda(np.ones(100), np.ones(100) * 2, max_shared=0.4) == 0.4
    assert estimate_shared_lambda(np.ones(100), -np.ones(100), max_shared=0.4) == 0.0


def test_low_high_contract():
    low, high = low_high(4.0, 3.0, 0.2)
    assert 0 <= low <= 1 and 0 <= high <= 1
    assert abs(low + high - 1) < 1e-9


def test_top4_exact_scores_are_unique_and_bounded():
    scores = top_scores(3.2, 2.7, 0.15, 4)
    assert len(scores) == 4
    assert len({score for score, _ in scores}) == 4
    assert all(np.isfinite(prob) and 0 <= prob <= 1 for _, prob in scores)


def test_npb_final_outcomes_normalize():
    home, draw, away = npb_final_outcomes(3.2, 2.7, 0.15)
    assert all(np.isfinite(x) and 0 <= x <= 1 for x in (home, draw, away))
    assert abs(home + draw + away - 1.0) < 1e-12


def test_symmetric_npb_model_is_symmetric():
    home, draw, away = npb_final_outcomes(3.0, 3.0, 0.0)
    assert abs(home - away) < 1e-12
    assert abs(home + draw + away - 1.0) < 1e-12


def test_npb_home_away_orientation_matches_score_grid():
    home, draw, away = npb_final_outcomes(5.0, 2.0, 0.0, extra_innings=0)
    m = grid(5.0, 2.0, 0.0)
    expected_home = float(np.tril(m, -1).sum())
    expected_away = float(np.triu(m, 1).sum())
    expected_draw = float(np.trace(m))
    assert abs(home - expected_home) < 1e-12
    assert abs(draw - expected_draw) < 1e-12
    assert abs(away - expected_away) < 1e-12


def test_shared_component_cannot_change_requested_marginal_means():
    # Oversized shared intensity must be clamped rather than producing a
    # different marginal model through the old per-component floor.
    m = grid(2.0, 5.0, 9.0, 14)
    home_mean = sum(i * m[i, j] for i in range(m.shape[0]) for j in range(m.shape[1]))
    away_mean = sum(j * m[i, j] for i in range(m.shape[0]) for j in range(m.shape[1]))
    assert abs(home_mean - 2.0) < 0.08
    assert abs(away_mean - 5.0) < 0.08

def test_canonical_output_contract_uses_shared_component(monkeypatch, tmp_path):
    import pandas as pd
    from research.backtest_output_contract import _repair_scores

    frame = pd.DataFrame({
        "lambda_home": [4.5],
        "lambda_away": [3.2],
        "shared_lambda": [0.75],
    })
    out = _repair_scores(frame)
    expected = grid(4.5, 3.2, 0.75, 20)
    assert np.isclose(out.loc[0, "low"], expected[np.indices(expected.shape).sum(axis=0) <= 6].sum())
    assert out.loc[0, "score1"] == top_scores(4.5, 3.2, 0.75, 4)[0][0]
