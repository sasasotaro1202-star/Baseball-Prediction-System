import numpy as np

from research.npb_candidate_replay import NPB_DRAW_SCALE_GRID, _draw_scale


def test_draw_scale_preserves_probability_contract():
    p = np.array([[0.50, 0.10, 0.40], [0.20, 0.20, 0.60]], dtype=float)
    q = _draw_scale(p, 2.0)
    assert np.all(np.isfinite(q))
    assert np.all(q >= 0.0)
    assert np.allclose(q.sum(axis=1), 1.0)
    assert np.all(q[:, 1] > p[:, 1] * 0.99)


def test_draw_scale_rejects_invalid_multiplier():
    import pytest
    p = np.array([[0.5, 0.2, 0.3]], dtype=float)
    with pytest.raises(ValueError):
        _draw_scale(p, 0.0)
    with pytest.raises(ValueError):
        _draw_scale(p, float("nan"))


def test_draw_scale_grid_is_bounded_and_monotone():
    assert len(NPB_DRAW_SCALE_GRID) >= 40
    assert np.isfinite(NPB_DRAW_SCALE_GRID).all()
    assert float(NPB_DRAW_SCALE_GRID[0]) >= 0.5
    assert float(NPB_DRAW_SCALE_GRID[-1]) <= 6.0
    assert np.all(np.diff(NPB_DRAW_SCALE_GRID) > 0)



def test_development_selection_skips_calibration_regression():
    from research.npb_candidate_replay import _select_development_candidate

    baseline = {"Accuracy": 0.52, "LogLoss": 0.80, "Brier": 0.52, "ECE": 0.02}
    development = {
        "ProductionEnsemble": baseline,
        "LowECE": {"Accuracy": 0.54, "LogLoss": 0.75, "Brier": 0.50, "ECE": 0.024},
        "BetterLogLossButBadECE": {"Accuracy": 0.56, "LogLoss": 0.70, "Brier": 0.48, "ECE": 0.040},
    }
    selected = _select_development_candidate(development, baseline, 0.005)
    assert selected is not None
    assert selected[0] == "LowECE"


def test_development_selection_returns_none_when_all_fail_calibration():
    from research.npb_candidate_replay import _select_development_candidate

    baseline = {"Accuracy": 0.52, "LogLoss": 0.80, "Brier": 0.52, "ECE": 0.02}
    development = {
        "ProductionEnsemble": baseline,
        "BadA": {"Accuracy": 0.56, "LogLoss": 0.70, "Brier": 0.48, "ECE": 0.040},
        "BadB": {"Accuracy": 0.55, "LogLoss": 0.73, "Brier": 0.49, "ECE": 0.050},
    }
    assert _select_development_candidate(development, baseline, 0.005) is None
