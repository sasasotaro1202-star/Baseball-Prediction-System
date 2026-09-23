import pandas as pd

import numpy as np
import pytest

from research.closed_loop_execute import apply_temperature, poisson_result_probs


def test_hilo_uses_total_run_boundary_not_per_team_thresholds():
    df = pd.DataFrame([{"lambda_home": 2.0, "lambda_away": 2.0}])
    low, high, _ = poisson_result_probs(2.0, 2.0, "NPB")
    # P(H+A <= 6) for independent Poisson(2)+Poisson(2) = Poisson(4).
    expected_low = sum(__import__("math").exp(-4.0) * 4.0**k / __import__("math").factorial(k) for k in range(7))
    assert abs(float(low) - expected_low) < 1e-10
    assert abs(float(low) + float(high) - 1.0) < 1e-12


def test_poisson_result_probs_returns_normalized_vector():
    p = poisson_result_probs(2.0, 2.0, "NPB")
    assert isinstance(p, np.ndarray)
    assert p.shape == (3,)
    assert np.isfinite(p).all()
    assert abs(float(p.sum()) - 1.0) < 1e-12


def test_apply_temperature_rejects_nonpositive_temperature():
    p = np.asarray([[0.6, 0.4]], dtype=float)
    with pytest.raises(ValueError, match="strictly positive"):
        apply_temperature(p, 0.0)


def test_apply_temperature_rejects_nonfinite_probabilities():
    p = np.asarray([[np.nan, 1.0]], dtype=float)
    with pytest.raises(ValueError, match="invalid values"):
        apply_temperature(p, 1.0)
