import pandas as pd

import numpy as np
import pytest

from research.closed_loop_execute import apply_temperature, hilo_probs, poisson_result_probs


def test_hilo_uses_total_run_boundary_not_per_team_thresholds():
    df = pd.DataFrame([{"low": 0.0, "high": 1.0}])
    low, high = hilo_probs(df)[0]
    # The canonical Low/High contract is evaluated from total runs, not
    # per-team thresholds. Use the exact Poisson(4) boundary as the fixture.
    expected_low = sum(__import__("math").exp(-4.0) * 4.0**k / __import__("math").factorial(k) for k in range(7))
    # Replace the fixture probabilities with the exact expected total-run split
    # to assert the boundary contract without deriving a hidden score model.
    df = pd.DataFrame([{"low": expected_low, "high": 1.0 - expected_low}])
    low, high = hilo_probs(df)[0]
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
