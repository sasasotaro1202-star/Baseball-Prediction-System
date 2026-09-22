import numpy as np
import pytest

from research.regime_calibration import fit_regime_relative_temperatures


def test_regime_calibration_shrinks_sparse_regime_to_neutral():
    rng = np.random.default_rng(42)
    n = 400
    labels = np.array(["stable"] * 300 + ["sparse"] * 100)
    y = rng.integers(0, 2, size=n)
    p = np.column_stack([np.where(y == 1, 0.30, 0.70), np.where(y == 1, 0.70, 0.30)])

    result = fit_regime_relative_temperatures(
        p, y, labels, min_rows=120, prior_strength=240.0
    )
    assert result["temperatures"]["sparse"] == pytest.approx(1.0)
    assert result["counts"]["stable"] == 300


def test_regime_calibration_does_not_change_row_count_or_class_shape():
    rng = np.random.default_rng(7)
    n = 360
    y = rng.integers(0, 2, size=n)
    labels = np.array(["a"] * 180 + ["b"] * 180)
    p = np.tile([0.55, 0.45], (n, 1))
    result = fit_regime_relative_temperatures(p, y, labels, min_rows=60)
    assert result["probabilities"].shape == (n, 2)
    assert np.allclose(result["probabilities"].sum(axis=1), 1.0)
