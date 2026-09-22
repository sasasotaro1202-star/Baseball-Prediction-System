import numpy as np
import pytest

from research.calibrated_stacking import (
    apply_calibrated_stacking,
    fit_calibrated_stacking,
)


def test_calibrated_stacking_is_deterministic_and_normalized():
    rng = np.random.default_rng(222)
    x = rng.normal(size=(180, 5))
    y = (x[:, 0] - 0.3 * x[:, 1] > 0).astype(int)
    p1 = np.column_stack([1.0 / (1.0 + np.exp(x[:, 0])), 1.0 / (1.0 + np.exp(-x[:, 0]))])
    p2 = np.column_stack([1.0 / (1.0 + np.exp(0.7 * x[:, 0] + 0.2 * x[:, 1])), 1.0 / (1.0 + np.exp(-0.7 * x[:, 0] - 0.2 * x[:, 1]))])
    p1 /= p1.sum(axis=1, keepdims=True)
    p2 /= p2.sum(axis=1, keepdims=True)
    a = fit_calibrated_stacking(predictions={"a": p1, "b": p2}, y=y, simplex_step=0.25)
    b = fit_calibrated_stacking(predictions={"a": p1, "b": p2}, y=y, simplex_step=0.25)
    assert a == b
    q = apply_calibrated_stacking(a, {"a": p1[:20], "b": p2[:20]})
    assert q.shape == (20, 2)
    assert np.allclose(q.sum(axis=1), 1.0)


def test_calibrated_stacking_requires_two_or_three_components():
    rng = np.random.default_rng(223)
    p = np.column_stack([np.full(80, 0.6), np.full(80, 0.4)])
    y = np.array([0, 1] * 40)
    with pytest.raises(ValueError):
        fit_calibrated_stacking(predictions={"a": p}, y=y)
