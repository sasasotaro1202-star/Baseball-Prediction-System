import numpy as np

from research.npb_candidate_replay import _draw_scale


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
