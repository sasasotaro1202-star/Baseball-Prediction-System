import numpy as np
import pytest

from baseball_backtest import clip_prob


def test_clip_prob_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        clip_prob([0.5, np.nan])
    with pytest.raises(ValueError):
        clip_prob([0.5, np.inf])
    with pytest.raises(ValueError):
        clip_prob([0.5, -0.1])


def test_clip_prob_normalizes_valid_values():
    p = clip_prob([0.2, 0.3, 0.5])
    assert np.allclose(p.sum(), 1.0)
    assert np.all(p > 0)
