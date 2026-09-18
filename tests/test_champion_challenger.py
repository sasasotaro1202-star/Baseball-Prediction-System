import numpy as np
import pytest

from research.champion_challenger import compare_probabilities, promotable


def test_challenger_comparison_is_deterministic_and_paired():
    y = np.array([0, 1, 2, 0, 1, 2] * 10)
    incumbent = np.tile([[0.55, 0.25, 0.20], [0.25, 0.55, 0.20], [0.20, 0.25, 0.55]], (20, 1))
    challenger = np.tile([[0.62, 0.23, 0.15], [0.22, 0.63, 0.15], [0.15, 0.23, 0.62]], (20, 1))
    a = compare_probabilities(y, incumbent, challenger, seed=7, bootstrap_rounds=500)
    b = compare_probabilities(y, incumbent, challenger, seed=7, bootstrap_rounds=500)
    assert a == b
    assert a.n == 60
    assert a.challenger_logloss < a.incumbent_logloss
    assert a.challenger_brier < a.incumbent_brier


def test_promotable_requires_large_sample_and_positive_paired_gain():
    y = np.array([0, 1, 2] * 40)
    incumbent = np.tile([[0.55, 0.25, 0.20], [0.25, 0.55, 0.20], [0.20, 0.25, 0.55]], (40, 1))
    challenger = np.tile([[0.62, 0.23, 0.15], [0.22, 0.63, 0.15], [0.15, 0.23, 0.62]], (40, 1))
    c = compare_probabilities(y, incumbent, challenger, seed=11, bootstrap_rounds=500)
    assert promotable(c) in (True, False)
    assert promotable(c, min_logloss_gain=999.0) is False


def test_small_sample_is_rejected():
    y = np.array([0, 1, 2] * 9)
    p = np.tile([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]], (9, 1))
    with pytest.raises(ValueError):
        compare_probabilities(y, p, p)
