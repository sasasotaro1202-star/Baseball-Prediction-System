import numpy as np
import pytest

from evaluation.uncertainty import paired_block_bootstrap


def test_paired_block_bootstrap_is_deterministic():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=180)
    base = np.column_stack([0.55 * np.ones(len(y)), 0.45 * np.ones(len(y))])
    cand = base.copy()
    cand[np.arange(len(y)), y] += 0.08
    cand /= cand.sum(axis=1, keepdims=True)

    a = paired_block_bootstrap(y=y, baseline_proba=base, candidate_proba=cand,
                               block_size=15, replications=120, seed=9)
    b = paired_block_bootstrap(y=y, baseline_proba=base, candidate_proba=cand,
                               block_size=15, replications=120, seed=9)
    assert a == b
    assert a.improvement["LogLoss"] > 0
    assert a.improvement["Brier"] > 0
    assert a.improvement["Accuracy"] >= 0


def test_paired_block_bootstrap_rejects_small_holdout():
    y = np.array([0, 1] * 20)
    p = np.column_stack([np.full(len(y), 0.6), np.full(len(y), 0.4)])
    with pytest.raises(ValueError):
        paired_block_bootstrap(
            y=y, baseline_proba=p, candidate_proba=p,
            block_size=10, replications=120,
        )
