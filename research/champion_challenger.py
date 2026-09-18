"""Conservative champion/challenger evaluation utilities.

These utilities are deliberately model-agnostic. They compare two already-frozen
OOS prediction ledgers without refitting either candidate, which prevents the
promotion decision itself from becoming another source of selection leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, log_loss


@dataclass(frozen=True)
class CandidateComparison:
    n: int
    incumbent_logloss: float
    challenger_logloss: float
    incumbent_brier: float
    challenger_brier: float
    incumbent_accuracy: float
    challenger_accuracy: float
    logloss_delta: float
    brier_delta: float
    accuracy_delta: float
    paired_loss_win_rate: float
    paired_logloss_bootstrap_low: float
    paired_logloss_bootstrap_high: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def _paired_bootstrap_delta(inc_loss: np.ndarray, chal_loss: np.ndarray, *, seed: int, rounds: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    delta = np.asarray(inc_loss, dtype=float) - np.asarray(chal_loss, dtype=float)
    if len(delta) < 2:
        x = float(delta.mean()) if len(delta) else 0.0
        return x, x
    idx = rng.integers(0, len(delta), size=(rounds, len(delta)))
    samples = delta[idx].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def compare_probabilities(y_true, incumbent, challenger, *, seed: int = 42, bootstrap_rounds: int = 2000) -> CandidateComparison:
    y = np.asarray(y_true, dtype=int)
    inc = np.asarray(incumbent, dtype=float)
    chal = np.asarray(challenger, dtype=float)
    if inc.shape != chal.shape or len(y) != len(inc):
        raise ValueError("y_true, incumbent and challenger shapes must agree")
    if len(y) < 30:
        raise ValueError("at least 30 paired OOS predictions are required")
    if inc.ndim != 2 or inc.shape[1] < 2:
        raise ValueError("probability matrices must be 2-D with at least two classes")
    inc = np.clip(inc, 1e-7, 1.0); inc /= inc.sum(axis=1, keepdims=True)
    chal = np.clip(chal, 1e-7, 1.0); chal /= chal.sum(axis=1, keepdims=True)
    inc_loss = -np.log(inc[np.arange(len(y)), y])
    chal_loss = -np.log(chal[np.arange(len(y)), y])
    low, high = _paired_bootstrap_delta(inc_loss, chal_loss, seed=seed, rounds=bootstrap_rounds)
    return CandidateComparison(
        n=int(len(y)),
        incumbent_logloss=float(log_loss(y, inc, labels=list(range(inc.shape[1])))),
        challenger_logloss=float(log_loss(y, chal, labels=list(range(chal.shape[1])))),
        incumbent_brier=multiclass_brier(y, inc),
        challenger_brier=multiclass_brier(y, chal),
        incumbent_accuracy=float(accuracy_score(y, inc.argmax(axis=1))),
        challenger_accuracy=float(accuracy_score(y, chal.argmax(axis=1))),
        logloss_delta=float(log_loss(y, inc, labels=list(range(inc.shape[1]))) - log_loss(y, chal, labels=list(range(chal.shape[1])))),
        brier_delta=float(multiclass_brier(y, inc) - multiclass_brier(y, chal)),
        accuracy_delta=float(accuracy_score(y, chal.argmax(axis=1)) - accuracy_score(y, inc.argmax(axis=1))),
        paired_loss_win_rate=float(np.mean(chal_loss < inc_loss)),
        paired_logloss_bootstrap_low=low,
        paired_logloss_bootstrap_high=high,
    )


def promotable(c: CandidateComparison, *, min_logloss_gain: float = 0.0, min_brier_gain: float = 0.0, max_accuracy_drop: float = 0.005, require_bootstrap_positive: bool = True) -> bool:
    """Return True only when a challenger clears conservative quality gates."""
    if c.n < 100:
        return False
    if c.logloss_delta <= min_logloss_gain:
        return False
    if c.brier_delta <= min_brier_gain:
        return False
    if c.accuracy_delta < -abs(max_accuracy_drop):
        return False
    if require_bootstrap_positive and c.paired_logloss_bootstrap_low <= 0.0:
        return False
    return True
