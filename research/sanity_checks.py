"""Fail-closed statistical sanity guards for baseball prediction research.

These helpers are deliberately model-agnostic. They do not improve a model by
themselves; they prevent apparently strong results from being accepted when
probabilities are malformed or a target signal survives permutation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np

@dataclass(frozen=True)
class PermutationCheck:
    observed: float
    shuffled_mean: float
    shuffled_std: float
    shuffled_scores: tuple[float, ...]
    suspicious: bool

def validate_probability_rows(probabilities: Sequence[Sequence[float]], *, tolerance: float = 1e-8) -> None:
    """Raise if probability rows are non-finite, negative, or not normalized."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    for i, row in enumerate(probabilities):
        values = [float(x) for x in row]
        if not values:
            raise ValueError(f"empty probability row at index {i}")
        if any(not math.isfinite(x) for x in values):
            raise ValueError(f"non-finite probability at index {i}")
        if any(x < -tolerance or x > 1.0 + tolerance for x in values):
            raise ValueError(f"probability outside [0,1] at index {i}")
        total = sum(values)
        if abs(total - 1.0) > tolerance:
            raise ValueError(f"probability row {i} sums to {total}, not 1")

def target_permutation_check(
    y_true: Sequence[int],
    predict_proba: Callable[[Sequence[int]], Sequence[Sequence[float]]],
    score: Callable[[Sequence[int], Sequence[Sequence[float]]], float],
    *, n_permutations: int = 25, seed: int = 42, min_gap_sd: float = 3.0,
) -> PermutationCheck:
    """Shuffle targets while keeping the rest of the evaluation path fixed.

    Callers should rerun the identical training/evaluation path with only y
    shuffled. A result is suspicious when the observed score is not separated
    from the shuffled distribution by the configured standard-deviation gap.
    """
    if n_permutations < 5:
        raise ValueError("n_permutations must be >= 5")
    if min_gap_sd <= 0:
        raise ValueError("min_gap_sd must be > 0")
    y = np.asarray(list(y_true))
    if y.ndim != 1 or y.size < 2:
        raise ValueError("y_true must contain at least two labels")
    observed_pred = predict_proba(y.tolist())
    observed = float(score(y.tolist(), observed_pred))
    if not math.isfinite(observed):
        raise ValueError("observed score must be finite")
    rng = np.random.default_rng(seed)
    shuffled: list[float] = []
    for _ in range(n_permutations):
        perm = y.copy()
        rng.shuffle(perm)
        value = float(score(perm.tolist(), predict_proba(perm.tolist())))
        if not math.isfinite(value):
            raise ValueError("permutation score must be finite")
        shuffled.append(value)
    mean = float(np.mean(shuffled))
    std = float(np.std(shuffled, ddof=1))
    gap = observed - mean
    suspicious = (std == 0.0 and gap <= 0.0) or (std > 0.0 and gap < min_gap_sd * std)
    return PermutationCheck(observed, mean, std, tuple(shuffled), bool(suspicious))
