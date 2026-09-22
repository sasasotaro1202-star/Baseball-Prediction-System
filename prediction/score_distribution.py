"""PIT-safe run-distribution utilities for production baseball predictions.

The production display contract is deliberately separated:
- score_candidates: the four highest-probability *exact* scorelines;
- low/high: probabilities from the complete joint score distribution,
  where LOW is total runs <= 6 and HIGH is total runs >= 7.

The top-four scorelines are never selected to make Low/High look consistent.
This separation prevents a common reporting bug where an arbitrary tail bucket
replaces a genuinely more likely exact scoreline.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from research.correlated_score import grid as correlated_grid


def score_distribution(
    home_lambda: float,
    away_lambda: float,
    *,
    max_runs: int = 14,
    shared_lambda: float = 0.0,
) -> np.ndarray:
    """Return a normalized joint exact-score matrix for 0..max_runs each."""
    if not math.isfinite(float(home_lambda)) or not math.isfinite(float(away_lambda)):
        raise ValueError("run means must be finite")
    if float(home_lambda) <= 0 or float(away_lambda) <= 0:
        raise ValueError("run means must be positive")
    if not math.isfinite(float(shared_lambda)):
        raise ValueError("shared run intensity must be finite")
    if int(max_runs) < 7:
        raise ValueError("max_runs must be at least 7")

    matrix = correlated_grid(
        float(home_lambda),
        float(away_lambda),
        float(shared_lambda),
        max_runs=int(max_runs),
    )
    if not np.isfinite(matrix).all() or matrix.sum() <= 0:
        raise ValueError("invalid score distribution")
    return matrix


def _outputs_from_matrix(matrix: np.ndarray, *, n: int = 4) -> tuple[list[dict[str, Any]], float, float]:
    """Derive all canonical outputs from one already-normalized score matrix."""
    if int(n) != 4:
        raise ValueError("production score contract requires exactly four candidates")
    flat = matrix.ravel()
    indices = sorted(
        range(flat.size),
        key=lambda i: (-float(flat[i]), int(i // matrix.shape[1]), int(i % matrix.shape[1])),
    )[:n]
    candidates: list[dict[str, Any]] = []
    for i in indices:
        home = i // matrix.shape[1]
        away = i % matrix.shape[1]
        candidates.append({"score": f"{home}-{away}", "probability": float(flat[i])})

    low = float(matrix[np.indices(matrix.shape).sum(axis=0) <= 6].sum())
    low = float(np.clip(low, 0.0, 1.0))
    return candidates, low, 1.0 - low


def top_score_candidates(
    home_lambda: float,
    away_lambda: float,
    *,
    n: int = 4,
    max_runs: int = 20,
    shared_lambda: float = 0.0,
) -> list[dict[str, Any]]:
    """Return exactly the n most probable exact scorelines, descending."""
    if int(n) != 4:
        raise ValueError("production score contract requires exactly four candidates")
    matrix = score_distribution(
        home_lambda, away_lambda, max_runs=max_runs, shared_lambda=shared_lambda
    )
    candidates, _, _ = _outputs_from_matrix(matrix, n=n)
    return candidates


def low_high_probabilities(
    home_lambda: float,
    away_lambda: float,
    *,
    max_runs: int = 20,
    shared_lambda: float = 0.0,
) -> tuple[float, float]:
    """Return P(total<=6), P(total>=7) from the full score distribution."""
    matrix = score_distribution(
        home_lambda, away_lambda, max_runs=max_runs, shared_lambda=shared_lambda
    )
    _, low, high = _outputs_from_matrix(matrix)
    return low, high


def build_score_outputs(
    home_lambda: float,
    away_lambda: float,
    *,
    max_runs: int = 20,
    shared_lambda: float = 0.0,
) -> dict[str, Any]:
    """Build the canonical score + Low/High output contract."""
    matrix = score_distribution(
        home_lambda, away_lambda, max_runs=max_runs, shared_lambda=shared_lambda
    )
    candidates, low, high = _outputs_from_matrix(matrix)
    return {
        "score_candidates": candidates,
        "low_probability": low,
        "high_probability": high,
        "low_high_boundary": 6.5,
        "low_definition": "total_runs <= 6",
        "high_definition": "total_runs >= 7",
    }
