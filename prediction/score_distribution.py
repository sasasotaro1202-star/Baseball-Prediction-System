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


def _poisson_pmf(k: int, lam: float) -> float:
    lam = max(float(lam), 1e-9)
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def score_distribution(
    home_lambda: float,
    away_lambda: float,
    *,
    max_runs: int = 20,
) -> np.ndarray:
    """Return a normalized joint exact-score matrix for 0..max_runs each."""
    if not math.isfinite(float(home_lambda)) or not math.isfinite(float(away_lambda)):
        raise ValueError("run means must be finite")
    if float(home_lambda) <= 0 or float(away_lambda) <= 0:
        raise ValueError("run means must be positive")
    if int(max_runs) < 7:
        raise ValueError("max_runs must be at least 7")

    h = np.asarray([_poisson_pmf(k, home_lambda) for k in range(max_runs + 1)], dtype=float)
    a = np.asarray([_poisson_pmf(k, away_lambda) for k in range(max_runs + 1)], dtype=float)
    matrix = np.outer(h, a)
    total = float(matrix.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("invalid score distribution")
    return matrix / total


def top_score_candidates(
    home_lambda: float,
    away_lambda: float,
    *,
    n: int = 4,
    max_runs: int = 20,
) -> list[dict[str, Any]]:
    """Return exactly the n most probable exact scorelines, descending."""
    if int(n) != 4:
        raise ValueError("production score contract requires exactly four candidates")
    matrix = score_distribution(home_lambda, away_lambda, max_runs=max_runs)
    flat = matrix.ravel()
    indices = sorted(
        range(flat.size),
        key=lambda i: (-float(flat[i]), int(i // matrix.shape[1]), int(i % matrix.shape[1])),
    )[:n]
    out = []
    for i in indices:
        home = i // matrix.shape[1]
        away = i % matrix.shape[1]
        out.append({"score": f"{home}-{away}", "probability": float(flat[i])})
    return out


def low_high_probabilities(
    home_lambda: float,
    away_lambda: float,
    *,
    max_runs: int = 20,
) -> tuple[float, float]:
    """Return P(total<=6), P(total>=7) from the full score distribution."""
    matrix = score_distribution(home_lambda, away_lambda, max_runs=max_runs)
    low = 0.0
    for home in range(matrix.shape[0]):
        for away in range(matrix.shape[1]):
            if home + away <= 6:
                low += float(matrix[home, away])
    low = float(np.clip(low, 0.0, 1.0))
    return low, 1.0 - low


def build_score_outputs(
    home_lambda: float,
    away_lambda: float,
    *,
    max_runs: int = 20,
) -> dict[str, Any]:
    """Build the canonical score + Low/High output contract."""
    candidates = top_score_candidates(home_lambda, away_lambda, max_runs=max_runs)
    low, high = low_high_probabilities(home_lambda, away_lambda, max_runs=max_runs)
    return {
        "score_candidates": candidates,
        "low_probability": low,
        "high_probability": high,
        "low_high_boundary": 6.5,
        "low_definition": "total_runs <= 6",
        "high_definition": "total runs >= 7",
    }
