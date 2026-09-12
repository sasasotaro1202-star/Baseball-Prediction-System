"""Poisson-based score distribution for top-K most likely scorelines.
Stage-0 approximation using historical average scoring. Per output spec,
choices are shown as raw % and do NOT need to sum to 100%.
"""
import math
from itertools import product
import pandas as pd


def _poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def expected_runs_goals(df, home_col="home_score", away_col="away_score"):
    return float(df[home_col].mean()), float(df[away_col].mean())


def score_grid_probabilities(home_lambda, away_lambda, max_score=12):
    grid = []
    for h, a in product(range(max_score + 1), range(max_score + 1)):
        p = _poisson_pmf(h, home_lambda) * _poisson_pmf(a, away_lambda)
        grid.append({"home_score": h, "away_score": a, "probability": p})
    return grid


def top_k_scores(home_lambda, away_lambda, k, max_score=12):
    grid = score_grid_probabilities(home_lambda, away_lambda, max_score)
    grid.sort(key=lambda r: r["probability"], reverse=True)
    top = grid[:k]
    return [
        {"home_score": r["home_score"], "away_score": r["away_score"],
         "probability_pct": round(r["probability"] * 100, 2)}
        for r in top
    ]
