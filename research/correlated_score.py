"""Coherent bivariate-Poisson score distribution.

A shared scoring component captures positive game-level correlation while
remaining conservative: the shared intensity is estimated only from
chronological validation residual covariance and is bounded.
"""

from __future__ import annotations

import math
import numpy as np


def poisson_pmf(k: int, lam: float) -> float:
    lam = max(float(lam), 1e-9)
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def grid(lam_home: float, lam_away: float, shared: float = 0.0, max_runs: int = 14) -> np.ndarray:
    """Return a normalized bivariate-Poisson score grid."""
    lh = max(float(lam_home) - float(shared), 1e-6)
    la = max(float(lam_away) - float(shared), 1e-6)
    lc = max(float(shared), 0.0)
    out = np.zeros((max_runs + 1, max_runs + 1), dtype=float)
    # Tail mass is normalized inside the visible grid, matching the existing
    # exact-score contract while preserving the shared component.
    for i in range(max_runs + 1):
        for j in range(max_runs + 1):
            s = 0.0
            for k in range(min(i, j) + 1):
                s += poisson_pmf(i-k, lh) * poisson_pmf(j-k, la) * poisson_pmf(k, lc)
            out[i, j] = s
    total = out.sum()
    return out / total if total > 0 else out


def top_scores(lam_home: float, lam_away: float, shared: float = 0.0, n: int = 4):
    m = grid(lam_home, lam_away, shared)
    cells = [(f"{i}-{j}", float(m[i, j])) for i in range(m.shape[0]) for j in range(m.shape[1])]
    cells.sort(key=lambda x: x[1], reverse=True)
    return cells[:n]


def low_high(lam_home: float, lam_away: float, shared: float = 0.0):
    m = grid(lam_home, lam_away, shared)
    low = float(m[:7, :7].sum())
    # Exact canonical contract: LOW total runs <= 6.
    low = float(sum(m[i, j] for i in range(m.shape[0]) for j in range(m.shape[1]) if i+j <= 6))
    return low, 1.0 - low


def estimate_shared_lambda(home_residuals, away_residuals, max_shared: float = 1.25) -> float:
    """Estimate common-run intensity from OOS residual covariance.

    The estimate is deliberately conservative and never negative. It must be
    computed from a chronological validation window, not target-game rows.
    """
    h = np.asarray(home_residuals, dtype=float)
    a = np.asarray(away_residuals, dtype=float)
    mask = np.isfinite(h) & np.isfinite(a)
    if mask.sum() < 25:
        return 0.0
    cov = float(np.mean(h[mask] * a[mask]))
    return float(np.clip(cov, 0.0, max_shared))
