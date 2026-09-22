"""Coherent bivariate-Poisson score distribution.

A shared scoring component captures positive game-level correlation while
remaining conservative: the shared intensity is estimated only from
chronological validation residual covariance and is bounded.
"""

from __future__ import annotations

import math
import numpy as np


def poisson_pmf(k: int, lam: float) -> float:
    value = float(lam)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Poisson intensity must be finite and positive")
    if int(k) < 0 or int(k) != k:
        raise ValueError("Poisson count must be a non-negative integer")
    return math.exp(-value + int(k) * math.log(value) - math.lgamma(int(k) + 1))


def grid(lam_home: float, lam_away: float, shared: float = 0.0, max_runs: int = 14) -> np.ndarray:
    """Return a normalized bivariate-Poisson score grid."""
    if not all(math.isfinite(float(x)) for x in (lam_home, lam_away, shared)):
        raise ValueError("score intensities must be finite")
    if float(lam_home) <= 0 or float(lam_away) <= 0 or float(shared) < 0:
        raise ValueError("score intensities must be positive/non-negative")
    if int(max_runs) < 7 or int(max_runs) != max_runs:
        raise ValueError("max_runs must be an integer >= 7")
    # A bivariate-Poisson shared component cannot exceed either marginal
    # intensity. Clamp it so the requested marginal means remain valid rather
    # than silently changing the model when a noisy estimator overshoots.
    lh_raw = max(float(lam_home), 1e-9)
    la_raw = max(float(lam_away), 1e-9)
    lc = min(max(float(shared), 0.0), lh_raw, la_raw)
    lh = max(lh_raw - lc, 1e-9)
    la = max(la_raw - lc, 1e-9)
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



def npb_final_outcomes(lam_home: float, lam_away: float, shared: float = 0.0,
                       max_runs: int = 14, extra_innings: int = 3):
    """Return coherent NPB final-result probabilities.

    The grid is the nine-inning distribution. A regulation tie is propagated
    through up to three extra innings; only a tie after the final inning is a
    final draw. This prevents 9-inning score ties from being treated as final
    game draws.
    """
    m = grid(lam_home, lam_away, shared, max_runs=max_runs)
    home_reg = float(np.tril(m, -1).sum())
    away_reg = float(np.triu(m, 1).sum())
    tie_reg = float(np.trace(m))

    lh_i = max(float(lam_home) / 9.0, 1e-9)
    la_i = max(float(lam_away) / 9.0, 1e-9)
    max_i = 12
    ph = np.array([poisson_pmf(k, lh_i) for k in range(max_i + 1)])
    pa = np.array([poisson_pmf(k, la_i) for k in range(max_i + 1)])
    ph /= ph.sum()
    pa /= pa.sum()

    state = {0: 1.0}
    h_abs = a_abs = 0.0
    for _ in range(int(extra_innings)):
        nxt = {}
        for d, pd in state.items():
            for h, hp in enumerate(ph):
                for a, ap in enumerate(pa):
                    q = pd * hp * ap
                    nd = d + h - a
                    if nd > 0:
                        h_abs += q
                    elif nd < 0:
                        a_abs += q
                    else:
                        nxt[0] = nxt.get(0, 0.0) + q
        state = nxt

    draw_cond = float(state.get(0, 0.0))
    home = home_reg + tie_reg * h_abs
    draw = tie_reg * draw_cond
    away = away_reg + tie_reg * a_abs
    z = home + draw + away
    if z <= 0 or not np.isfinite(z):
        raise RuntimeError("invalid NPB final-outcome normalization")
    return float(home/z), float(draw/z), float(away/z)

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
