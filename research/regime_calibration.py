"""Bounded regime-relative calibration for chronological OOS evidence.

The function operates only on already-calibrated validation probabilities and
never touches holdout rows. Sparse regimes are shrunk toward the neutral
temperature 1.0 with a fixed prior strength to limit variance.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.calibration import fit_temperature



def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Small dependency-free multiclass log-loss used by the calibration helper."""
    labels = np.asarray(y, dtype=int)
    probs = np.asarray(p, dtype=float)
    if probs.ndim != 2 or len(labels) != len(probs) or len(labels) == 0:
        raise ValueError("invalid target/probability shapes")
    if (labels < 0).any() or (labels >= probs.shape[1]).any():
        raise ValueError("target labels are outside probability columns")
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    rows = np.arange(len(labels))
    return float(-np.mean(np.log(np.clip(probs[rows, labels], 1e-15, 1.0))))

def _apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    x = np.asarray(p, dtype=float)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("probabilities must be a non-empty 2-D matrix")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    x = np.clip(x, 1e-12, 1.0)
    z = np.log(x) / float(temperature)
    z -= np.max(z, axis=1, keepdims=True)
    q = np.exp(z)
    q /= q.sum(axis=1, keepdims=True)
    return q


def fit_regime_relative_temperatures(
    probabilities: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    *,
    min_rows: int = 120,
    prior_strength: float = 240.0,
) -> dict[str, Any]:
    """Fit shrinkage temperatures by regime on validation OOS rows.

    Returns the transformed probabilities for the same rows plus the chosen
    relative temperature for every observed regime. A temperature of 1.0
    means the regime receives no additional calibration.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y, dtype=int)
    labels = np.asarray(labels).astype(str)
    if p.ndim != 2 or len(p) == 0 or len(p) != len(y) or len(p) != len(labels):
        raise ValueError("probabilities, y, and labels must have equal non-zero rows")
    if len(np.unique(y)) < 2:
        raise ValueError("validation target must contain at least two classes")
    if min_rows < 2 or not np.isfinite(prior_strength) or prior_strength <= 0:
        raise ValueError("invalid regime calibration constraints")

    base = p.copy()
    out = base.copy()
    temperatures: dict[str, float] = {}
    counts: dict[str, int] = {}

    for regime in np.unique(labels):
        mask = labels == regime
        n = int(mask.sum())
        counts[str(regime)] = n
        if n < int(min_rows) or len(np.unique(y[mask])) < 2:
            temperatures[str(regime)] = 1.0
            continue

        base_ll = float(_log_loss(y[mask], base[mask]))
        fitted = fit_temperature(base[mask], y[mask])
        raw_t = float(fitted.temperature)

        # Shrink multiplicatively in log-temperature space toward neutral.
        alpha = float(n / (n + prior_strength))
        shrunk_t = float(np.exp(alpha * np.log(raw_t)))
        candidate = _apply_temperature(base[mask], shrunk_t)
        candidate_ll = float(_log_loss(y[mask], candidate))

        if candidate_ll < base_ll - 1e-9:
            temperatures[str(regime)] = shrunk_t
            out[mask] = candidate
        else:
            temperatures[str(regime)] = 1.0

    return {
        "probabilities": out,
        "temperatures": temperatures,
        "counts": counts,
    }
