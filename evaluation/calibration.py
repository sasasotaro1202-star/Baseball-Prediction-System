"""Calibration utilities used at the final probability boundary.

The calibrator is deliberately independent from model training so a frozen
calibration artifact can be versioned and applied to production predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float = 1.0
    version: str = "temperature-v1"

    def transform(self, probabilities: Any) -> np.ndarray:
        p = np.asarray(probabilities, dtype=float)
        if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2:
            raise ValueError("probabilities must be a non-empty 2D array with at least two classes")
        if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
            raise ValueError("probabilities must be finite and in [0,1]")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        row_sums = p.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError("each probability row must have positive mass")
        # Normalize first so callers cannot accidentally calibrate an
        # unnormalized model score vector as if it were a probability vector.
        p = p / row_sums[:, None]
        logits = np.log(np.clip(p, 1e-15, 1.0)) / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_temperature(probabilities: Any, y_true: Any, *, grid: np.ndarray | None = None) -> TemperatureCalibration:
    """Fit temperature on a development/calibration set only."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y_true, dtype=int)
    if p.ndim != 2 or p.shape[0] == 0 or len(p) != len(y) or p.shape[1] < 2:
        raise ValueError("probabilities and y_true are incompatible or empty")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("probabilities must be finite and in [0,1]")
    row_sums = p.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("each probability row must have positive mass")
    if np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("y_true contains an invalid class")
    candidates = np.asarray(grid if grid is not None else np.geomspace(0.5, 3.0, 61), dtype=float)
    candidates = candidates[np.isfinite(candidates) & (candidates > 0)]
    if candidates.size == 0:
        raise ValueError("temperature grid must contain at least one positive finite value")
    best_t, best_loss = 1.0, float("inf")
    for t in candidates:
        q = TemperatureCalibration(float(t)).transform(p)
        loss = float(-np.mean(np.log(np.clip(q[np.arange(len(y)), y], 1e-15, 1.0))))
        if math.isfinite(loss) and loss < best_loss:
            best_t, best_loss = float(t), loss
    if not math.isfinite(best_loss):
        raise ValueError("could not fit a finite calibration loss")
    return TemperatureCalibration(best_t)


def calibration_report(y_true: Any, raw: Any, calibrated: Any) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    r = np.asarray(raw, dtype=float)
    c = np.asarray(calibrated, dtype=float)
    if r.ndim != 2 or c.ndim != 2 or len(y) == 0 or len(y) != len(r) or len(y) != len(c) or r.shape != c.shape:
        raise ValueError("calibration arrays have incompatible or empty shapes")
    if np.any(y < 0) or np.any(y >= r.shape[1]):
        raise ValueError("y_true contains an invalid class")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(c)):
        raise ValueError("calibration probabilities must be finite")
    raw_nll = float(-np.mean(np.log(np.clip(r[np.arange(len(y)), y], 1e-15, 1.0))))
    cal_nll = float(-np.mean(np.log(np.clip(c[np.arange(len(y)), y], 1e-15, 1.0))))
    return {"raw_logloss": raw_nll, "calibrated_logloss": cal_nll, "improvement": raw_nll - cal_nll, "rows": int(len(y))}
