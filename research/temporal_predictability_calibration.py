"""PIT-safe walk-forward calibration for a predictability score.

Research-only. A target case may use calibration outcomes only when:
- the calibration prediction_time is strictly earlier than the target prediction_time;
- the calibration outcome was confirmed at or before the target prediction_time.

Rows with tied prediction times, unavailable/invalid maturity timestamps, or
insufficient history are never used to fit a calibrator. The module does not
modify production predictions.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 1e-12


def _timestamps(values: Sequence[Any], *, name: str) -> pd.Series:
    out = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    if out.empty or out.isna().any():
        raise ValueError(f"{name} must be non-empty and timezone-parseable")
    return out


def _validate_binary(values: Sequence[Any], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain finite values")
    if not np.isin(arr, [0.0, 1.0]).all():
        raise ValueError(f"{name} must contain only 0/1 values")
    return arr.astype(int)


def _validate_score(values: Sequence[Any], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain finite values")
    if ((arr < 0.0) | (arr > 1.0)).any():
        raise ValueError(f"{name} must be within [0,1]")
    return arr


def _fit_or_fallback(
    x: np.ndarray,
    y: np.ndarray,
    current: float,
    *,
    min_calibration: int,
    prior_mean: float,
) -> tuple[float, str]:
    if len(y) < int(min_calibration) or len(np.unique(y)) < 2:
        return float(np.clip(prior_mean, 0.01, 0.99)), "FALLBACK_PRIOR_MEAN"

    model = LogisticRegression(C=0.50, max_iter=1000, random_state=13013)
    model.fit(x.reshape(-1, 1), y)
    value = float(model.predict_proba(np.asarray([[current]], dtype=float))[0, 1])
    return float(np.clip(value, 0.01, 0.99)), "FITTED_PRIOR_ONLY_LOGISTIC"


def calibrate_predictability_walk_forward(
    raw_predictability: Sequence[float],
    correctness: Sequence[int],
    prediction_times: Sequence[Any],
    outcome_confirmed_at: Sequence[Any],
    *,
    min_calibration: int = 30,
    max_calibration: int = 120,
    prior_mean: float = 0.5,
) -> dict[str, Any]:
    """Calibrate predictability using only temporally mature prior outcomes.

    The returned score for row i never uses correctness from row i or from a
    later/tied prediction time. A bounded rolling calibration history is used.
    """
    if int(min_calibration) < 2:
        raise ValueError("min_calibration must be >= 2")
    if int(max_calibration) < int(min_calibration):
        raise ValueError("max_calibration must be >= min_calibration")
    if not np.isfinite(float(prior_mean)) or not 0.0 <= float(prior_mean) <= 1.0:
        raise ValueError("prior_mean must be finite and within [0,1]")

    raw = _validate_score(raw_predictability, name="raw_predictability")
    y = _validate_binary(correctness, name="correctness")
    pt = _timestamps(prediction_times, name="prediction_times")
    mature = _timestamps(outcome_confirmed_at, name="outcome_confirmed_at")

    n = len(raw)
    if len(y) != n or len(pt) != n or len(mature) != n:
        raise ValueError("all inputs must have equal length")

    if not pt.is_monotonic_increasing:
        raise ValueError("prediction_times must be monotonically non-decreasing")
    if (mature < pt).any():
        raise ValueError("outcome_confirmed_at cannot precede prediction_time")

    calibrated = np.full(n, np.nan, dtype=float)
    counts = np.zeros(n, dtype=int)
    methods: list[str] = []
    latest_maturity: list[str | None] = []

    for i in range(n):
        target_pt = pt.iloc[i]
        prior = np.arange(i, dtype=int)
        eligible = prior[
            (pt.iloc[prior] < target_pt).to_numpy()
            & (mature.iloc[prior] <= target_pt).to_numpy()
        ]

        if len(eligible) > int(max_calibration):
            eligible = eligible[-int(max_calibration):]

        counts[i] = int(len(eligible))
        latest_maturity.append(
            mature.iloc[int(eligible[-1])].isoformat() if len(eligible) else None
        )

        x = raw[eligible]
        yy = y[eligible]
        value, method = _fit_or_fallback(
            x,
            yy,
            raw[i],
            min_calibration=int(min_calibration),
            prior_mean=float(prior_mean),
        )
        calibrated[i] = value
        methods.append(method)

    return {
        "status": "EXECUTED_PIT_SAFE",
        "calibrated_predictability": calibrated,
        "calibration_count": counts,
        "method": methods,
        "latest_calibration_outcome_confirmed_at": latest_maturity,
        "min_calibration": int(min_calibration),
        "max_calibration": int(max_calibration),
        "contract": {
            "prior_prediction_time_strictly_earlier": True,
            "outcome_confirmed_at_le_target_prediction_time": True,
            "same_prediction_time_excluded": True,
            "bounded_history": True,
            "current_outcome_excluded": True,
            "production_changed": False,
            "promotion_allowed": False,
        },
    }


def calibration_diagnostics(
    correctness: Sequence[int],
    raw_predictability: Sequence[float],
    calibrated_predictability: Sequence[float],
) -> dict[str, float]:
    """Return Brier/ECE-like diagnostics for the binary correctness target."""
    y = _validate_binary(correctness, name="correctness")
    raw = _validate_score(raw_predictability, name="raw_predictability")
    calibrated = _validate_score(calibrated_predictability, name="calibrated_predictability")
    if len(y) != len(raw) or len(y) != len(calibrated):
        raise ValueError("diagnostic inputs must align")

    def _brier(p: np.ndarray) -> float:
        return float(np.mean((p - y) ** 2))

    def _ece(p: np.ndarray, bins: int = 10) -> float:
        edges = np.linspace(0.0, 1.0, bins + 1)
        result = 0.0
        for j in range(bins):
            hi = p <= edges[j + 1] if j == bins - 1 else p < edges[j + 1]
            mask = (p >= edges[j]) & hi
            if mask.any():
                result += float(mask.mean()) * abs(
                    float(p[mask].mean()) - float(y[mask].mean())
                )
        return float(result)

    return {
        "rows": float(len(y)),
        "raw_brier": _brier(raw),
        "calibrated_brier": _brier(calibrated),
        "raw_ece": _ece(raw),
        "calibrated_ece": _ece(calibrated),
    }


__all__ = [
    "calibrate_predictability_walk_forward",
    "calibration_diagnostics",
]
