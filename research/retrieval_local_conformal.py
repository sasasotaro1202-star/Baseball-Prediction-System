"""Research-only retrieval-local conformal prediction sets for baseball events.

This candidate localizes the calibration distribution to temporally eligible
historical cases that are similar in a supplied, PIT-safe state vector.

It is deliberately *not* wired into production. Localized conformal calibration
can change empirical coverage under temporal dependence or poor retrieval, so
the output includes explicit diagnostics and never grants promotion.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

EPS = 1e-12


def _timestamps(values: Sequence[Any], *, name: str) -> pd.Series:
    out = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    if out.empty or out.isna().any():
        raise ValueError(f"{name} must be non-empty and timezone-parseable")
    return out


def _probabilities(p: Any, n: int | None = None) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("probabilities must be a 2-D matrix with at least two classes")
    if n is not None and len(arr) != int(n):
        raise ValueError("probabilities must align with timestamps/features")
    if not np.isfinite(arr).all() or (arr < 0.0).any():
        raise ValueError("probabilities must be finite and non-negative")
    total = arr.sum(axis=1, keepdims=True)
    if np.any(total <= EPS):
        raise ValueError("probability rows must have positive mass")
    return arr / total


def _labels(y: Sequence[Any], n: int) -> np.ndarray:
    arr = np.asarray(y, dtype=int).reshape(-1)
    if len(arr) != n or np.any(arr < 0):
        raise ValueError("labels must be non-negative and aligned")
    return arr


def _features(X: Any, n: int) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] < 1:
        raise ValueError("features must be a 2-D matrix aligned with cases")
    if not np.isfinite(arr).all():
        raise ValueError("features must be finite; imputation is not implicit")
    return arr


def _local_pvalues(scores: np.ndarray, p_row: np.ndarray) -> np.ndarray:
    scores = np.sort(np.clip(np.asarray(scores, dtype=float).reshape(-1), 0.0, 1.0))
    thresholds = 1.0 - np.asarray(p_row, dtype=float)
    idx = np.searchsorted(scores, thresholds, side="left")
    n = len(scores)
    return (1.0 + n - idx) / (n + 1.0)


def retrieval_local_conformal_sets(
    y: Sequence[Any],
    probabilities: Any,
    features: Any,
    prediction_times: Sequence[Any],
    outcome_confirmed_at: Sequence[Any],
    *,
    alpha: float = 0.10,
    min_calibration: int = 30,
    max_pool: int = 200,
    k: int = 30,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Construct PIT-safe local conformal sets from retrieved prior cases.

    For target row i, the retrieval pool satisfies:
      prediction_time[j] < prediction_time[i]
      outcome_confirmed_at[j] <= prediction_time[i]

    Only the most recent max_pool eligible rows enter retrieval; the k nearest
    rows in standardized feature space form the calibration set. The target's
    own outcome is never used.
    """
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    if int(min_calibration) < 2:
        raise ValueError("min_calibration must be >= 2")
    if int(max_pool) < int(min_calibration):
        raise ValueError("max_pool must be >= min_calibration")
    if int(k) < int(min_calibration):
        raise ValueError("k must be >= min_calibration")
    if int(k) > int(max_pool):
        raise ValueError("k must be <= max_pool")

    pt = _timestamps(prediction_times, name="prediction_times")
    mature = _timestamps(outcome_confirmed_at, name="outcome_confirmed_at")
    n = len(pt)
    p = _probabilities(probabilities, n)
    labels = _labels(y, n)
    X = _features(features, n)

    if not pt.is_monotonic_increasing:
        raise ValueError("prediction_times must be monotonically non-decreasing")
    if len(mature) != n:
        raise ValueError("outcome_confirmed_at must align with prediction_times")
    if (mature < pt).any():
        raise ValueError("outcome_confirmed_at cannot precede prediction_time")
    if labels.max(initial=-1) >= p.shape[1]:
        raise ValueError("label exceeds probability class count")

    names = [str(x) for x in (
        class_names if class_names is not None else range(p.shape[1])
    )]
    if len(names) != p.shape[1] or len(set(names)) != len(names):
        raise ValueError("class_names must uniquely identify every class")

    prediction_sets: list[list[str]] = []
    actions: list[str] = []
    set_sizes = np.zeros(n, dtype=int)
    retrieval_counts = np.zeros(n, dtype=int)
    calibration_counts = np.zeros(n, dtype=int)
    nearest_distance = np.full(n, np.nan, dtype=float)
    pool_counts = np.zeros(n, dtype=int)
    pvalues = np.full_like(p, np.nan, dtype=float)

    for i in range(n):
        prior = np.arange(i, dtype=int)
        eligible = prior[
            (pt.iloc[prior] < pt.iloc[i]).to_numpy()
            & (mature.iloc[prior] <= pt.iloc[i]).to_numpy()
        ]

        if len(eligible) > int(max_pool):
            eligible = eligible[-int(max_pool):]
        pool_counts[i] = int(len(eligible))

        if len(eligible) < int(min_calibration):
            prediction_sets.append([])
            actions.append("ABSTAIN")
            continue

        # Standardization is estimated only from the temporally eligible pool.
        # This prevents future feature values from entering the target scaling.
        pool_x = X[eligible]
        mu = pool_x.mean(axis=0)
        sd = pool_x.std(axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
        z_pool = (pool_x - mu) / sd
        z_current = (X[i] - mu) / sd
        d2 = np.sum((z_pool - z_current[None, :]) ** 2, axis=1)

        kk = min(int(k), len(eligible))
        local_order = np.argpartition(d2, kk - 1)[:kk]
        local_order = local_order[np.argsort(d2[local_order], kind="mergesort")]
        selected = eligible[local_order]

        calibration_counts[i] = int(len(selected))
        retrieval_counts[i] = int(len(selected))
        nearest_distance[i] = float(np.sqrt(d2[local_order[0]]))

        scores = 1.0 - p[selected, labels[selected]]
        pv = _local_pvalues(scores, p[i])
        pvalues[i] = pv

        included = pv > alpha
        ps = [names[j] for j in np.flatnonzero(included)]
        size = int(included.sum())
        prediction_sets.append(ps)
        set_sizes[i] = size
        actions.append(
            "ABSTAIN" if size == 0 else ("SINGLE" if size == 1 else "SET")
        )

    eligible_rows = calibration_counts >= int(min_calibration)
    return {
        "method": "retrieval_local_conformal",
        "alpha": alpha,
        "class_names": names,
        "prediction_sets": prediction_sets,
        "action": actions,
        "set_size": set_sizes,
        "predicted_class": np.argmax(p, axis=1),
        "pvalues": pvalues,
        "retrieval_pool_count": pool_counts,
        "retrieved_calibration_count": retrieval_counts,
        "calibration_count": calibration_counts,
        "nearest_distance": nearest_distance,
        "eligible_rows": int(eligible_rows.sum()),
        "total_rows": int(n),
        "contract": {
            "prior_prediction_time_strictly_earlier": True,
            "outcome_confirmed_at_le_target_prediction_time": True,
            "same_prediction_time_excluded": True,
            "current_outcome_excluded": True,
            "feature_scaling_uses_only_temporally_eligible_pool": True,
            "bounded_retrieval_pool": True,
            "bounded_calibration": True,
            "production_changed": False,
            "promotion_allowed": False,
        },
        "limitations": {
            "coverage_guarantee_under_temporal_dependence": "not_claimed",
            "localized_retrieval_is_research_candidate": True,
            "requires_fresh_chronological_oos_validation": True,
        },
    }


def retrieval_conformal_metrics(
    result: dict[str, Any],
    y: Sequence[Any],
) -> dict[str, float]:
    labels = np.asarray(y, dtype=int).reshape(-1)
    sets = result["prediction_sets"]
    sizes = np.asarray(result["set_size"], dtype=int)
    counts = np.asarray(result["calibration_count"], dtype=int)
    eligible = counts >= int(np.nanmin(counts[counts > 0])) if np.any(counts > 0) else np.zeros(len(labels), dtype=bool)
    if len(labels) != len(sets) or len(sizes) != len(labels):
        raise ValueError("result/y length mismatch")
    if np.any(labels < 0):
        raise ValueError("y contains negative labels")
    names = [str(x) for x in result["class_names"]]
    contained = np.asarray(
        [
            int(v) < len(names) and names[int(v)] in set(ps)
            for v, ps in zip(labels, sets)
        ],
        dtype=bool,
    )
    return {
        "rows": float(len(labels)),
        "eligible_rows": float(eligible.sum()),
        "coverage": float(contained[eligible].mean()) if eligible.any() else float("nan"),
        "mean_set_size": float(sizes[eligible].mean()) if eligible.any() else float("nan"),
        "singleton_rate": float((sizes[eligible] == 1).mean()) if eligible.any() else float("nan"),
        "empty_rate": float((sizes[eligible] == 0).mean()) if eligible.any() else float("nan"),
    }


__all__ = ["retrieval_local_conformal_sets", "retrieval_conformal_metrics"]
