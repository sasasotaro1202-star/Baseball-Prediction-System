"""Chronological conformal prediction with explicit outcome-maturity gating.

Research-only by default. Calibration for case i may use only earlier
prediction times whose outcomes were confirmed at or before case i's
prediction time. This prevents retrospective outcome feedback from entering
a historical prediction point.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

EPS = 1e-12


def _probability_matrix(p: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{name} must be a 2-D matrix with at least two classes")
    if not np.isfinite(arr).all() or (arr < 0.0).any():
        raise ValueError(f"{name} must contain finite non-negative probabilities")
    total = arr.sum(axis=1, keepdims=True)
    if np.any(total <= EPS):
        raise ValueError(f"{name} contains a zero-mass row")
    return arr / total


def _labels(y: Any, n: int) -> np.ndarray:
    arr = np.asarray(y, dtype=int)
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError("y must be 1-D and aligned with probabilities")
    if np.any(arr < 0):
        raise ValueError("y contains negative labels")
    return arr


def _timestamps(values: Sequence[Any], *, name: str) -> pd.Series:
    out = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    if out.empty or out.isna().any():
        raise ValueError(f"{name} must be non-empty and timezone-parseable")
    return out


def _class_names(class_names: Sequence[str] | None, n_classes: int) -> list[str]:
    names = [str(x) for x in (class_names if class_names is not None else range(n_classes))]
    if len(names) != n_classes or len(set(names)) != n_classes:
        raise ValueError("class_names must uniquely identify every class")
    return names


def _pvalues(scores_sorted: np.ndarray, p_row: np.ndarray) -> np.ndarray:
    thresholds = 1.0 - p_row
    idx = np.searchsorted(scores_sorted, thresholds, side="left")
    n = len(scores_sorted)
    return (1.0 + n - idx) / (n + 1.0)


def validate_maturity_contract(
    prediction_times: Sequence[Any],
    outcome_confirmed_at: Sequence[Any],
) -> dict[str, Any]:
    pt = _timestamps(prediction_times, name="prediction_times")
    mature = _timestamps(outcome_confirmed_at, name="outcome_confirmed_at")
    if len(pt) != len(mature):
        raise ValueError("outcome_confirmed_at must align with prediction_times")
    if not pt.is_monotonic_increasing:
        raise ValueError("prediction_times must be monotonically non-decreasing")
    if (mature < pt).any():
        raise ValueError("outcome_confirmed_at cannot precede prediction_time")
    return {
        "status": "PASS",
        "rows": int(len(pt)),
        "earliest_prediction_time": pt.min().isoformat(),
        "latest_prediction_time": pt.max().isoformat(),
        "latest_outcome_confirmed_at": mature.max().isoformat(),
    }


def walk_forward_split_conformal(
    y: Sequence[int],
    probabilities: Any,
    prediction_times: Sequence[Any],
    outcome_confirmed_at: Sequence[Any],
    *,
    alpha: float = 0.10,
    class_names: Sequence[str] | None = None,
    min_calibration: int = 30,
    max_calibration: int | None = None,
) -> dict[str, Any]:
    """Build one conformal set per chronological case with maturity gating."""
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    if int(min_calibration) < 1:
        raise ValueError("min_calibration must be >= 1")
    if max_calibration is not None and int(max_calibration) < int(min_calibration):
        raise ValueError("max_calibration must be >= min_calibration")

    p = _probability_matrix(probabilities, name="probabilities")
    y_arr = _labels(y, len(p))
    pt = _timestamps(prediction_times, name="prediction_times")
    mature = _timestamps(outcome_confirmed_at, name="outcome_confirmed_at")
    if len(pt) != len(p) or len(mature) != len(p):
        raise ValueError("time arrays must align with probabilities")
    validate_maturity_contract(pt, mature)

    if y_arr.max(initial=-1) >= p.shape[1]:
        raise ValueError("y contains a class id outside probability columns")
    names = _class_names(class_names, p.shape[1])
    scores = 1.0 - p[np.arange(len(p)), y_arr]
    predicted = np.argmax(p, axis=1)

    sets: list[list[str]] = []
    actions: list[str] = []
    set_sizes: list[int] = []
    calibration_counts: list[int] = []
    calibration_latest: list[str | None] = []
    pvalues = np.full_like(p, np.nan, dtype=float)

    for i in range(len(p)):
        prior = np.arange(i)
        mature_mask = mature.iloc[:i].le(pt.iloc[i]).to_numpy()
        earlier_prediction_mask = pt.iloc[:i].lt(pt.iloc[i]).to_numpy()
        calibration_idx = prior[mature_mask & earlier_prediction_mask]
        if max_calibration is not None and len(calibration_idx) > int(max_calibration):
            calibration_idx = calibration_idx[-int(max_calibration):]

        count = int(len(calibration_idx))
        calibration_counts.append(count)
        calibration_latest.append(
            mature.iloc[int(calibration_idx[-1])].isoformat() if count else None
        )

        if count < int(min_calibration):
            sets.append([])
            actions.append("ABSTAIN")
            set_sizes.append(0)
            continue

        sorted_scores = np.sort(np.clip(scores[calibration_idx], 0.0, 1.0))
        pv = _pvalues(sorted_scores, p[i])
        pvalues[i] = pv
        included = pv > alpha
        label_set = [names[j] for j in np.flatnonzero(included)]
        size = int(included.sum())
        sets.append(label_set)
        actions.append("ABSTAIN" if size == 0 else ("SINGLE" if size == 1 else "SET"))
        set_sizes.append(size)

    eligible = np.asarray(calibration_counts, dtype=int) >= int(min_calibration)
    return {
        "method": "walk_forward_split_conformal",
        "alpha": alpha,
        "class_names": names,
        "prediction_sets": sets,
        "action": actions,
        "set_size": np.asarray(set_sizes, dtype=int),
        "predicted_class": predicted,
        "pvalues": pvalues,
        "calibration_count": np.asarray(calibration_counts, dtype=int),
        "calibration_latest_outcome_confirmed_at": calibration_latest,
        "eligible_rows": int(eligible.sum()),
        "total_rows": int(len(p)),
        "min_calibration": int(min_calibration),
        "max_calibration": int(max_calibration) if max_calibration is not None else None,
        "maturity_contract": {
            "status": "PASS",
            "requires_outcome_confirmed_at": True,
            "rule": "prior prediction_time < target prediction_time AND outcome_confirmed_at <= target prediction_time",
            "same_prediction_time_excluded": True,
        },
    }


def temporal_prediction_set_metrics(
    result: Mapping[str, Any],
    y: Sequence[int],
) -> dict[str, float]:
    y_arr = np.asarray(y, dtype=int)
    sets = result["prediction_sets"]
    sizes = np.asarray(result["set_size"], dtype=int)
    pred = np.asarray(result["predicted_class"], dtype=int)
    counts = np.asarray(result["calibration_count"], dtype=int)
    eligible = counts >= int(result["min_calibration"])

    if y_arr.ndim != 1 or len(y_arr) != len(sets):
        raise ValueError("y must align with prediction sets")
    if np.any(y_arr < 0):
        raise ValueError("y contains negative labels")

    names = [str(x) for x in result["class_names"]]
    contained = np.asarray(
        [
            int(label) < len(names) and names[int(label)] in set(ps)
            for label, ps in zip(y_arr, sets)
        ],
        dtype=bool,
    )
    singleton = sizes == 1
    singleton_eligible = eligible & singleton

    return {
        "total_rows": float(len(y_arr)),
        "eligible_rows": float(eligible.sum()),
        "maturity_gated_fraction": float(eligible.mean()) if len(eligible) else float("nan"),
        "coverage_eligible": float(contained[eligible].mean()) if eligible.any() else float("nan"),
        "mean_set_size_eligible": float(sizes[eligible].mean()) if eligible.any() else float("nan"),
        "singleton_rate_eligible": float(singleton[eligible].mean()) if eligible.any() else float("nan"),
        "singleton_accuracy_eligible": float((pred[singleton_eligible] == y_arr[singleton_eligible]).mean())
        if singleton_eligible.any() else float("nan"),
        "abstain_rate_total": float((sizes == 0).mean()) if len(sizes) else float("nan"),
        "abstain_rate_eligible": float((sizes[eligible] == 0).mean()) if eligible.any() else float("nan"),
    }
