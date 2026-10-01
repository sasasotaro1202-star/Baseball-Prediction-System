"""PIT-safe conformal prediction sets for baseball classification.

Research-only by default. The conformalizer consumes calibration predictions
from a strictly earlier chronological slice and produces a prediction set for
future cases. It does not alter the incumbent production model or promotion
status.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

EPS = 1e-12
DEFAULT_METHOD = "split_conformal"
ALLOWED_METHODS = {"split_conformal", "group_split_conformal"}
ALLOWED_ACTIONS = {"SINGLE", "SET", "ABSTAIN"}


def _validate_probability_matrix(p: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{name} must be a 2-D matrix with at least two classes")
    if not np.isfinite(arr).all() or (arr < 0.0).any():
        raise ValueError(f"{name} must contain finite non-negative probabilities")
    row_sum = arr.sum(axis=1, keepdims=True)
    if np.any(row_sum <= EPS):
        raise ValueError(f"{name} contains a zero-mass row")
    return arr / row_sum


def _validate_labels(y: Any, n: int, *, name: str) -> np.ndarray:
    arr = np.asarray(y, dtype=int)
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError(f"{name} must be 1-D and aligned with probabilities")
    if np.any(arr < 0):
        raise ValueError(f"{name} contains negative class ids")
    return arr


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not 0.0 < value < 1.0:
        raise ValueError("alpha must be in (0,1)")
    return value


def validate_chronological_calibration(
    calibration_prediction_times: Sequence[Any],
    test_prediction_times: Sequence[Any],
    *,
    calibration_available_at: Sequence[Any] | None = None,
    test_available_at: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Fail-closed temporal contract for calibration -> future test usage."""
    cal = pd.to_datetime(pd.Series(calibration_prediction_times), utc=True, errors="coerce")
    test = pd.to_datetime(pd.Series(test_prediction_times), utc=True, errors="coerce")
    if cal.empty or test.empty or cal.isna().any() or test.isna().any():
        raise ValueError("prediction times must be non-empty and timezone-parseable")
    if cal.max() >= test.min():
        raise ValueError("calibration predictions must be strictly earlier than test predictions")

    if calibration_available_at is not None:
        ca = pd.to_datetime(pd.Series(calibration_available_at), utc=True, errors="coerce")
        if len(ca) != len(cal) or ca.isna().any() or (ca > cal).any():
            raise ValueError("calibration available_at must be valid and <= calibration prediction_time")

    if test_available_at is not None:
        ta = pd.to_datetime(pd.Series(test_available_at), utc=True, errors="coerce")
        if len(ta) != len(test) or ta.isna().any() or (ta > test).any():
            raise ValueError("test available_at must be valid and <= test prediction_time")

    return {
        "status": "PASS",
        "calibration_rows": int(len(cal)),
        "test_rows": int(len(test)),
        "calibration_latest_prediction_time": cal.max().isoformat(),
        "test_earliest_prediction_time": test.min().isoformat(),
    }


def _prediction_set_from_pvalues(
    pvalue: np.ndarray,
    alpha: float,
    class_names: Sequence[str],
) -> tuple[list[list[str]], np.ndarray]:
    included = pvalue > alpha
    sets = [
        [str(class_names[j]) for j in np.flatnonzero(row)]
        for row in included
    ]
    sizes = included.sum(axis=1).astype(int)
    return sets, sizes


def split_conformal_prediction_sets(
    y_calibration: Sequence[int],
    p_calibration: Any,
    p_test: Any,
    *,
    alpha: float = 0.10,
    class_names: Sequence[str] | None = None,
    min_calibration: int = 30,
) -> dict[str, Any]:
    """Distribution-free split-conformal prediction sets.

    Nonconformity score is 1 - predicted probability assigned to the realized
    class on the calibration slice. Test classes receive finite-sample
    p-values using the standard +1 correction.
    """
    alpha = _validate_alpha(alpha)
    p_cal = _validate_probability_matrix(p_calibration, name="p_calibration")
    p_out = _validate_probability_matrix(p_test, name="p_test")
    if p_cal.shape[1] != p_out.shape[1]:
        raise ValueError("calibration/test class counts differ")
    y_cal = _validate_labels(y_calibration, len(p_cal), name="y_calibration")
    if len(p_cal) < int(min_calibration):
        raise ValueError("calibration slice is too small")
    if y_cal.max(initial=-1) >= p_cal.shape[1]:
        raise ValueError("calibration label exceeds class count")

    names = [str(x) for x in (class_names if class_names is not None else range(p_cal.shape[1]))]
    if len(names) != p_cal.shape[1] or len(set(names)) != len(names):
        raise ValueError("class_names must uniquely identify every class")

    scores = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]
    scores = np.sort(np.clip(scores, 0.0, 1.0))

    # rows=test cases, columns=class labels
    thresholds = 1.0 - p_out
    pvalues = np.empty_like(thresholds)
    n = len(scores)
    for j in range(pvalues.shape[1]):
        idx = np.searchsorted(scores, thresholds[:, j], side="left")
        pvalues[:, j] = (1.0 + n - idx) / (n + 1.0)

    prediction_sets, set_sizes = _prediction_set_from_pvalues(pvalues, alpha, names)
    predicted_class = np.argmax(p_out, axis=1)
    predicted_class_pvalue = pvalues[np.arange(len(p_out)), predicted_class]
    actions = np.where(set_sizes == 0, "ABSTAIN", np.where(set_sizes == 1, "SINGLE", "SET"))

    return {
        "method": DEFAULT_METHOD,
        "alpha": alpha,
        "class_names": names,
        "pvalues": pvalues,
        "prediction_sets": prediction_sets,
        "set_size": set_sizes,
        "predicted_class": predicted_class,
        "predicted_class_pvalue": predicted_class_pvalue,
        "action": actions.tolist(),
        "calibration_size": int(n),
        "finite_sample_correction": True,
    }


def group_split_conformal_prediction_sets(
    y_calibration: Sequence[int],
    p_calibration: Any,
    groups_calibration: Sequence[Any],
    p_test: Any,
    groups_test: Sequence[Any],
    *,
    alpha: float = 0.10,
    class_names: Sequence[str] | None = None,
    min_group_size: int = 30,
) -> dict[str, Any]:
    """Group-conditional conformal sets with deterministic global fallback."""
    alpha = _validate_alpha(alpha)
    if int(min_group_size) < 20:
        raise ValueError("min_group_size must be >= 20")
    p_cal = _validate_probability_matrix(p_calibration, name="p_calibration")
    p_out = _validate_probability_matrix(p_test, name="p_test")
    if p_cal.shape[1] != p_out.shape[1]:
        raise ValueError("calibration/test class counts differ")
    y_cal = _validate_labels(y_calibration, len(p_cal), name="y_calibration")
    if y_cal.max(initial=-1) >= p_cal.shape[1]:
        raise ValueError("calibration label exceeds class count")

    g_cal = np.asarray(groups_calibration, dtype=str)
    g_test = np.asarray(groups_test, dtype=str)
    if g_cal.ndim != 1 or len(g_cal) != len(y_cal):
        raise ValueError("groups_calibration must align with calibration rows")
    if g_test.ndim != 1 or len(g_test) != len(p_out):
        raise ValueError("groups_test must align with test rows")

    names = [str(x) for x in (class_names if class_names is not None else range(p_cal.shape[1]))]
    if len(names) != p_cal.shape[1] or len(set(names)) != len(names):
        raise ValueError("class_names must uniquely identify every class")

    scores_all = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]
    global_sorted = np.sort(np.clip(scores_all, 0.0, 1.0))
    pvalues = np.empty_like(p_out)
    fallback = np.zeros(len(p_out), dtype=bool)

    for group in np.unique(g_test):
        test_idx = np.flatnonzero(g_test == group)
        cal_idx = np.flatnonzero(g_cal == group)
        if len(cal_idx) < int(min_group_size):
            scores = global_sorted
            fallback[test_idx] = True
        else:
            scores = np.sort(np.clip(scores_all[cal_idx], 0.0, 1.0))
        n = len(scores)
        for j in range(p_out.shape[1]):
            thresholds = 1.0 - p_out[test_idx, j]
            idx = np.searchsorted(scores, thresholds, side="left")
            pvalues[test_idx, j] = (1.0 + n - idx) / (n + 1.0)

    prediction_sets, set_sizes = _prediction_set_from_pvalues(pvalues, alpha, names)
    predicted_class = np.argmax(p_out, axis=1)
    predicted_class_pvalue = pvalues[np.arange(len(p_out)), predicted_class]
    actions = np.where(set_sizes == 0, "ABSTAIN", np.where(set_sizes == 1, "SINGLE", "SET"))

    return {
        "method": "group_split_conformal",
        "alpha": alpha,
        "class_names": names,
        "pvalues": pvalues,
        "prediction_sets": prediction_sets,
        "set_size": set_sizes,
        "predicted_class": predicted_class,
        "predicted_class_pvalue": predicted_class_pvalue,
        "action": actions.tolist(),
        "calibration_size": int(len(y_cal)),
        "group_fallback": fallback,
        "fallback_rate": float(fallback.mean()) if len(fallback) else 0.0,
        "finite_sample_correction": True,
    }


def prediction_set_metrics(
    result: Mapping[str, Any],
    y_test: Sequence[int],
) -> dict[str, float]:
    """Evaluate coverage/risk without modifying the precomputed conformal result."""
    y = np.asarray(y_test, dtype=int)
    pred = np.asarray(result["predicted_class"], dtype=int)
    sizes = np.asarray(result["set_size"], dtype=int)
    psets = result["prediction_sets"]
    if y.ndim != 1 or len(y) != len(psets):
        raise ValueError("y_test must align with prediction sets")
    if np.any(y < 0):
        raise ValueError("y_test contains negative labels")

    names = [str(x) for x in result["class_names"]]
    index = {name: i for i, name in enumerate(names)}
    contains = np.asarray(
        [int(y_i) < len(names) and names[int(y_i)] in set(ps) for y_i, ps in zip(y, psets)],
        dtype=bool,
    )
    singleton = sizes == 1
    return {
        "alpha": float(result["alpha"]),
        "n_test": float(len(y)),
        "set_coverage": float(contains.mean()) if len(contains) else float("nan"),
        "mean_set_size": float(sizes.mean()) if len(sizes) else float("nan"),
        "singleton_rate": float(singleton.mean()) if len(singleton) else float("nan"),
        "singleton_accuracy": float((pred[singleton] == y[singleton]).mean()) if singleton.any() else float("nan"),
        "empty_rate": float((sizes == 0).mean()) if len(sizes) else float("nan"),
        "mean_predicted_class_pvalue": float(np.mean(result["predicted_class_pvalue"])) if len(y) else float("nan"),
    }


def decision_from_prediction_set(prediction_set: Sequence[str]) -> dict[str, Any]:
    """Map a prediction set to an operationally explicit output action."""
    names = [str(x) for x in prediction_set]
    if len(names) == 0:
        action = "ABSTAIN"
    elif len(names) == 1:
        action = "SINGLE"
    else:
        action = "SET"
    return {"action": action, "prediction_set": names}


def validate_prediction_set_output(
    prediction_set: Sequence[str] | None,
    *,
    class_names: Sequence[str],
) -> None:
    """Validate a production-bound prediction set without calculating it."""
    if prediction_set is None:
        return
    names = [str(x) for x in prediction_set]
    expected = {str(x) for x in class_names}
    if not names or len(names) != len(set(names)):
        raise ValueError("prediction_set must be a non-empty list of unique class names")
    if not set(names).issubset(expected):
        raise ValueError("prediction_set contains an unknown class")
