"""Research-only retrieval-local conformal prediction sets (v2).

v2 hardens the retrieval candidate around the baseball system's immutable
game-level experience contract:

* game_id is required, so multiple pregame revisions cannot be silently treated
  as independent calibration cases;
* the latest mature pre-cutoff snapshot for each prior game is selected;
* the target game's own historical revisions are excluded explicitly;
* nearest-neighbour ordering is fully deterministic, including exact-distance
  ties;
* all standardisation statistics are fit from temporally eligible prior games
  only.

This module is not wired into production and makes no coverage guarantee under
temporal dependence or localized retrieval. It is a research candidate for
chronological OOS evaluation.
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


def _probabilities(values: Any, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] < 2:
        raise ValueError("probabilities must be a 2-D matrix aligned with cases")
    if not np.isfinite(arr).all() or (arr < 0.0).any():
        raise ValueError("probabilities must be finite and non-negative")
    total = arr.sum(axis=1, keepdims=True)
    if np.any(total <= EPS):
        raise ValueError("probability rows must have positive mass")
    return arr / total


def _labels(values: Sequence[Any], n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=int).reshape(-1)
    if len(arr) != n or np.any(arr < 0):
        raise ValueError("labels must be non-negative and aligned")
    return arr


def _features(values: Any, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] < 1:
        raise ValueError("features must be a 2-D matrix aligned with cases")
    if not np.isfinite(arr).all():
        raise ValueError("features must be finite; imputation is not implicit")
    return arr


def _game_ids(values: Sequence[Any], n: int) -> np.ndarray:
    games = np.asarray(values, dtype=object).reshape(-1)
    if len(games) != n:
        raise ValueError("game_ids must align with cases")
    normalized = np.asarray([str(v).strip() for v in games], dtype=object)
    if np.any(normalized == "") or np.isin(normalized, ["nan", "None"]).any():
        raise ValueError("game_ids must be explicit non-empty identifiers")
    return normalized


def _class_names(values: Sequence[str] | None, n_classes: int) -> list[str]:
    names = [str(v) for v in (values if values is not None else range(n_classes))]
    if len(names) != n_classes or len(set(names)) != n_classes:
        raise ValueError("class_names must uniquely identify every class")
    return names


def _pvalues(scores: np.ndarray, p_row: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.clip(np.asarray(scores, dtype=float).reshape(-1), 0.0, 1.0))
    threshold = 1.0 - np.asarray(p_row, dtype=float)
    idx = np.searchsorted(ordered, threshold, side="left")
    n = len(ordered)
    return (1.0 + n - idx) / (n + 1.0)


def _latest_mature_snapshot_indices(
    eligible: np.ndarray,
    games: np.ndarray,
    prediction_times: pd.Series,
) -> np.ndarray:
    latest: dict[str, int] = {}
    for idx in eligible.tolist():
        key = str(games[idx])
        prior = latest.get(key)
        if prior is None:
            latest[key] = int(idx)
            continue
        current_key = (prediction_times.iloc[idx], int(idx))
        prior_key = (prediction_times.iloc[prior], int(prior))
        if current_key > prior_key:
            latest[key] = int(idx)

    selected = sorted(
        latest.values(),
        key=lambda idx: (prediction_times.iloc[idx], int(idx)),
    )
    return np.asarray(selected, dtype=int)


def retrieval_local_conformal_sets_v2(
    y: Sequence[Any],
    probabilities: Any,
    features: Any,
    prediction_times: Sequence[Any],
    outcome_confirmed_at: Sequence[Any],
    game_ids: Sequence[Any],
    *,
    alpha: float = 0.10,
    min_calibration_games: int = 30,
    max_pool_games: int = 200,
    k_neighbors: int = 30,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create one retrieval-local conformal set per chronological game snapshot.

    The target row i can use only one latest mature snapshot from each prior
    game j satisfying:
      prediction_time[j] < prediction_time[i]
      outcome_confirmed_at[j] <= prediction_time[i]

    The target game's own game_id is always excluded from the retrieval pool.
    """
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    if int(min_calibration_games) < 2:
        raise ValueError("min_calibration_games must be >= 2")
    if int(max_pool_games) < int(min_calibration_games):
        raise ValueError("max_pool_games must be >= min_calibration_games")
    if int(k_neighbors) < int(min_calibration_games):
        raise ValueError("k_neighbors must be >= min_calibration_games")
    if int(k_neighbors) > int(max_pool_games):
        raise ValueError("k_neighbors must be <= max_pool_games")

    pt = _timestamps(prediction_times, name="prediction_times")
    mature = _timestamps(outcome_confirmed_at, name="outcome_confirmed_at")
    n = len(pt)
    if len(mature) != n:
        raise ValueError("outcome_confirmed_at must align with prediction_times")
    if not pt.is_monotonic_increasing:
        raise ValueError("prediction_times must be monotonically non-decreasing")
    if (mature < pt).any():
        raise ValueError("outcome_confirmed_at cannot precede prediction_time")

    p = _probabilities(probabilities, n)
    labels = _labels(y, n)
    if labels.max(initial=-1) >= p.shape[1]:
        raise ValueError("label exceeds probability class count")
    X = _features(features, n)
    games = _game_ids(game_ids, n)
    names = _class_names(class_names, p.shape[1])

    sets: list[list[str]] = []
    actions: list[str] = []
    set_sizes = np.zeros(n, dtype=int)
    pool_counts = np.zeros(n, dtype=int)
    calibration_counts = np.zeros(n, dtype=int)
    nearest_distance = np.full(n, np.nan, dtype=float)
    pvalues = np.full_like(p, np.nan, dtype=float)

    for i in range(n):
        prior = np.arange(i, dtype=int)
        eligible = prior[
            (pt.iloc[prior] < pt.iloc[i]).to_numpy()
            & (mature.iloc[prior] <= pt.iloc[i]).to_numpy()
            & (games[prior] != games[i])
        ]

        eligible = _latest_mature_snapshot_indices(eligible, games, pt)
        if len(eligible) > int(max_pool_games):
            eligible = eligible[-int(max_pool_games):]
        pool_counts[i] = int(len(eligible))

        if len(eligible) < int(min_calibration_games):
            sets.append([])
            actions.append("ABSTAIN")
            continue

        pool_x = X[eligible]
        mean = pool_x.mean(axis=0)
        scale = pool_x.std(axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
        z_pool = (pool_x - mean) / scale
        z_target = (X[i] - mean) / scale
        distances = np.sum((z_pool - z_target[None, :]) ** 2, axis=1)

        order = sorted(
            range(len(eligible)),
            key=lambda pos: (
                float(distances[pos]),
                int(pt.iloc[int(eligible[pos])].value),
                int(eligible[pos]),
            ),
        )
        selected_pos = np.asarray(order[: int(k_neighbors)], dtype=int)
        selected = eligible[selected_pos]

        calibration_counts[i] = int(len(selected))
        nearest_distance[i] = float(np.sqrt(distances[selected_pos[0]]))

        scores = 1.0 - p[selected, labels[selected]]
        pv = _pvalues(scores, p[i])
        pvalues[i] = pv

        included = pv > alpha
        current_set = [names[j] for j in np.flatnonzero(included)]
        size = int(included.sum())
        sets.append(current_set)
        set_sizes[i] = size
        actions.append("ABSTAIN" if size == 0 else ("SINGLE" if size == 1 else "SET"))

    eligible_rows = calibration_counts >= int(min_calibration_games)
    return {
        "method": "retrieval_local_conformal_v2",
        "alpha": alpha,
        "class_names": names,
        "prediction_sets": sets,
        "action": actions,
        "set_size": set_sizes,
        "predicted_class": np.argmax(p, axis=1),
        "pvalues": pvalues,
        "retrieval_pool_count": pool_counts,
        "retrieved_calibration_count": calibration_counts,
        "calibration_count": calibration_counts,
        "nearest_distance": nearest_distance,
        "eligible_rows": int(eligible_rows.sum()),
        "total_rows": int(n),
        "min_calibration_games": int(min_calibration_games),
        "max_pool_games": int(max_pool_games),
        "k_neighbors": int(k_neighbors),
        "contract": {
            "prior_prediction_time_strictly_earlier": True,
            "outcome_confirmed_at_le_target_prediction_time": True,
            "same_prediction_time_excluded": True,
            "current_outcome_excluded": True,
            "game_id_required": True,
            "same_game_revisions_collapsed": True,
            "target_game_excluded": True,
            "feature_scaling_uses_only_temporally_eligible_prior_games": True,
            "bounded_retrieval_pool_in_games": True,
            "bounded_calibration_in_games": True,
            "deterministic_tie_break": True,
            "production_changed": False,
            "promotion_allowed": False,
        },
        "limitations": {
            "formal_coverage_under_temporal_dependence": "not_claimed",
            "localized_retrieval_requires_fresh_chronological_oos": True,
            "set_efficiency_requires_oos_validation": True,
        },
    }


def retrieval_conformal_metrics_v2(
    result: dict[str, Any],
    y: Sequence[Any],
) -> dict[str, float]:
    labels = np.asarray(y, dtype=int).reshape(-1)
    sets = result["prediction_sets"]
    sizes = np.asarray(result["set_size"], dtype=int)
    counts = np.asarray(result["calibration_count"], dtype=int)
    min_games = int(result.get("min_calibration_games", 1))
    if len(labels) != len(sets) or len(sizes) != len(labels):
        raise ValueError("result/y length mismatch")
    if np.any(labels < 0):
        raise ValueError("y contains negative labels")
    names = [str(v) for v in result["class_names"]]
    eligible = counts >= min_games
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


__all__ = ["retrieval_local_conformal_sets_v2", "retrieval_conformal_metrics_v2"]
