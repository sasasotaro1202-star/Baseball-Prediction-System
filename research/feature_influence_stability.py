"""Chronological feature influence stability diagnostics.

Permutation importance is computed separately inside each validation window
using only that window's unseen targets. Feature ranking stability is reported
across windows; no feature is promoted or removed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np

Array = np.ndarray
Predict = Callable[[object, Array], Array]
Fit = Callable[[Array, Array, int], object]


def _clip_probs(p: Array) -> Array:
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = np.column_stack([1.0 - p, p])
    if p.ndim != 2 or len(p) == 0 or p.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty 2-D matrix with >=2 classes")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    sums = p.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("probability rows must have positive sums")
    return p / sums


def negative_logloss(y: Array, p: Array) -> float:
    p = _clip_probs(p)
    y = np.asarray(y, dtype=int)
    if len(y) != len(p):
        raise ValueError("target/probability row mismatch")
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("target class outside probability columns")
    idx = np.arange(len(y))
    return -float(np.mean(np.log(np.maximum(p[idx, y], 1e-15))))


def _rankdata(values: Array) -> Array:
    order = np.argsort(-np.asarray(values, dtype=float), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1, dtype=float)
    return ranks


def _spearman(a: Array, b: Array) -> float:
    a = _rankdata(a)
    b = _rankdata(b)
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@dataclass(frozen=True)
class WindowImportance:
    window: str
    rows: int
    baseline_logloss: float
    mean_importance: dict[str, float]
    positive_feature_count: int


@dataclass(frozen=True)
class FeatureInfluenceStability:
    status: str
    windows: tuple[WindowImportance, ...]
    pairwise_rank_spearman: dict[str, float]
    mean_rank_spearman: float
    top_feature_frequency: dict[str, float]
    assumptions: tuple[str, ...]


def permutation_importance_by_window(
    *,
    windows: Sequence[tuple[str, Array, Array, Array, Array]],
    feature_names: Sequence[str],
    fit_model: Fit,
    predict_proba: Predict,
    repeats: int = 5,
    seed: int = 42,
) -> FeatureInfluenceStability:
    """Measure permutation importance across chronological windows.

    Each item in windows is:
      (window_name, X_train, y_train, X_eval, y_eval)

    The model is refit from scratch per window. During importance evaluation,
    only X_eval columns are permuted; y_eval remains fixed and unseen by fit.
    """
    names = tuple(str(x) for x in feature_names)
    if not names:
        raise ValueError("feature_names is empty")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    if repeats < 2:
        raise ValueError("repeats must be >= 2")
    if not windows:
        raise ValueError("at least one chronological window is required")

    reports: list[WindowImportance] = []
    rng = np.random.default_rng(seed)

    for window_name, X_train, y_train, X_eval, y_eval in windows:
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train, dtype=int)
        X_eval = np.asarray(X_eval)
        y_eval = np.asarray(y_eval, dtype=int)
        if X_train.ndim != 2 or X_eval.ndim != 2:
            raise ValueError("X arrays must be 2-D")
        if X_train.shape[1] != len(names) or X_eval.shape[1] != len(names):
            raise ValueError("feature count mismatch")
        if len(X_train) != len(y_train) or len(X_eval) != len(y_eval):
            raise ValueError("feature/target row mismatch")
        if len(X_train) < 20 or len(X_eval) < 20:
            raise ValueError("each validation window requires >=20 rows")

        model = fit_model(X_train, y_train, int(rng.integers(0, 2**31 - 1)))
        base_p = _clip_probs(predict_proba(model, X_eval))
        baseline = negative_logloss(y_eval, base_p)

        importance = np.zeros(len(names), dtype=float)
        for j in range(len(names)):
            losses: list[float] = []
            for _ in range(repeats):
                Xp = X_eval.copy()
                perm = rng.permutation(len(Xp))
                Xp[:, j] = Xp[perm, j]
                pp = _clip_probs(predict_proba(model, Xp))
                losses.append(negative_logloss(y_eval, pp))
            importance[j] = float(np.mean(losses) - baseline)

        reports.append(
            WindowImportance(
                window=str(window_name),
                rows=int(len(y_eval)),
                baseline_logloss=baseline,
                mean_importance={n: float(v) for n, v in zip(names, importance)},
                positive_feature_count=int(np.sum(importance > 0)),
            )
        )

    matrices = [np.asarray([r.mean_importance[n] for n in names], dtype=float) for r in reports]
    pairwise: dict[str, float] = {}
    for i in range(len(reports)):
        for j in range(i + 1, len(reports)):
            pairwise[f"{reports[i].window}__vs__{reports[j].window}"] = _spearman(matrices[i], matrices[j])
    mean_spearman = float(np.mean(list(pairwise.values()))) if pairwise else 1.0

    top_counts = {n: 0 for n in names}
    for matrix in matrices:
        top_counts[names[int(np.argmax(matrix))]] += 1
    top_frequency = {n: float(c / len(matrices)) for n, c in top_counts.items() if c}

    status = "STABLE" if len(reports) == 1 or mean_spearman >= 0.50 else "UNSTABLE"
    return FeatureInfluenceStability(
        status=status,
        windows=tuple(reports),
        pairwise_rank_spearman=pairwise,
        mean_rank_spearman=mean_spearman,
        top_feature_frequency=top_frequency,
        assumptions=(
            "models are refit independently inside each chronological window",
            "feature permutation is applied only to the unseen evaluation window",
            "evaluation targets are never used for model fitting",
            "importance is diagnostic and not an adoption decision",
        ),
    )


def to_dict(report: FeatureInfluenceStability) -> dict[str, object]:
    return asdict(report)
