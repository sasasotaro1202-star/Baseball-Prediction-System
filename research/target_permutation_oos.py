"""Leakage diagnostic via deterministic target-permutation OOS retraining.

This module is research-only. It does not modify production models or select
candidates. The strongest form of the audit keeps evaluation features fixed
and retrains the same estimator after shuffling only the training target.

Interpretation:
- SEPARATED: real-target model clearly separates from shuffled-target null.
- NO_SEPARATION: real and shuffled-target models overlap; this is inconclusive
  for leakage and should trigger investigation, not automatic promotion.
- INCONCLUSIVE: insufficient valid permutations or malformed evidence.

A suspiciously strong shuffled-target result can expose target-bearing features.
It is a diagnostic, not a mathematical proof of zero leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np

Array = np.ndarray
FitPredict = Callable[[Array, Array, Array, int], Array]


def _clip_probs(p: Array) -> Array:
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        if len(p) == 0:
            raise ValueError("empty probability vector")
        p = np.column_stack([1.0 - p, p])
    if p.ndim != 2 or len(p) == 0 or p.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty 2-D matrix with >=2 classes")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    sums = p.sum(axis=1, keepdims=True)
    if np.any(sums <= 0) or np.any(np.abs(sums - 1.0) > 1e-5):
        raise ValueError("probability rows must sum to 1")
    return p / sums


def multiclass_metrics(y: Array, p: Array) -> dict[str, float]:
    y = np.asarray(y)
    p = _clip_probs(p)
    if len(y) != len(p):
        raise ValueError("target/probability row mismatch")
    y = y.astype(int, copy=False)
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("target labels are outside probability columns")
    rows = np.arange(len(y))
    ll = -float(np.mean(np.log(np.maximum(p[rows, y], 1e-15))))
    one_hot = np.zeros_like(p)
    one_hot[rows, y] = 1.0
    brier = float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))
    accuracy = float(np.mean(np.argmax(p, axis=1) == y))
    return {"rows": int(len(y)), "LogLoss": ll, "Brier": brier, "Accuracy": accuracy}


def _percentile(values: Sequence[float], q: float) -> float:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        raise ValueError("cannot percentile an empty sequence")
    return float(np.quantile(a, q))


@dataclass(frozen=True)
class TargetPermutationAudit:
    status: str
    risk_flag: bool
    seeds: tuple[int, ...]
    real: dict[str, float]
    null_summary: dict[str, float]
    permutation_p_values: dict[str, float]
    separation: dict[str, float]
    assumptions: tuple[str, ...]


def audit_target_permutation(
    *,
    X_train: Array,
    y_train: Array,
    X_eval: Array,
    y_eval: Array,
    fit_predict: FitPredict,
    seeds: Sequence[int] = (7, 19, 43, 71, 101, 137, 181, 223),
    min_permutations: int = 6,
    logloss_tolerance: float = 0.01,
    accuracy_tolerance: float = 0.02,
) -> TargetPermutationAudit:
    """Retrain on shuffled training targets while keeping all features fixed.

    fit_predict must implement:
        fit_predict(X_train, y_train_variant, X_eval, seed) -> predict_proba

    The same model family and feature matrix must be used for real and every
    shuffled-target run. No evaluation target is touched.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_eval = np.asarray(X_eval)
    y_eval = np.asarray(y_eval)

    if len(X_train) != len(y_train):
        raise ValueError("X_train/y_train mismatch")
    if len(X_eval) != len(y_eval):
        raise ValueError("X_eval/y_eval mismatch")
    if len(X_train) < 10 or len(X_eval) < 10:
        raise ValueError("target permutation audit requires at least 10 train/eval rows")
    if min_permutations < 2:
        raise ValueError("min_permutations must be >= 2")
    unique = np.unique(y_train)
    if unique.size < 2:
        raise ValueError("training target must contain at least two classes")

    seed_list = tuple(int(s) for s in seeds)
    if len(seed_list) < min_permutations:
        raise ValueError("seeds must contain at least min_permutations entries")

    real_p = _clip_probs(fit_predict(X_train, y_train, X_eval, seed_list[0]))
    real = multiclass_metrics(y_eval.astype(int), real_p)

    null_metrics: list[dict[str, float]] = []
    for seed in seed_list:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(y_train)
        p = _clip_probs(fit_predict(X_train, shuffled, X_eval, seed))
        null_metrics.append(multiclass_metrics(y_eval.astype(int), p))

    if len(null_metrics) < min_permutations:
        return TargetPermutationAudit(
            status="INCONCLUSIVE",
            risk_flag=False,
            seeds=seed_list,
            real=real,
            null_summary={},
            permutation_p_values={},
            separation={},
            assumptions=(
                "evaluation features remain fixed across all permutations",
                "only training targets are shuffled",
                "same fit_predict contract is used for real and null runs",
            ),
        )

    null_ll = np.asarray([m["LogLoss"] for m in null_metrics], dtype=float)
    null_brier = np.asarray([m["Brier"] for m in null_metrics], dtype=float)
    null_acc = np.asarray([m["Accuracy"] for m in null_metrics], dtype=float)

    p_ll = float((1 + np.sum(null_ll <= real["LogLoss"])) / (len(null_ll) + 1))
    p_brier = float((1 + np.sum(null_brier <= real["Brier"])) / (len(null_brier) + 1))
    p_acc = float((1 + np.sum(null_acc >= real["Accuracy"])) / (len(null_acc) + 1))

    null_summary = {
        "LogLoss_mean": float(null_ll.mean()),
        "LogLoss_q05": _percentile(null_ll, 0.05),
        "LogLoss_q50": _percentile(null_ll, 0.50),
        "LogLoss_q95": _percentile(null_ll, 0.95),
        "Brier_mean": float(null_brier.mean()),
        "Brier_q05": _percentile(null_brier, 0.05),
        "Brier_q50": _percentile(null_brier, 0.50),
        "Brier_q95": _percentile(null_brier, 0.95),
        "Accuracy_mean": float(null_acc.mean()),
        "Accuracy_q05": _percentile(null_acc, 0.05),
        "Accuracy_q50": _percentile(null_acc, 0.50),
        "Accuracy_q95": _percentile(null_acc, 0.95),
        "permutations": int(len(null_metrics)),
    }
    separation = {
        "LogLoss_null_mean_minus_real": float(null_ll.mean() - real["LogLoss"]),
        "Brier_null_mean_minus_real": float(null_brier.mean() - real["Brier"]),
        "Accuracy_real_minus_null_mean": float(real["Accuracy"] - null_acc.mean()),
    }

    competitive_ll = real["LogLoss"] >= null_ll.mean() - float(logloss_tolerance)
    competitive_acc = real["Accuracy"] <= null_acc.mean() + float(accuracy_tolerance)
    competitive_brier = real["Brier"] >= null_brier.mean() - float(logloss_tolerance)
    risk_flag = bool(sum((competitive_ll, competitive_acc, competitive_brier)) >= 2)

    strong_ll = real["LogLoss"] < _percentile(null_ll, 0.05)
    strong_acc = real["Accuracy"] > _percentile(null_acc, 0.95)
    strong_brier = real["Brier"] < _percentile(null_brier, 0.05)
    strong_signals = int(sum((strong_ll, strong_acc, strong_brier)))

    status = "SUSPICIOUS" if risk_flag else ("SEPARATED" if strong_signals >= 2 else "NO_SEPARATION")

    return TargetPermutationAudit(
        status=status,
        risk_flag=risk_flag,
        seeds=seed_list,
        real=real,
        null_summary=null_summary,
        permutation_p_values={"LogLoss": p_ll, "Brier": p_brier, "Accuracy": p_acc},
        separation=separation,
        assumptions=(
            "evaluation features remain fixed across all permutations",
            "only training targets are shuffled",
            "same fit_predict contract is used for real and null runs",
            "a non-separated result is diagnostic evidence, not proof of leakage",
        ),
    )


def to_dict(report: TargetPermutationAudit) -> dict[str, object]:
    return asdict(report)
