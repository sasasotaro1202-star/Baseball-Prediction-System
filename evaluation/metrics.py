"""Evaluation metrics used by the Baseball research layer.

These functions are independent of model training so they can be reused for
chronological OOS folds, candidate validation and locked-holdout evaluation.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error


def multiclass_brier(y_true, proba, classes=None) -> float:
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2:
        raise ValueError("proba must be a 2D array")
    if classes is None:
        classes = list(range(p.shape[1]))
    index = {c: i for i, c in enumerate(classes)}
    y = np.asarray(y_true)
    if any(v not in index for v in y):
        raise ValueError("y_true contains a class absent from classes")
    one_hot = np.zeros_like(p)
    for row, value in enumerate(y):
        one_hot[row, index[value]] = 1.0
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def expected_calibration_error(y_true, proba, classes=None, bins=10) -> float:
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2 or len(p) != len(y_true):
        raise ValueError("proba and y_true have incompatible shapes")
    if classes is None:
        classes = list(range(p.shape[1]))
    pred_idx = np.argmax(p, axis=1)
    pred_labels = np.asarray(classes)[pred_idx]
    confidence = p[np.arange(len(p)), pred_idx]
    correct = (pred_labels == np.asarray(y_true)).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def classification_metrics(y_true, proba, classes=None) -> dict:
    p = np.asarray(proba, dtype=float)
    if classes is None:
        classes = list(range(p.shape[1]))
    pred = np.asarray(classes)[np.argmax(p, axis=1)]
    return {
        "Accuracy": float(accuracy_score(y_true, pred)),
        "LogLoss": float(log_loss(y_true, p, labels=list(classes))),
        "Brier": multiclass_brier(y_true, p, classes),
        "ECE": expected_calibration_error(y_true, p, classes),
        "rows": int(len(y_true)),
    }


def score_metrics(y_home, y_away, pred_home, pred_away) -> dict:
    yh, ya, ph, pa = map(np.asarray, (y_home, y_away, pred_home, pred_away))
    if not (len(yh) == len(ya) == len(ph) == len(pa)):
        raise ValueError("score arrays must have equal length")
    return {
        "HomeRunMAE": float(mean_absolute_error(yh, ph)),
        "AwayRunMAE": float(mean_absolute_error(ya, pa)),
        "ScoreMAE": float((mean_absolute_error(yh, ph) + mean_absolute_error(ya, pa)) / 2.0),
        "rows": int(len(yh)),
    }
