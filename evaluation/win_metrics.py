"""Baseball win prediction evaluation metrics."""

from math import log


def accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def brier_score(y_true, probabilities):
    if not y_true:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probabilities, y_true)) / len(y_true)


def log_loss(y_true, probabilities, eps=1e-15):
    if not y_true:
        return 0.0
    total = 0
    for y, p in zip(y_true, probabilities):
        p = max(min(p, 1 - eps), eps)
        total += y * log(p) + (1-y) * log(1-p)
    return -total / len(y_true)
