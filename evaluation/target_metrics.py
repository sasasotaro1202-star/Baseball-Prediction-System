"""Baseball multi-target evaluation helpers.

Keeps win probability, score candidates, and Low/High as separate targets.
No target is allowed to silently substitute for another target.
"""
from __future__ import annotations

from typing import Iterable, Sequence
import math
import numpy as np


def _binary_metrics(y_true: Iterable[int], p_yes: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=float)
    p = np.clip(np.asarray(list(p_yes), dtype=float), 1e-15, 1 - 1e-15)
    if len(y) != len(p):
        raise ValueError("y_true and p_yes must have equal length")
    pred = (p >= 0.5).astype(float)
    return {
        "Accuracy": float(np.mean(pred == y)),
        "LogLoss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "Brier": float(np.mean((p - y) ** 2)),
        "rows": int(len(y)),
    }


def hilo_metrics(y_high: Iterable[int], p_high: Iterable[float]) -> dict[str, float]:
    """Evaluate the Low/High market independently; 1 means High."""
    return _binary_metrics(y_high, p_high)


def score_top4_metrics(
    y_home: Iterable[int],
    y_away: Iterable[int],
    candidates: Sequence[Sequence[Sequence[int]]],
    probabilities: Sequence[Sequence[float]] | None = None,
) -> dict[str, float]:
    """Evaluate four score candidates per game.

    ``candidates[i]`` must contain four ``(home_runs, away_runs)`` pairs.
    ``probabilities`` is optional; when supplied, top-1 probability calibration
    is reported as a simple mean absolute calibration error against top-1 hit.
    """
    yh = list(y_home)
    ya = list(y_away)
    if len(yh) != len(ya) or len(yh) != len(candidates):
        raise ValueError("score inputs must have equal length")
    top1 = 0
    top4 = 0
    for h, a, choices in zip(yh, ya, candidates):
        if len(choices) != 4:
            raise ValueError("each game must have exactly four score candidates")
        pairs = {(int(x[0]), int(x[1])) for x in choices}
        top1 += int((int(choices[0][0]), int(choices[0][1])) == (int(h), int(a)))
        top4 += int((int(h), int(a)) in pairs)
    out = {
        "Top1HitRate": float(top1 / len(yh)) if yh else math.nan,
        "Top4HitRate": float(top4 / len(yh)) if yh else math.nan,
        "rows": int(len(yh)),
    }
    if probabilities is not None:
        if len(probabilities) != len(yh):
            raise ValueError("probabilities length mismatch")
        p = np.asarray([float(row[0]) for row in probabilities], dtype=float)
        hit = np.asarray([
            int((int(c[0][0]), int(c[0][1])) == (int(h), int(a)))
            for h, a, c in zip(yh, ya, candidates)
        ], dtype=float)
        if len(p):
            out["Top1ProbabilityMAE"] = float(np.mean(np.abs(p - hit)))
    return out


def score_distribution_metrics(
    y_home: Iterable[int],
    y_away: Iterable[int],
    expected_home: Iterable[float],
    expected_away: Iterable[float],
) -> dict[str, float]:
    """Continuous score-distribution diagnostics using MAE/RMSE."""
    yh, ya = np.asarray(list(y_home), dtype=float), np.asarray(list(y_away), dtype=float)
    ph, pa = np.asarray(list(expected_home), dtype=float), np.asarray(list(expected_away), dtype=float)
    if not (len(yh) == len(ya) == len(ph) == len(pa)):
        raise ValueError("score arrays must have equal length")
    if not len(yh):
        return {"HomeRunMAE": math.nan, "AwayRunMAE": math.nan, "ScoreMAE": math.nan, "ScoreRMSE": math.nan, "rows": 0}
    errors = np.concatenate([ph - yh, pa - ya])
    return {
        "HomeRunMAE": float(np.mean(np.abs(ph - yh))),
        "AwayRunMAE": float(np.mean(np.abs(pa - ya))),
        "ScoreMAE": float((np.mean(np.abs(ph - yh)) + np.mean(np.abs(pa - ya))) / 2),
        "ScoreRMSE": float(np.sqrt(np.mean(errors ** 2))),
        "rows": int(len(yh)),
    }


def multi_target_summary(
    *,
    win: dict[str, float],
    score: dict[str, float],
    hilo: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Return an explicit target-by-target evaluation record for research logs."""
    return {"win": dict(win), "score": dict(score), "low_high": dict(hilo)}
