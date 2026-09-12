"""NPB-specific three-outcome prediction contract.

NPB regulation games are evaluated as Home Win / Draw / Away Win.  This
module centralizes the label order and validation so downstream prediction,
evaluation, and reporting code cannot accidentally collapse NPB to binary
win/loss.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

NPB_OUTCOME_LABELS = ("HOME_WIN", "DRAW", "AWAY_WIN")
NPB_OUTCOME_COUNT = 3


def npb_result_label(home_score: float, away_score: float) -> str:
    """Return the canonical NPB three-way outcome label."""
    h = float(home_score)
    a = float(away_score)
    if h > a:
        return NPB_OUTCOME_LABELS[0]
    if h == a:
        return NPB_OUTCOME_LABELS[1]
    return NPB_OUTCOME_LABELS[2]


def validate_npb_probabilities(probabilities: Sequence[float]) -> np.ndarray:
    """Validate and normalize one NPB [home, draw, away] probability vector."""
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (NPB_OUTCOME_COUNT,):
        raise ValueError("NPB probabilities must contain exactly home/draw/away")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("NPB probabilities must be finite and non-negative")
    total = float(p.sum())
    if total <= 0.0:
        raise ValueError("NPB probabilities must have positive total mass")
    return p / total


def add_npb_outcome_columns(row: Mapping[str, object]) -> dict[str, object]:
    """Return a copy with explicit, unambiguous NPB probability field names.

    Existing ``pred_home/pred_draw/pred_away`` fields are preserved for
    backward compatibility. The explicit fields are the canonical reporting
    contract and make it impossible to mistake DRAW for a binary class.
    """
    out = dict(row)
    if "pred_home" not in out or "pred_draw" not in out or "pred_away" not in out:
        raise KeyError("NPB row requires pred_home, pred_draw and pred_away")
    p = validate_npb_probabilities(
        [float(out["pred_home"]), float(out["pred_draw"]), float(out["pred_away"])]
    )
    out["home_win_probability"] = float(p[0])
    out["draw_probability"] = float(p[1])
    out["away_win_probability"] = float(p[2])
    out["outcome_labels"] = list(NPB_OUTCOME_LABELS)
    return out


def score_matrix_to_outcomes(score_matrix: np.ndarray) -> np.ndarray:
    """Convert a home/away score probability matrix to NPB outcomes.

    The matrix must use row=home runs and column=away runs.  The returned
    vector is [home win, draw, away win] and sums to one after normalization.
    """
    m = np.asarray(score_matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("score matrix must be a square 2-D array")
    if not np.all(np.isfinite(m)) or np.any(m < 0.0):
        raise ValueError("score matrix must be finite and non-negative")
    home = float(np.tril(m, -1).sum())
    draw = float(np.trace(m))
    away = float(np.triu(m, 1).sum())
    return validate_npb_probabilities([home, draw, away])
