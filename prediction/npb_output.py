"""Production-facing NPB prediction output contract.

This adapter is deliberately model-agnostic: the numerical model supplies
home/draw/away probabilities and score/Low-High fields, while this module
normalizes, validates, and exposes the canonical user-facing contract.
"""
from __future__ import annotations

from typing import Any, Mapping

from evaluation.npb_outcome import NPB_OUTCOME_LABELS, validate_npb_probabilities


def build_npb_prediction(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build one production NPB prediction without inventing missing values."""
    required = ("home", "away", "pred_home", "pred_draw", "pred_away")
    missing = [key for key in required if key not in row or row[key] in (None, "")]
    if missing:
        raise ValueError("NPB production prediction missing: " + ", ".join(missing))

    p = validate_npb_probabilities([
        float(row["pred_home"]),
        float(row["pred_draw"]),
        float(row["pred_away"]),
    ])
    winner_idx = int(max(range(3), key=lambda i: p[i]))
    out = dict(row)
    out.update({
        "outcome_labels": list(NPB_OUTCOME_LABELS),
        "home_win_probability": float(p[0]),
        "draw_probability": float(p[1]),
        "away_win_probability": float(p[2]),
        "predicted_outcome": NPB_OUTCOME_LABELS[winner_idx],
        "prediction_contract": "NPB_HOME_DRAW_AWAY_V1",
    })
    return out


def validate_npb_prediction(row: Mapping[str, Any]) -> None:
    """Fail closed if a production NPB row is not a valid three-way output."""
    built = build_npb_prediction(row)
    p = [built["home_win_probability"], built["draw_probability"], built["away_win_probability"]]
    if abs(sum(p) - 1.0) > 1e-9:
        raise ValueError("NPB production probabilities do not sum to 1")
