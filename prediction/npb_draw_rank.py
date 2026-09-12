"""NPB separate draw-probability prediction contract."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def select_top_draw_prediction(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the single eligible NPB game with the highest draw probability."""
    if not rows:
        raise ValueError("no NPB prediction rows supplied")
    eligible = []
    for row in rows:
        for key in ("home", "away", "draw_probability"):
            if key not in row or row[key] in (None, ""):
                raise ValueError(f"NPB draw ranking requires {key}")
        p = float(row["draw_probability"])
        if not 0.0 <= p <= 1.0:
            raise ValueError("draw_probability must be between 0 and 1")
        eligible.append(row)
    # Stable deterministic tie-break: preserve source order after probability.
    winner = max(enumerate(eligible), key=lambda item: (float(item[1]["draw_probability"]), -item[0]))[1]
    out = dict(winner)
    out.update({
        "prediction_type": "NPB_TOP_DRAW_PROBABILITY_V1",
        "draw_prediction": True,
        "draw_rank": 1,
    })
    return out
