"""Canonical cross-game prediction ranking helpers.

This module is intentionally model-agnostic: callers provide already-validated
per-game probabilities.  It guarantees that the highest draw probability among
eligible draw-capable games is surfaced as a separate draw candidate, without
altering the normal winner ranking or fabricating a draw prediction.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def _prob(value: Any, name: str) -> float:
    p = float(value)
    if not math.isfinite(p) or p < 0.0 or p > 1.0:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return p


def rank_games(rows: Iterable[Mapping[str, Any]], *, draw_capable: bool) -> dict[str, Any]:
    """Return normal winner ranking plus the single highest-draw game.

    Each row must contain ``event_id``, ``home_probability`` and
    ``away_probability``. Draw-capable competitions additionally require
    ``draw_probability``. No row is mutated and the draw candidate is selected
    solely by its game-specific draw probability.
    """
    normalized: list[dict[str, Any]] = []
    for row in rows:
        event_id = str(row.get("event_id", "")).strip()
        if not event_id:
            raise ValueError("event_id is required")
        home = _prob(row.get("home_probability"), "home_probability")
        away = _prob(row.get("away_probability"), "away_probability")
        draw = _prob(row.get("draw_probability"), "draw_probability") if draw_capable else None
        total = home + away + (draw or 0.0)
        if abs(total - 1.0) > 1e-8:
            raise ValueError("game probabilities must sum to 1")
        winner = "home" if home >= away else "away"
        winner_probability = max(home, away)
        item = dict(row)
        item.update({
            "home_probability": home,
            "away_probability": away,
            "winner": winner,
            "winner_probability": winner_probability,
        })
        if draw_capable:
            item["draw_probability"] = draw
        normalized.append(item)

    winner_ranking = sorted(
        normalized,
        key=lambda x: (-float(x["winner_probability"]), str(x["event_id"])),
    )
    draw_candidate = None
    if draw_capable and normalized:
        draw_candidate = max(
            normalized,
            key=lambda x: (float(x["draw_probability"]), str(x["event_id"])),
        )
        # Return a detached object so consumers cannot accidentally mutate the
        # canonical normal-ranking row through the special draw slot.
        draw_candidate = dict(draw_candidate)
        draw_candidate["ranking_type"] = "highest_draw_probability"

    return {
        "winner_ranking": winner_ranking,
        "draw_candidate": draw_candidate,
        "draw_capable": bool(draw_capable),
        "game_count": len(normalized),
    }
