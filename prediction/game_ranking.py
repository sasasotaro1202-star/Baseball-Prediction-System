"""Canonical cross-game prediction ranking helpers."""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def _prob(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    try:
        p = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(p) or p < 0.0 or p > 1.0:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return p


def _confidence(home: float, draw: float | None, away: float) -> tuple[float, str]:
    """Return a conservative game-confidence score and status.

    Confidence is based on the probability margin between the best and second
    best outcome. It is deliberately descriptive rather than a promise of
    correctness. Thresholds are explicit so downstream ranking cannot silently
    manufacture certainty.
    """
    probs = [home, away] if draw is None else [home, draw, away]
    ordered = sorted(probs, reverse=True)
    margin = ordered[0] - ordered[1]
    if ordered[0] >= 0.65 and margin >= 0.15:
        status = "HIGH"
    elif ordered[0] >= 0.55 and margin >= 0.08:
        status = "MEDIUM"
    else:
        status = "LOW"
    return margin, status


def rank_games(rows: Iterable[Mapping[str, Any]], *, draw_capable: bool) -> dict[str, Any]:
    """Return winner ranking, confidence/uncertainty buckets and Top-Draw.

    For draw-capable competitions, ``draw_candidate`` is the single eligible
    game with the highest game-specific draw probability. Its percentage is
    exposed as ``draw_probability_pct``. This selection never changes the
    normal winner prediction.
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
        margin, confidence_status = _confidence(home, draw, away)
        item = dict(row)
        item.update({
            "home_probability": home,
            "away_probability": away,
            "winner": winner,
            "winner_probability": winner_probability,
            "confidence_margin": margin,
            "confidence_status": confidence_status,
        })
        if draw_capable:
            item["draw_probability"] = draw
            item["draw_probability_pct"] = round(draw * 100.0, 2)
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
        draw_candidate = dict(draw_candidate)
        draw_candidate["ranking_type"] = "highest_draw_probability"
        draw_candidate["draw_probability_pct"] = round(float(draw_candidate["draw_probability"]) * 100.0, 2)

    low_confidence_games = [x for x in normalized if x["confidence_status"] == "LOW"]
    return {
        "winner_ranking": winner_ranking,
        "draw_candidate": draw_candidate,
        "draw_capable": bool(draw_capable),
        "game_count": len(normalized),
        "low_confidence_games": low_confidence_games,
        "low_confidence_count": len(low_confidence_games),
    }
