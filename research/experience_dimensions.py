"""Stable, outcome-free dimensions for experience performance reporting.

Grouping uses prediction-time metadata only. Unknown/ambiguous competition or phase
is preserved as UNKNOWN and never guessed into a regular-season bucket.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from research.competition_taxonomy import classify_game

UNKNOWN = "UNKNOWN"


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _label_from_row(row: dict[str, Any]) -> dict[str, str]:
    league = _text(row.get("league")).upper()
    if league in {"", UNKNOWN, "NAN", "NONE"}:
        candidate = _text(row.get("competition_id")).upper()
        if candidate in {"NPB", "MLB"}:
            league = candidate
    if not league:
        target = _text(row.get("target")).upper()
        if target in {"NPB", "MLB"}:
            league = target
    if not league:
        league = UNKNOWN

    # Prefer explicitly stored taxonomy metadata from prediction time.
    explicit = {
        key: _text(row.get(key))
        for key in (
            "competition_key",
            "competition",
            "competition_stage",
            "season_type",
            "game_class",
            "competition_classification_status",
        )
    }
    if explicit["competition_key"] and explicit["competition"]:
        return {
            "league": league,
            "competition": explicit["competition"],
            "competition_stage": explicit["competition_stage"] or UNKNOWN,
            "season_type": explicit["season_type"] or UNKNOWN,
            "game_class": explicit["game_class"] or UNKNOWN,
            "competition_key": explicit["competition_key"],
            "competition_classification_status": (
                explicit["competition_classification_status"] or "classified"
            ),
        }

    # Otherwise classify only from prediction-time competition metadata.
    label = classify_game(
        league,
        game_type=row.get("game_type", ""),
        series_description=row.get("series_description", ""),
    )
    return {
        "league": league,
        "competition": label.competition.upper() if label.competition else UNKNOWN,
        "competition_stage": label.stage.upper() if label.stage else UNKNOWN,
        "season_type": label.season_type.upper() if label.season_type else UNKNOWN,
        "game_class": label.game_class.upper() if label.game_class else UNKNOWN,
        "competition_key": (
            f"{league}:{UNKNOWN}:{UNKNOWN}"
            if label.competition_key == "baseball:unknown"
            else (label.competition_key if label.competition_key else f"{league}:{UNKNOWN}:{UNKNOWN}")
        ),
        "competition_classification_status": label.status.upper() if label.status else "UNKNOWN",
    }


def add_dimensions(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach league/competition/phase dimensions without altering outcome values."""
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    labels = [_label_from_row(row) for row in out.to_dict("records")]
    for key in (
        "league",
        "competition",
        "competition_stage",
        "season_type",
        "game_class",
        "competition_key",
        "competition_classification_status",
    ):
        out[key] = [label[key] for label in labels]
    return out
