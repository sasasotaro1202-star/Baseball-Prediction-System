#!/usr/bin/env python3
"""Research-only adapter for MLB observations returned by the StatsHawk connector.

This module deliberately does not call StatsHawk itself. The connected
StatsHawk tool is the acquisition layer; this adapter validates and normalizes
its returned JSON before any repository-side research code can consume it.

Safety contract:
- StatsHawk is secondary/research validation, never canonical MLB identity/result.
- No prediction-time availability is inferred from retrieval time or kickoff.
- Probable pitchers are never treated as proof of an official historical
  starter announcement.
- Pregame use fails closed unless explicit announcement_at evidence exists.
- Postgame result observations may be used for secondary reconciliation only
  when status is final and scores are present.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, Iterable


FINAL_STATUSES = {"final", "game over", "completed", "complete"}
NON_FINAL_STATUSES = {"scheduled", "in progress", "postponed", "cancelled", "canceled", "suspended"}


def _parse_json(payload: Any) -> dict[str, Any]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("payload_must_be_object")
    return payload


def _iso(value: Any, *, field: str) -> str:
    if value in (None, ""):
        raise ValueError(f"{field}_missing")
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}_invalid") from exc
    if ts.tzinfo is None:
        raise ValueError(f"{field}_must_be_timezone_aware")
    return ts.isoformat()


def _finite_number(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}_invalid") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field}_nonfinite")
    return number


def _games(payload: dict[str, Any]) -> list[dict[str, Any]]:
    games = payload.get("games")
    if not isinstance(games, list):
        raise ValueError("games_must_be_list")
    return [g for g in games if isinstance(g, dict)]


def normalize_matchups(payload: Any, *, retrieved_at: str) -> list[dict[str, Any]]:
    """Normalize get_mlb_matchups/search_games-shaped observations.

    retrieved_at is collector metadata only. It is intentionally not copied
    into available_at, because retrieval is not proof of public availability
    at prediction time.
    """
    raw = _parse_json(payload)
    retrieved = _iso(retrieved_at, field="retrieved_at")
    rows: list[dict[str, Any]] = []

    for index, game in enumerate(_games(raw)):
        event_id = game.get("contest") or game.get("game_id")
        if not event_id:
            raise ValueError(f"games[{index}].event_id_missing")

        status = str(game.get("status", "")).strip().lower()
        if not status:
            raise ValueError(f"games[{index}].status_missing")

        scheduled_at = game.get("scheduled_at") or game.get("kickoff")
        scheduled = _iso(scheduled_at, field=f"games[{index}].scheduled_at")

        home = game.get("home") if isinstance(game.get("home"), dict) else {}
        away = game.get("away") if isinstance(game.get("away"), dict) else {}

        home_team = home.get("team") or game.get("home_team")
        away_team = away.get("team") or game.get("away_team")
        home_team_id = home.get("team_id") or game.get("home_team_id")
        away_team_id = away.get("team_id") or game.get("away_team_id")
        if not home_team_id or not away_team_id:
            raise ValueError(f"games[{index}].team_identity_missing")

        row: dict[str, Any] = {
            "source": "statshawk",
            "source_role": "secondary_validation",
            "observation_index": index,
            "event_id": str(event_id),
            "status": status,
            "scheduled_at": scheduled,
            "retrieved_at": retrieved,
            "available_at": None,
            "availability_proof": "UNVERIFIED",
            "home_team_id": str(home_team_id),
            "home_team": str(home_team) if home_team else None,
            "away_team_id": str(away_team_id),
            "away_team": str(away_team) if away_team else None,
        }

        for side, obj in (("home", home), ("away", away)):
            probable = obj.get("probable_pitcher")
            if isinstance(probable, dict) and probable.get("person_id") and probable.get("name"):
                row[f"{side}_probable_pitcher_id"] = str(probable["person_id"])
                row[f"{side}_probable_pitcher"] = str(probable["name"])
                row[f"{side}_probable_pitcher_throws"] = probable.get("throws")
            else:
                row[f"{side}_probable_pitcher_id"] = None
                row[f"{side}_probable_pitcher"] = None
                row[f"{side}_probable_pitcher_throws"] = None

            lineup = obj.get("lineup")
            if lineup is not None and not isinstance(lineup, list):
                raise ValueError(f"games[{index}].{side}.lineup_invalid")
            row[f"{side}_lineup_count"] = len(lineup or [])
            row[f"{side}_lineup_confirmed_count"] = sum(
                1 for player in (lineup or [])
                if isinstance(player, dict) and player.get("confirmed") is True
            )

        home_score = home.get("score") if home else game.get("home_score")
        away_score = away.get("score") if away else game.get("away_score")
        if home_score is not None or away_score is not None:
            row["home_score"] = _finite_number(home_score, field=f"games[{index}].home_score")
            row["away_score"] = _finite_number(away_score, field=f"games[{index}].away_score")
        else:
            row["home_score"] = None
            row["away_score"] = None

        row["result_reconciliation_eligible"] = (
            status in FINAL_STATUSES
            and row["home_score"] is not None
            and row["away_score"] is not None
        )

        rows.append(row)

    return rows


def evaluate_pregame_eligibility(row: dict[str, Any], prediction_cutoff: str) -> tuple[bool, str]:
    """Return strict PIT eligibility for pregame use of a StatsHawk observation.

    This intentionally rejects normal StatsHawk matchup output because it
    exposes probable pitchers but no explicit historical announcement_at.
    """
    cutoff = datetime.fromisoformat(_iso(prediction_cutoff, field="prediction_cutoff"))

    try:
        start = datetime.fromisoformat(_iso(row.get("scheduled_at"), field="scheduled_at"))
    except ValueError:
        return False, "scheduled_at_invalid"

    if start <= cutoff:
        return False, "game_not_future_at_cutoff"

    for side in ("home", "away"):
        announcement = row.get(f"{side}_starter_announced_at")
        starter = row.get(f"{side}_starter")
        if not starter or not announcement:
            return False, f"{side}_starter_announcement_unproven"
        try:
            announced = datetime.fromisoformat(
                _iso(announcement, field=f"{side}_starter_announced_at")
            )
        except ValueError:
            return False, f"{side}_starter_announcement_invalid"
        if announced > cutoff:
            return False, f"{side}_starter_announced_after_cutoff"

    return True, "eligible"


def validate_reconciliation_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate finalized secondary observations without changing canonical IDs."""
    validated: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("row_must_be_object")
        if row.get("source") != "statshawk":
            raise ValueError("unexpected_source")
        if not row.get("event_id"):
            raise ValueError("event_id_missing")
        status = str(row.get("status", "")).lower()
        if status in FINAL_STATUSES:
            if row.get("home_score") is None or row.get("away_score") is None:
                raise ValueError("final_score_missing")
            validated.append({**row, "reconciliation_role": "secondary_result_check"})
        else:
            validated.append({**row, "reconciliation_role": "observation_only"})
    return validated
