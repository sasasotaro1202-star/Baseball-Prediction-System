#!/usr/bin/env python3
"""Detect a one-time 60-90 minute NPB pregame research-shadow slot.

This lane exists to obtain a PIT-safe research forecast earlier than the
canonical 60-minute automatic slot. It never changes production eligibility.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import prediction.pregame_scheduler as scheduler

JST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[1]


def _archived_game_ids(target_date: str, prediction_source: str) -> set[str]:
    path = (
        ROOT
        / "data"
        / "experience"
        / "research_shadow"
        / "predictions"
        / f"{target_date}.jsonl"
    )
    if not path.exists():
        return set()
    result: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("prediction_source") or "") != prediction_source:
            continue
        game_id = str(row.get("game_id") or "").strip()
        if game_id:
            result.add(game_id)
    return result


def rapid_due_games(
    *,
    now_utc: datetime | None = None,
    min_lead_minutes: float = 60.0,
    max_lead_minutes: float = 90.0,
    prediction_source: str = "RESEARCH_SHADOW_AUTO_90M",
) -> dict[str, Any]:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    target_date = now.astimezone(JST).date().isoformat()

    if min_lead_minutes < 0:
        raise ValueError("min_lead_minutes must be non-negative")
    if max_lead_minutes <= min_lead_minutes:
        raise ValueError("max_lead_minutes must be greater than min_lead_minutes")

    games = scheduler._schedule_for_date(target_date)
    archived = _archived_game_ids(target_date, prediction_source)
    due: list[dict[str, Any]] = []

    for game_index, game in enumerate(games, start=1):
        start = datetime.fromisoformat(
            f"{target_date}T{game['official_start_time']}:00+09:00"
        ).astimezone(timezone.utc)
        lead = (start - now).total_seconds() / 60.0
        game_id = f"NPB-{target_date}-{game_index}"

        if (
            min_lead_minutes < lead <= max_lead_minutes
            and lead > 0
            and game_id not in archived
        ):
            preferred_cutoff = start - timedelta(minutes=max_lead_minutes)
            due.append(
                {
                    "league": "NPB",
                    "target_date": target_date,
                    "game_id": game_id,
                    "game_index": game_index,
                    "home": game["home"],
                    "away": game["away"],
                    "official_start_time": game["official_start_time"],
                    "prediction_cutoff_utc": now.isoformat(),
                    "preferred_prediction_cutoff_utc": preferred_cutoff.isoformat(),
                    "preferred_target_lead_minutes": float(max_lead_minutes),
                    "lead_minutes": round(lead, 3),
                    "prediction_source": prediction_source,
                    "prediction_eligibility": "RESEARCH_SHADOW_PIT_SAFE_STARTERS_REQUIRED",
                    "status": "RESEARCH_RAPID_DUE",
                }
            )

    return {
        "schema_version": "npb-rapid-pregame-v1",
        "checked_at_utc": now.isoformat(),
        "target_date": target_date,
        "min_lead_minutes": float(min_lead_minutes),
        "max_lead_minutes": float(max_lead_minutes),
        "prediction_source": prediction_source,
        "due_games": due,
        "due_dates": sorted({row["target_date"] for row in due}),
        "status": "DUE" if due else "NO_DUE_GAMES",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-lead-minutes", type=float, default=60.0)
    parser.add_argument("--max-lead-minutes", type=float, default=90.0)
    parser.add_argument(
        "--prediction-source",
        default="RESEARCH_SHADOW_AUTO_90M",
    )
    args = parser.parse_args()

    result = rapid_due_games(
        min_lead_minutes=args.min_lead_minutes,
        max_lead_minutes=args.max_lead_minutes,
        prediction_source=args.prediction_source,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
