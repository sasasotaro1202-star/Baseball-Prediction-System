#!/usr/bin/env python3
"""Fail-closed production gate for PIT starter eligibility.

This gate is intentionally independent of model accuracy. It answers only:
"Can this game legally enter a strict pregame prediction set at this cutoff?"
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.atomic_io import atomic_write_text
from core.pit_evidence import EvidenceLevel, StarterEvidence, strict_eligible


def _event_start(value: Any) -> datetime:
    """Parse the scheduled game start time required by the pregame gate."""
    if value in (None, ""):
        raise ValueError("missing_event_start")
    ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        raise ValueError("event_start_must_be_timezone_aware")
    return ts


def _ts(value: Any) -> str:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).isoformat()


def check_game(row: dict[str, Any], cutoff: str) -> tuple[bool, str]:
    if str(row.get("status", "")).upper() in {"CANCELLED", "POSTPONED"}:
        return False, "game_not_playable"
    if not row.get("event_id") or not row.get("league"):
        return False, "missing_event_identity"
    try:
        start_value = row.get("event_start_at") or row.get("datetime") or row.get("start_datetime")
        if start_value in (None, "") and row.get("official_start_time") and row.get("target_date"):
            start_value = f"{row['target_date']}T{row['official_start_time']}:00+09:00"
        start_ts = _event_start(start_value)
        cutoff_ts = datetime.fromisoformat(str(cutoff).replace("Z", "+00:00"))
        if start_ts <= cutoff_ts:
            return False, "game_already_started_or_not_future_at_cutoff"
    except (TypeError, ValueError):
        return False, "event_start_invalid"
    for side in ("home", "away"):
        starter = row.get(f"{side}_starter")
        level = row.get(f"{side}_starter_evidence_level", "NONE")
        announced = row.get(f"{side}_starter_announced_at")
        if not starter:
            return False, f"{side}_starter_missing"
        try:
            evidence = StarterEvidence(
                level=EvidenceLevel[str(level).upper()],
                timestamp=_ts(announced) if announced else None,
                source=str(row.get(f"{side}_starter_source", row.get("source", ""))),
                starter=str(starter),
                evidence_id=row.get(f"{side}_starter_evidence_id"),
            )
        except (KeyError, TypeError, ValueError):
            return False, f"{side}_starter_evidence_invalid"
        if not strict_eligible(evidence, cutoff):
            return False, f"{side}_starter_not_strictly_eligible"
    return True, "eligible"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cutoff = _ts(args.cutoff)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("games", [])
    if not isinstance(rows, list):
        raise SystemExit("PIT gate input must contain a list of games")
    if not rows:
        raise SystemExit("PIT gate received zero games; empty evidence cannot pass")
    results = []
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("PIT gate input contains a non-object game row")
        eligible, reason = check_game(dict(row), cutoff)
        results.append({**row, "production_eligible": eligible, "eligibility_reason": reason})
    report = {
        "schema": "production-pit-gate-v1",
        "cutoff": cutoff,
        "strict_contract": "OFFICIAL_ANNOUNCEMENT_ONLY",
        "games": results,
        "eligible_count": sum(bool(x["production_eligible"]) for x in results),
        "excluded_count": sum(not bool(x["production_eligible"]) for x in results),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        atomic_write_text(args.output, text)
    else:
        print(text)
    if any(not x["production_eligible"] for x in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
