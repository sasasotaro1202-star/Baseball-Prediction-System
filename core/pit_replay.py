"""Point-in-time replay over the immutable Baseball PIT snapshot ledger.

The replay rule is strict: only observations whose availability timestamp is
at or before the requested cutoff are eligible. Records with unknown
availability are excluded from a historical pregame replay unless the caller
explicitly requests observed-only mode.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def iter_snapshots(path: str | Path) -> Iterable[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                yield row


def replay(path: str | Path, *, cutoff: str, event_id: str | None = None,
           league: str | None = None, include_unverifiable: bool = False) -> list[dict[str, Any]]:
    c = _dt(cutoff)
    rows: list[dict[str, Any]] = []
    for row in iter_snapshots(path):
        if event_id is not None and str(row.get("event_id")) != str(event_id):
            continue
        if league is not None and str(row.get("league")) != str(league):
            continue
        status = str(row.get("status", "KNOWN"))
        if status in {"UNAVAILABLE", "MISSING"}:
            continue
        available = row.get("available_at")
        if not available:
            if not include_unverifiable:
                continue
        else:
            if _dt(str(available)) > c:
                continue
        retrieved = row.get("retrieved_at")
        if retrieved and _dt(str(retrieved)) > c:
            # A source cannot have been observed after the prediction cutoff.
            continue
        rows.append(row)
    rows.sort(key=lambda r: (_dt(str(r.get("available_at") or r.get("retrieved_at"))), str(r.get("entity_id"))))
    return rows


def latest_known_by_entity(path: str | Path, *, cutoff: str,
                           league: str | None = None) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in replay(path, cutoff=cutoff, league=league):
        key = f"{row.get('league')}:{row.get('entity_type')}:{row.get('entity_id')}:{row.get('source')}"
        latest[key] = row
    return latest
