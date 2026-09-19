"""Audit the PIT ledger for production-readiness evidence.

This is deliberately an evidence audit, not a promotion decision. It checks
that the append-only ledger is structurally valid, distinguishes observed
starter names from verifiable announcement times, and reports market-line and
timestamp coverage without converting missing evidence into a pass.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIT = ROOT / "data" / "pit"
FILES = {
    "events": PIT / "event_observations.jsonl",
    "availability": PIT / "availability_observations.jsonl",
    "snapshots": PIT / "source_snapshots.jsonl",
    "runs": PIT / "acquisition_runs.jsonl",
}


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path}:{n}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise RuntimeError(f"{path}:{n}: row is not an object")
        rows.append(obj)
    return rows


def _dt(value):
    if value in (None, ""):
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


def audit() -> dict:
    rows = {name: _load(path) for name, path in FILES.items()}
    allowed = {"KNOWN", "OBSERVED", "OBSERVED_UNVERIFIABLE_ANNOUNCEMENT_TIME",
               "UNAVAILABLE", "MISSING", "UNVERIFIABLE"}
    status_counts = Counter()
    for name, data in rows.items():
        for row in data:
            status = row.get("status")
            if status is not None:
                status_counts[(name, str(status))] += 1
                if status not in allowed:
                    raise RuntimeError(f"invalid PIT status {status!r} in {name}")

    availability = rows["availability"]
    starter_both = 0
    starter_time_both = 0
    cutoff_safe_both = 0
    event_ids = set()
    duplicate_event_rows = 0
    for row in availability:
        event_id = str(row.get("event_id", ""))
        if event_id in event_ids:
            duplicate_event_rows += 1
        if event_id:
            event_ids.add(event_id)
        hs, aws = row.get("home_starter"), row.get("away_starter")
        ha, aa = row.get("home_starter_announced_at"), row.get("away_starter_announced_at")
        if hs and aws:
            starter_both += 1
        hdt, adt, cutoff = _dt(ha), _dt(aa), _dt(row.get("prediction_cutoff"))
        if hdt and adt:
            starter_time_both += 1
        if hdt and adt and cutoff and hdt <= cutoff and adt <= cutoff:
            cutoff_safe_both += 1

    snapshot_types = Counter(str(x.get("entity_type", "")) for x in rows["snapshots"])
    market_like = [x for x in rows["snapshots"]
                   if "market" in str(x.get("entity_type", "")).lower()
                   or "line" in str(x.get("entity_type", "")).lower()]
    timestamp_like = [x for x in rows["snapshots"]
                      if str(x.get("entity_type", "")) in {"game_feed_timestamps", "game_content"}]

    latest_retrieved = None
    for row in rows["snapshots"]:
        dt = _dt(row.get("retrieved_at"))
        if dt and (latest_retrieved is None or dt > latest_retrieved):
            latest_retrieved = dt

    return {
        "status": "AUDIT_COMPLETE",
        "files": {k: len(v) for k, v in rows.items()},
        "status_counts": {f"{k[0]}:{k[1]}": v for k, v in sorted(status_counts.items())},
        "availability": {
            "rows": len(availability),
            "unique_event_ids": len(event_ids),
            "duplicate_event_rows": duplicate_event_rows,
            "both_starters_observed": starter_both,
            "both_starter_announcement_times": starter_time_both,
            "both_starters_cutoff_safe": cutoff_safe_both,
            "starter_time_coverage": (starter_time_both / len(availability)) if availability else 0.0,
            "cutoff_safe_coverage": (cutoff_safe_both / len(availability)) if availability else 0.0,
        },
        "snapshots": {
            "entity_types": dict(snapshot_types),
            "market_or_line_snapshots": len(market_like),
            "mlb_timeline_supporting_snapshots": len(timestamp_like),
            "latest_retrieved_at": latest_retrieved.isoformat() if latest_retrieved else None,
        },
        "production_readiness": {
            "starter_announcement_evidence": "READY" if cutoff_safe_both else "BLOCKED",
            "historical_market_line_evidence": "READY" if market_like else "BLOCKED",
            "promotion": "BLOCKED_UNTIL_ALL_REQUIRED_PIT_EVIDENCE_AND_HOLDOUT_GATES_PASS",
        },
    }


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2, sort_keys=True))
