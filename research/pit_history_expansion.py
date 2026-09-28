"""Measure replayable PIT history across the expanded baseball scope.

This is an evidence/coverage audit only. It never infers historical availability,
never rewrites snapshots, and never promotes a competition.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from core.pit_replay import replay

ROOT = Path(__file__).resolve().parents[1]
PIT_FILE = ROOT / "data" / "pit" / "source_snapshots.jsonl"


def _load_rows() -> list[dict]:
    if not PIT_FILE.exists() or PIT_FILE.stat().st_size == 0:
        return []
    rows: list[dict] = []
    for line_no, line in enumerate(PIT_FILE.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"non-object PIT row at line {line_no}")
        rows.append(obj)
    return rows


def audit() -> dict:
    rows = _load_rows()
    leagues = sorted({str(r.get("league", "")).strip() for r in rows if str(r.get("league", "")).strip()})
    sources = sorted({str(r.get("source", "")).strip() for r in rows if str(r.get("source", "")).strip()})

    status_counts = Counter(str(r.get("status", "KNOWN")) for r in rows)
    available = []
    for row in rows:
        available_at = row.get("available_at")
        retrieved_at = row.get("retrieved_at")
        cutoff = row.get("prediction_cutoff")
        if not available_at or not retrieved_at or not cutoff:
            continue
        try:
            a = datetime.fromisoformat(str(available_at).replace("Z", "+00:00"))
            r = datetime.fromisoformat(str(retrieved_at).replace("Z", "+00:00"))
            c = datetime.fromisoformat(str(cutoff).replace("Z", "+00:00"))
        except ValueError:
            continue
        if a.tzinfo is None or r.tzinfo is None or c.tzinfo is None:
            continue
        if a.astimezone(timezone.utc) <= c.astimezone(timezone.utc) and r.astimezone(timezone.utc) <= c.astimezone(timezone.utc):
            available.append(row)

    replayable_by_league: dict[str, int] = {}
    replayable_by_source: dict[str, int] = defaultdict(int)
    cutoff_counts: Counter[str] = Counter()
    for row in available:
        league = str(row.get("league", "")).strip()
        source = str(row.get("source", "")).strip()
        if league:
            replayable_by_league[league] = replayable_by_league.get(league, 0) + 1
        if source:
            replayable_by_source[source] += 1
        cutoff = str(row.get("prediction_cutoff", "")).strip()
        if cutoff:
            cutoff_counts[cutoff] += 1

    # Validate a bounded set of real recorded cutoffs, never synthesizing one.
    replay_checks = []
    for cutoff in sorted(cutoff_counts)[-10:]:
        for league in leagues:
            matched = replay(PIT_FILE, cutoff=cutoff, league=league)
            replay_checks.append({
                "cutoff": cutoff,
                "league": league,
                "rows": len(matched),
            })

    return {
        "status": "AUDIT_COMPLETE",
        "source_snapshot_file": (
            str(PIT_FILE.relative_to(ROOT))
            if PIT_FILE.is_relative_to(ROOT)
            else str(PIT_FILE)
        ),
        "snapshot_rows": len(rows),
        "leagues": leagues,
        "sources": sources,
        "status_counts": dict(sorted(status_counts.items())),
        "replayable_rows": len(available),
        "replayable_by_league": dict(sorted(replayable_by_league.items())),
        "replayable_by_source": dict(sorted(replayable_by_source.items())),
        "recorded_cutoffs_checked": len(sorted(cutoff_counts)[-10:]),
        "replay_checks": replay_checks,
        "production_promotion": "NEVER_BY_PIT_HISTORY_AUDIT",
        "unknown_pit": "FAIL_CLOSED",
    }


def main() -> int:
    print(json.dumps(audit(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
