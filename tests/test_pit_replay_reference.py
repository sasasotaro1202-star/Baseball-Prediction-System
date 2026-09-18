from __future__ import annotations

import json
from pathlib import Path

from core.pit_replay import replay


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _reference_replay(rows: list[dict], cutoff: str) -> list[dict]:
    """Slow, obvious PIT reference used only for regression testing."""
    from datetime import datetime, timezone

    c = datetime.fromisoformat(cutoff.replace("Z", "+00:00")).astimezone(timezone.utc)

    def dt(value: str):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)

    out = []
    for row in rows:
        status = str(row.get("status", "KNOWN"))
        if status in {"UNAVAILABLE", "MISSING"}:
            continue
        available = row.get("available_at")
        retrieved = row.get("retrieved_at")
        if not available:
            continue
        if dt(str(available)) > c:
            continue
        if retrieved and dt(str(retrieved)) > c:
            continue
        out.append(row)

    out.sort(key=lambda r: (
        dt(str(r.get("available_at") or r.get("retrieved_at"))),
        dt(str(r.get("retrieved_at") or r.get("available_at"))),
        str(r.get("entity_id")),
        str(r.get("source")),
    ))
    return out


def test_replay_matches_slow_reference_implementation(tmp_path: Path):
    rows = [
        {
            "event_id": "MLB:1",
            "league": "MLB",
            "entity_type": "game",
            "entity_id": "1",
            "source": "B",
            "status": "KNOWN",
            "available_at": "2026-09-12T09:00:00Z",
            "retrieved_at": "2026-09-12T09:06:00Z",
        },
        {
            "event_id": "MLB:1",
            "league": "MLB",
            "entity_type": "game",
            "entity_id": "1",
            "source": "A",
            "status": "KNOWN",
            "available_at": "2026-09-12T09:00:00Z",
            "retrieved_at": "2026-09-12T09:05:00Z",
        },
        {
            "event_id": "MLB:1",
            "league": "MLB",
            "entity_type": "game",
            "entity_id": "1",
            "source": "FUTURE",
            "status": "KNOWN",
            "available_at": "2026-09-12T11:00:00Z",
            "retrieved_at": "2026-09-12T11:01:00Z",
        },
        {
            "event_id": "MLB:1",
            "league": "MLB",
            "entity_type": "game",
            "entity_id": "1",
            "source": "UNKNOWN",
            "status": "UNVERIFIABLE",
            "available_at": None,
            "retrieved_at": "2026-09-12T09:30:00Z",
        },
    ]
    p = tmp_path / "snapshots.jsonl"
    _write(p, rows)

    got = replay(p, cutoff="2026-09-12T10:00:00Z")
    expected = _reference_replay(rows, "2026-09-12T10:00:00Z")

    assert got == expected


def test_replay_is_invariant_to_snapshot_file_permutation(tmp_path: Path):
    rows = [
        {
            "event_id": "MLB:1",
            "league": "MLB",
            "entity_type": "game",
            "entity_id": "1",
            "source": source,
            "status": "KNOWN",
            "available_at": "2026-09-12T09:00:00Z",
            "retrieved_at": f"2026-09-12T09:0{minute}:00Z",
        }
        for source, minute in [("A", 5), ("B", 6), ("C", 7)]
    ]

    p1 = tmp_path / "ordered.jsonl"
    p2 = tmp_path / "permuted.jsonl"
    _write(p1, rows)
    _write(p2, [rows[2], rows[0], rows[1]])

    got1 = replay(p1, cutoff="2026-09-12T10:00:00Z")
    got2 = replay(p2, cutoff="2026-09-12T10:00:00Z")

    assert got1 == got2
