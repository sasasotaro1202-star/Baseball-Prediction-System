"""PIT snapshot/provenance contract for Baseball data.

Snapshots are immutable observations. A snapshot may have been retrieved after
an arbitrary historical cutoff; that is precisely why replay must filter by
retrieval/availability time instead of refusing to store the observation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

_ALLOWED = {"KNOWN", "MISSING", "UNAVAILABLE", "UNVERIFIABLE"}


def _dt(value: str) -> datetime:
    ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        raise ValueError("PIT timestamps must be timezone-aware")
    return ts


@dataclass(frozen=True)
class SourceSnapshot:
    event_id: str
    league: str
    entity_type: str
    entity_id: str
    source: str
    source_timestamp: str | None
    retrieved_at: str
    available_at: str | None
    prediction_cutoff: str
    payload_hash: str
    status: str = "KNOWN"

    def validate(self) -> None:
        if not self.event_id or not self.entity_id or not self.source:
            raise ValueError("event_id, entity_id and source are required")
        if self.league not in {"NPB", "MLB"}:
            raise ValueError("league must be NPB or MLB")
        if self.status not in _ALLOWED:
            raise ValueError(f"invalid snapshot status: {self.status}")
        _dt(self.retrieved_at)
        _dt(self.prediction_cutoff)
        if self.source_timestamp:
            _dt(self.source_timestamp)
        if self.available_at:
            available = _dt(self.available_at)
            retrieved = _dt(self.retrieved_at)
            if available > retrieved:
                raise ValueError("available_at cannot be after retrieved_at")
            # Do not reject future observations here. core.pit_replay is the
            # authoritative PIT gate and excludes observations unavailable at
            # the requested replay cutoff.


def payload_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def make_snapshot(*, event_id: str, league: str, entity_type: str, entity_id: str,
                  source: str, payload: Any, prediction_cutoff: str,
                  available_at: str | None, source_timestamp: str | None = None,
                  retrieved_at: str | None = None, status: str = "KNOWN") -> SourceSnapshot:
    snap = SourceSnapshot(
        event_id=str(event_id), league=league, entity_type=entity_type,
        entity_id=str(entity_id), source=source,
        source_timestamp=source_timestamp,
        retrieved_at=retrieved_at or datetime.now(timezone.utc).isoformat(),
        available_at=available_at, prediction_cutoff=prediction_cutoff,
        payload_hash=payload_hash(payload), status=status,
    )
    snap.validate()
    return snap


def append_snapshot(snapshot: SourceSnapshot, path: str | Path) -> None:
    """Append an auditable JSONL snapshot; existing records are never rewritten."""
    snapshot.validate()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(snapshot), ensure_ascii=False, sort_keys=True) + "\n")


def validate_snapshot_rows(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {k: 0 for k in _ALLOWED}
    for row in rows:
        status = str(row.get("status", "KNOWN")).upper()
        if status not in counts:
            raise ValueError(f"invalid snapshot status: {status}")
        counts[status] += 1
    return counts
