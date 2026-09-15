"""Detect source revisions and backfills in the immutable PIT snapshot ledger."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


def _dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        raise ValueError("PIT timestamps must be timezone-aware")
    return ts.astimezone(timezone.utc)


def _key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("league", "")),
        str(row.get("entity_type", "")),
        str(row.get("entity_id", row.get("event_id", ""))),
        str(row.get("source", "")),
    )


@dataclass(frozen=True)
class RevisionFinding:
    key: tuple[str, str, str, str]
    kind: str
    previous_hash: str | None
    current_hash: str | None
    previous_available_at: str | None
    current_available_at: str | None
    current_retrieved_at: str | None


def detect_revisions(rows: Iterable[dict[str, Any]]) -> list[RevisionFinding]:
    """Return deterministic revision/backfill findings without mutating the ledger.

    A changed payload hash for the same source/entity is a REVISION. A later
    observation that claims an availability/source timestamp earlier than an
    already observed timestamp is a BACKFILL. Unknown hashes are never treated
    as equal, so missing provenance cannot silently suppress a finding.
    """
    history: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    findings: list[RevisionFinding] = []
    ordered = sorted(
        rows,
        key=lambda r: (_dt(r.get("retrieved_at")) or datetime.min.replace(tzinfo=timezone.utc), str(r.get("payload_hash", ""))),
    )
    for row in ordered:
        key = _key(row)
        current_hash = str(row["payload_hash"]) if row.get("payload_hash") not in (None, "") else None
        current_available = _dt(row.get("available_at"))
        current_source = _dt(row.get("source_timestamp"))
        current_retrieved = _dt(row.get("retrieved_at"))
        previous = history.get(key)
        if previous is not None:
            previous_hash = previous.get("payload_hash")
            previous_available = previous.get("available_at")
            previous_source = previous.get("source_timestamp")
            if current_hash is not None and previous_hash is not None and current_hash != previous_hash:
                findings.append(RevisionFinding(key, "REVISION", previous_hash, current_hash,
                                                previous_available.isoformat() if previous_available else None,
                                                current_available.isoformat() if current_available else None,
                                                current_retrieved.isoformat() if current_retrieved else None))
            if ((current_available and previous_available and current_available < previous_available)
                    or (current_source and previous_source and current_source < previous_source)):
                findings.append(RevisionFinding(key, "BACKFILL", previous_hash, current_hash,
                                                previous_available.isoformat() if previous_available else None,
                                                current_available.isoformat() if current_available else None,
                                                current_retrieved.isoformat() if current_retrieved else None))
        history[key] = {
            "payload_hash": current_hash,
            "available_at": current_available,
            "source_timestamp": current_source,
        }
    return findings
