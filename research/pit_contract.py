"""Point-in-time evidence contract for baseball pregame information.

A starter name alone is not evidence that the name was knowable before the
prediction cutoff. Production eligibility therefore requires an explicit
official-announcement status plus a publication/availability timestamp that is
at or before the prediction cutoff.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping, Sequence


def _parse(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def validate_starter_evidence(
    records: Sequence[Mapping[str, object]],
    *,
    prediction_cutoff: datetime,
) -> tuple[bool, tuple[str, ...]]:
    cutoff = _parse(prediction_cutoff)
    if cutoff is None:
        return False, ("invalid_prediction_cutoff",)
    reasons: list[str] = []
    for i, record in enumerate(records):
        if record.get("status") != "official_announced":
            reasons.append(f"record_{i}:starter_not_officially_announced")
            continue
        available = _parse(record.get("available_at") or record.get("published_at"))
        if available is None:
            reasons.append(f"record_{i}:missing_publication_or_availability_time")
            continue
        if available > cutoff:
            reasons.append(f"record_{i}:starter_available_after_prediction_cutoff")
    return not reasons, tuple(reasons)


def production_starter_eligible(record: Mapping[str, object], *, prediction_cutoff: datetime) -> bool:
    ok, _ = validate_starter_evidence([record], prediction_cutoff=prediction_cutoff)
    return ok
