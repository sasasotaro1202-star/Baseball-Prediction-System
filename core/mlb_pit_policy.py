"""MLB starter evidence policy.

External feeds may be useful corroboration, but they are never promoted to
OFFICIAL_ANNOUNCEMENT automatically. First-seen timestamps are retained as
separate evidence and remain research-only under the strict production gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from urllib.parse import urlparse
from typing import Mapping


class EvidenceClass(str, Enum):
    OFFICIAL_ANNOUNCEMENT = "OFFICIAL_ANNOUNCEMENT"
    OFFICIAL_PUBLICATION = "OFFICIAL_PUBLICATION"
    OFFICIAL_FIRST_OBSERVED = "OFFICIAL_FIRST_OBSERVED"
    THIRD_PARTY_FIRST_SEEN = "THIRD_PARTY_FIRST_SEEN"
    RETRIEVAL_ONLY = "RETRIEVAL_ONLY"
    NONE = "NONE"


@dataclass(frozen=True)
class MLBStarterEvidence:
    game_id: str
    side: str
    starter_id: str | None
    starter_name: str | None
    evidence_class: EvidenceClass
    timestamp: datetime | None
    source: str
    source_url: str | None = None
    source_record_id: str | None = None


def _is_official_mlb_source(source: str, source_url: str | None) -> bool:
    """Require an allowlisted MLB first-party host for production evidence."""
    value = (source_url or source or "").strip()
    if not value:
        return False
    try:
        host = (urlparse(value).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    return host == "mlb.com" or host.endswith(".mlb.com")



def _parse_observation_time(row: Mapping[str, object]) -> datetime | None:
    """Return a PIT-safe observation time without treating it as announcement time."""
    retrieval_value = row.get("retrieved_at")
    retrieval = None
    if retrieval_value not in (None, ""):
        try:
            retrieval = datetime.fromisoformat(str(retrieval_value).replace("Z", "+00:00"))
        except ValueError:
            retrieval = None

    # Prefer explicit availability/observation timestamps. If one is present
    # but malformed or later than retrieval, do not silently substitute the
    # retrieval instant; that would fabricate a PIT boundary from invalid data.
    has_explicit_boundary = any(
        row.get(key) not in (None, "") for key in ("available_at", "observed_at")
    )
    for key in ("available_at", "observed_at"):
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            continue
        if ts.tzinfo is None:
            continue
        if retrieval is not None and retrieval.tzinfo is not None and ts > retrieval:
            continue
        return ts

    if has_explicit_boundary:
        return None
    if retrieval is not None and retrieval.tzinfo is not None:
        return retrieval
    return None


def derive_first_observed_evidence(
    observations: list[Mapping[str, object]],
    *,
    game_id: str,
    side: str,
    starter_id: str | None = None,
    starter_name: str | None = None,
) -> MLBStarterEvidence:
    """Derive the earliest official observation of a specific starter.

    This timestamp is an availability boundary proven by the supplied immutable
    observation ledger, not an inferred announcement timestamp. A sparse ledger
    can establish "available by T" but never "announced at T". Exact starter
    identity matching is required; fuzzy name matching is not used.
    """
    if side not in {"home", "away"}:
        raise ValueError("side must be home or away")
    if not starter_id and not starter_name:
        return MLBStarterEvidence(
            game_id=str(game_id), side=side, starter_id=starter_id,
            starter_name=starter_name, evidence_class=EvidenceClass.NONE,
            timestamp=None, source="",
        )

    matches: list[tuple[datetime, Mapping[str, object]]] = []
    id_key = f"{side}_starter_id"
    name_key = f"{side}_starter"
    for row in observations:
        if str(row.get("game_id") or row.get("event_id") or "") != str(game_id):
            continue
        source = str(row.get("source") or row.get("source_url") or "")
        source_url = str(row.get("source_url") or "") or None
        if not _is_official_mlb_source(source, source_url):
            continue
        status = str(row.get("status", "KNOWN")).upper()
        if status in {"UNAVAILABLE", "MISSING", "SOURCE_FAILED"}:
            continue
        observed_id = str(row.get(id_key) or "")
        observed_name = str(row.get(name_key) or "")
        identity_match = (
            bool(starter_id) and bool(observed_id) and observed_id == str(starter_id)
        ) or (
            bool(starter_name) and bool(observed_name) and observed_name == str(starter_name)
        )
        if not identity_match:
            continue
        ts = _parse_observation_time(row)
        if ts is not None:
            matches.append((ts, row))

    if not matches:
        return MLBStarterEvidence(
            game_id=str(game_id), side=side, starter_id=starter_id,
            starter_name=starter_name, evidence_class=EvidenceClass.NONE,
            timestamp=None, source="",
        )

    first_ts, first_row = min(
        matches,
        key=lambda item: (item[0], str(item[1].get("payload_hash") or "")),
    )
    source = str(first_row.get("source") or first_row.get("source_url") or "")
    source_url = str(first_row.get("source_url") or "") or None
    return MLBStarterEvidence(
        game_id=str(game_id),
        side=side,
        starter_id=starter_id,
        starter_name=starter_name,
        evidence_class=EvidenceClass.OFFICIAL_FIRST_OBSERVED,
        timestamp=first_ts,
        source=source,
        source_url=source_url,
        source_record_id=str(first_row.get("payload_hash") or "") or None,
    )


def pit_available_by(evidence: MLBStarterEvidence, cutoff: datetime) -> bool:
    """Check whether the evidence proves starter availability by cutoff.

    FIRST_OBSERVED is accepted only for research PIT replay. The production gate
    remains stricter and requires an explicit official announcement timestamp.
    """
    return (
        evidence.evidence_class in {
            EvidenceClass.OFFICIAL_ANNOUNCEMENT,
            EvidenceClass.OFFICIAL_FIRST_OBSERVED,
            EvidenceClass.OFFICIAL_PUBLICATION,
        }
        and evidence.timestamp is not None
        and evidence.timestamp.tzinfo is not None
        and cutoff.tzinfo is not None
        and evidence.timestamp <= cutoff
        and _is_official_mlb_source(evidence.source, evidence.source_url)
    )


def production_eligible(evidence: MLBStarterEvidence, cutoff: datetime) -> bool:
    return (
        evidence.evidence_class is EvidenceClass.OFFICIAL_ANNOUNCEMENT
        and bool(evidence.game_id)
        and evidence.side in {"home", "away"}
        and bool(evidence.starter_id or evidence.starter_name)
        and evidence.timestamp is not None
        and evidence.timestamp.tzinfo is not None
        and cutoff.tzinfo is not None
        and evidence.timestamp <= cutoff
        and _is_official_mlb_source(evidence.source, evidence.source_url)
    )


def research_only(evidence: MLBStarterEvidence) -> bool:
    return evidence.evidence_class in {
        EvidenceClass.OFFICIAL_PUBLICATION,
        EvidenceClass.OFFICIAL_FIRST_OBSERVED,
        EvidenceClass.THIRD_PARTY_FIRST_SEEN,
        EvidenceClass.RETRIEVAL_ONLY,
    }
