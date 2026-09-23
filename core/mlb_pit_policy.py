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


class EvidenceClass(str, Enum):
    OFFICIAL_ANNOUNCEMENT = "OFFICIAL_ANNOUNCEMENT"
    OFFICIAL_PUBLICATION = "OFFICIAL_PUBLICATION"
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
        EvidenceClass.THIRD_PARTY_FIRST_SEEN,
        EvidenceClass.RETRIEVAL_ONLY,
    }
