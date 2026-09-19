"""Evidence ladder for strict point-in-time starter eligibility.

The key distinction is between:
- proof that a starter was officially announced;
- proof that an official source publicly named the starter;
- third-party first-seen observations; and
- local retrieval observations.

Only the first class can satisfy the strict production starter-announcement
contract. Lower classes remain useful for audit/research, but are never promoted
to an official announcement timestamp by inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from typing import Any, Mapping


class EvidenceLevel(IntEnum):
    NONE = 0
    RETRIEVAL_ONLY = 1
    THIRD_PARTY_FIRST_SEEN = 2
    OFFICIAL_PUBLICATION = 3
    OFFICIAL_ANNOUNCEMENT = 4


@dataclass(frozen=True)
class StarterEvidence:
    level: EvidenceLevel
    timestamp: str | None
    source: str
    starter: str | None = None
    evidence_id: str | None = None
    note: str = ""

    def validate(self) -> None:
        if not self.source:
            raise ValueError("source is required")
        if self.timestamp is not None:
            ts = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                raise ValueError("evidence timestamp must be timezone-aware")
        if self.level >= EvidenceLevel.OFFICIAL_ANNOUNCEMENT and not self.starter:
            raise ValueError("official announcement evidence requires starter identity")


def normalize_level(value: Any) -> EvidenceLevel:
    if isinstance(value, EvidenceLevel):
        return value
    return EvidenceLevel[str(value).upper()]


def strict_eligible(
    evidence: StarterEvidence | Mapping[str, Any] | None,
    cutoff: str,
) -> bool:
    """Return true only for explicit official-announcement evidence <= cutoff."""
    if evidence is None:
        return False
    if not isinstance(evidence, StarterEvidence):
        evidence = StarterEvidence(
            level=normalize_level(evidence.get("level", "NONE")),
            timestamp=evidence.get("timestamp"),
            source=str(evidence.get("source", "")),
            starter=evidence.get("starter"),
            evidence_id=evidence.get("evidence_id"),
            note=str(evidence.get("note", "")),
        )
    evidence.validate()
    if evidence.level != EvidenceLevel.OFFICIAL_ANNOUNCEMENT or not evidence.timestamp:
        return False
    cutoff_ts = datetime.fromisoformat(str(cutoff).replace("Z", "+00:00"))
    if cutoff_ts.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    evidence_ts = datetime.fromisoformat(evidence.timestamp.replace("Z", "+00:00"))
    return evidence_ts <= cutoff_ts


def best_evidence(
    candidates: list[StarterEvidence],
    cutoff: str,
) -> StarterEvidence | None:
    """Select the strongest evidence that existed by cutoff; never backdate."""
    valid = []
    cutoff_ts = datetime.fromisoformat(str(cutoff).replace("Z", "+00:00"))
    if cutoff_ts.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    for item in candidates:
        item.validate()
        if not item.timestamp:
            continue
        ts = datetime.fromisoformat(item.timestamp.replace("Z", "+00:00"))
        if ts <= cutoff_ts:
            valid.append(item)
    if not valid:
        return None
    return max(valid, key=lambda x: (int(x.level), x.timestamp or ""))


def eligibility_reason(
    evidence: StarterEvidence | Mapping[str, Any] | None,
    cutoff: str,
) -> str:
    if evidence is None:
        return "starter_announcement_evidence_missing"
    if strict_eligible(evidence, cutoff):
        return "eligible_official_announcement"
    if isinstance(evidence, Mapping):
        level = normalize_level(evidence.get("level", "NONE"))
    else:
        level = evidence.level
    if level == EvidenceLevel.OFFICIAL_PUBLICATION:
        return "official_publication_not_announcement_proof"
    if level == EvidenceLevel.THIRD_PARTY_FIRST_SEEN:
        return "third_party_first_seen_not_official"
    if level == EvidenceLevel.RETRIEVAL_ONLY:
        return "retrieval_time_not_announcement_proof"
    return "starter_announcement_evidence_unverifiable"
