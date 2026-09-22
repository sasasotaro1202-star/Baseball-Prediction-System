from core.pit_evidence import (
    EvidenceLevel,
    StarterEvidence,
    best_evidence,
    eligibility_reason,
    strict_eligible,
)


def test_only_official_announcement_can_pass():
    cutoff = "2026-09-19T12:00:00+00:00"
    assert strict_eligible(
        StarterEvidence(EvidenceLevel.OFFICIAL_ANNOUNCEMENT, "2026-09-19T10:00:00+00:00", "https://www.mlb.com/team/example", "Pitcher"),
        cutoff,
    )
    assert not strict_eligible(
        StarterEvidence(EvidenceLevel.OFFICIAL_PUBLICATION, "2026-09-19T10:00:00+00:00", "mlb.com", "Pitcher"),
        cutoff,
    )
    assert not strict_eligible(
        StarterEvidence(EvidenceLevel.THIRD_PARTY_FIRST_SEEN, "2026-09-19T10:00:00+00:00", "tracker", "Pitcher"),
        cutoff,
    )


def test_never_backdate_future_evidence():
    cutoff = "2026-09-19T12:00:00+00:00"
    future = StarterEvidence(EvidenceLevel.OFFICIAL_ANNOUNCEMENT, "2026-09-19T13:00:00+00:00", "https://www.mlb.com/team/example", "Pitcher")
    assert not strict_eligible(future, cutoff)
    assert best_evidence([future], cutoff) is None


def test_reason_is_explicit_for_lower_grade_evidence():
    cutoff = "2026-09-19T12:00:00+00:00"
    pub = StarterEvidence(EvidenceLevel.OFFICIAL_PUBLICATION, "2026-09-19T10:00:00+00:00", "mlb.com", "Pitcher")
    assert eligibility_reason(pub, cutoff) == "official_publication_not_announcement_proof"


def test_official_announcement_requires_allowlisted_official_domain():
    import pytest
    bad = StarterEvidence(
        EvidenceLevel.OFFICIAL_ANNOUNCEMENT,
        "2026-09-19T10:00:00+00:00",
        "https://example.com/mlb.com/announcement",
        "Pitcher",
    )
    with pytest.raises(ValueError, match="allowlisted official source"):
        strict_eligible(bad, "2026-09-19T12:00:00+00:00")
