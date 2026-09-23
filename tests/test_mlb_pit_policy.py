from datetime import datetime, timezone
from core.mlb_pit_policy import EvidenceClass, MLBStarterEvidence, production_eligible, research_only


def ev(level, ts="2026-09-19T10:00:00+00:00", source="https://www.mlb.com/"):
    return MLBStarterEvidence("1","home","123","Pitcher",level,datetime.fromisoformat(ts), source)


def test_only_official_announcement_is_production_eligible():
    cutoff = datetime(2026,9,19,12,tzinfo=timezone.utc)
    assert production_eligible(ev(EvidenceClass.OFFICIAL_ANNOUNCEMENT), cutoff)
    assert not production_eligible(ev(EvidenceClass.OFFICIAL_PUBLICATION), cutoff)
    assert not production_eligible(ev(EvidenceClass.THIRD_PARTY_FIRST_SEEN), cutoff)
    assert not production_eligible(ev(EvidenceClass.RETRIEVAL_ONLY), cutoff)


def test_lower_grade_is_research_only():
    assert research_only(ev(EvidenceClass.THIRD_PARTY_FIRST_SEEN))


def test_official_announcement_requires_first_party_mlb_source():
    cutoff = datetime(2026, 9, 19, 12, tzinfo=timezone.utc)
    assert production_eligible(
        ev(EvidenceClass.OFFICIAL_ANNOUNCEMENT, source="https://www.mlb.com/schedule/")
        , cutoff
    )
    assert not production_eligible(
        ev(EvidenceClass.OFFICIAL_ANNOUNCEMENT, source="https://example.com/mlb")
        , cutoff
    )
    assert not production_eligible(
        ev(EvidenceClass.OFFICIAL_ANNOUNCEMENT, source="")
        , cutoff
    )


def test_official_announcement_requires_timezone_aware_timestamps():
    cutoff = datetime(2026, 9, 19, 12, tzinfo=timezone.utc)
    naive = MLBStarterEvidence(
        "1", "home", "123", "Pitcher",
        EvidenceClass.OFFICIAL_ANNOUNCEMENT,
        datetime(2026, 9, 19, 10),
        "https://www.mlb.com/",
    )
    assert not production_eligible(naive, cutoff)
