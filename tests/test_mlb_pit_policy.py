from datetime import datetime, timezone
from core.mlb_pit_policy import EvidenceClass, MLBStarterEvidence, production_eligible, research_only


def ev(level, ts="2026-09-19T10:00:00+00:00"):
    return MLBStarterEvidence("1","home","123","Pitcher",level,datetime.fromisoformat(ts), "test")


def test_only_official_announcement_is_production_eligible():
    cutoff = datetime(2026,9,19,12,tzinfo=timezone.utc)
    assert production_eligible(ev(EvidenceClass.OFFICIAL_ANNOUNCEMENT), cutoff)
    assert not production_eligible(ev(EvidenceClass.OFFICIAL_PUBLICATION), cutoff)
    assert not production_eligible(ev(EvidenceClass.THIRD_PARTY_FIRST_SEEN), cutoff)
    assert not production_eligible(ev(EvidenceClass.RETRIEVAL_ONLY), cutoff)


def test_lower_grade_is_research_only():
    assert research_only(ev(EvidenceClass.THIRD_PARTY_FIRST_SEEN))
