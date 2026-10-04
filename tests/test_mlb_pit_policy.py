from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

from core.mlb_pit_policy import (
    EvidenceClass,
    MLBStarterEvidence,
    derive_first_observed_evidence,
    pit_available_by,
    production_eligible,
    research_only,
)


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



def test_first_observed_is_research_availability_evidence_not_announcement():
    cutoff = datetime(2026, 9, 19, 12, tzinfo=timezone.utc)
    observations = [
        {
            "game_id": "1",
            "home_starter_id": "123",
            "home_starter": "Pitcher",
            "source": "https://www.mlb.com/schedule/",
            "available_at": "2026-09-19T10:30:00+00:00",
            "observed_at": "2026-09-19T10:30:00+00:00",
            "retrieved_at": "2026-09-19T10:31:00+00:00",
            "payload_hash": "later",
        },
        {
            "game_id": "1",
            "home_starter_id": "123",
            "home_starter": "Pitcher",
            "source": "https://www.mlb.com/schedule/",
            "available_at": "2026-09-19T10:00:00+00:00",
            "observed_at": "2026-09-19T10:00:00+00:00",
            "retrieved_at": "2026-09-19T10:01:00+00:00",
            "payload_hash": "earlier",
        },
    ]
    evidence = derive_first_observed_evidence(
        observations, game_id="1", side="home", starter_id="123", starter_name="Pitcher"
    )
    assert evidence.evidence_class is EvidenceClass.OFFICIAL_FIRST_OBSERVED
    assert evidence.timestamp == datetime(2026, 9, 19, 10, tzinfo=timezone.utc)
    assert pit_available_by(evidence, cutoff)
    assert not production_eligible(evidence, cutoff)



def test_first_observed_ignores_availability_after_retrieval():
    observations = [
        {
            "game_id": "1",
            "home_starter_id": "123",
            "home_starter": "Pitcher",
            "source": "https://www.mlb.com/schedule/",
            "available_at": "2026-09-19T10:30:00+00:00",
            "observed_at": "2026-09-19T09:05:00+00:00",
            "retrieved_at": "2026-09-19T09:10:00+00:00",
            "payload_hash": "malformed-available",
        }
    ]
    evidence = derive_first_observed_evidence(
        observations, game_id="1", side="home", starter_id="123", starter_name="Pitcher"
    )
    assert evidence.evidence_class is EvidenceClass.OFFICIAL_FIRST_OBSERVED
    assert evidence.timestamp == datetime(2026, 9, 19, 9, 5, tzinfo=timezone.utc)


def test_first_observed_rejects_only_future_observation_timestamp():
    observations = [
        {
            "game_id": "1",
            "home_starter_id": "123",
            "home_starter": "Pitcher",
            "source": "https://www.mlb.com/schedule/",
            "available_at": "2026-09-19T10:30:00+00:00",
            "retrieved_at": "2026-09-19T09:10:00+00:00",
            "payload_hash": "future-only",
        }
    ]
    evidence = derive_first_observed_evidence(
        observations, game_id="1", side="home", starter_id="123", starter_name="Pitcher"
    )
    assert evidence.evidence_class is EvidenceClass.NONE
    assert evidence.timestamp is None

def test_first_observed_does_not_backdate_changed_starter_or_accept_third_party():
    cutoff = datetime(2026, 9, 19, 12, tzinfo=timezone.utc)
    observations = [
        {
            "game_id": "1",
            "home_starter_id": "111",
            "home_starter": "Pitcher A",
            "source": "https://www.mlb.com/schedule/",
            "available_at": "2026-09-19T08:00:00+00:00",
            "payload_hash": "a",
        },
        {
            "game_id": "1",
            "home_starter_id": "222",
            "home_starter": "Pitcher B",
            "source": "https://third-party.example/",
            "available_at": "2026-09-19T08:30:00+00:00",
            "payload_hash": "third-party",
        },
        {
            "game_id": "1",
            "home_starter_id": "222",
            "home_starter": "Pitcher B",
            "source": "https://www.mlb.com/schedule/",
            "available_at": "2026-09-19T10:00:00+00:00",
            "payload_hash": "b",
        },
    ]
    evidence = derive_first_observed_evidence(
        observations, game_id="1", side="home", starter_id="222", starter_name="Pitcher B"
    )
    assert evidence.evidence_class is EvidenceClass.OFFICIAL_FIRST_OBSERVED
    assert evidence.timestamp == datetime(2026, 9, 19, 10, tzinfo=timezone.utc)
    assert pit_available_by(evidence, cutoff)


def test_mlb_pit_policy_workflow_is_ref_scoped():
    workflow = (ROOT / ".github" / "workflows" / "mlb_pit_policy.yml").read_text(encoding="utf-8")
    assert "group: mlb-pit-policy-${{ github.event.pull_request.number || github.ref }}" in workflow


def test_mlb_starter_id_is_read_from_probable_pitcher():
    import data.pit_acquisition as pit

    assert pit._starter_id({"probablePitcher": {"id": 123}}) == "123"
    assert pit._starter_id({"probablePitcher": {"playerId": "456"}}) == "456"
    assert pit._starter_id({"probablePitcher": {}}) is None
    assert pit._starter_id({}) is None

def test_mlb_pit_policy_workflow_installs_requests_dependency():
    workflow = (ROOT / ".github" / "workflows" / "mlb_pit_policy.yml").read_text(encoding="utf-8")
    assert "python -m pip install --disable-pip-version-check pytest requests" in workflow


def test_explicit_mlb_starter_timestamp_rejects_future_of_retrieval():
    import data.pit_acquisition as pit

    retrieval = "2026-09-19T09:10:00+00:00"
    assert pit._explicit_timestamp(
        {"home_starter_announced_at": "2026-09-19T09:11:00+00:00"},
        "home",
        "announcement",
        retrieved_at=retrieval,
    ) is None


def test_explicit_mlb_starter_timestamp_accepts_at_retrieval():
    import data.pit_acquisition as pit

    retrieval = "2026-09-19T09:10:00+00:00"
    assert pit._explicit_timestamp(
        {"home_starter_announced_at": retrieval},
        "home",
        "announcement",
        retrieved_at=retrieval,
    ) == retrieval


def test_explicit_mlb_starter_timestamp_rejects_malformed_retrieval_boundary():
    import data.pit_acquisition as pit

    assert pit._explicit_timestamp(
        {"home_starter_announced_at": "2026-09-19T09:05:00+00:00"},
        "home",
        "announcement",
        retrieved_at="not-a-timestamp",
    ) is None


def test_explicit_mlb_starter_timestamp_rejects_naive_retrieval_boundary():
    import data.pit_acquisition as pit

    assert pit._explicit_timestamp(
        {"home_starter_announced_at": "2026-09-19T09:05:00+00:00"},
        "home",
        "announcement",
        retrieved_at="2026-09-19T09:10:00",
    ) is None
