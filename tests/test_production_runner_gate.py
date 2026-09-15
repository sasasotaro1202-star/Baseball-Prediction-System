import pytest

from data.availability import AvailabilityRecord
from prediction.runner import eligibility_gate


def availability(league="NPB"):
    return AvailabilityRecord(
        event_id="g1",
        league=league,
        home_team="HOME",
        away_team="AWAY",
        home_starter="P1",
        away_starter="P2",
        home_starter_announced_at="2026-01-01T00:00:00+00:00",
        away_starter_announced_at="2026-01-01T00:00:00+00:00",
        lineup_status="UNVERIFIABLE",
        lineup_announced_at=None,
        source="test",
        retrieved_at="2026-01-01T00:05:00+00:00",
        prediction_cutoff="2026-01-01T00:05:00+00:00",
    )


def test_research_gate_can_validate_registered_wbc():
    ok, reasons = eligibility_gate(
        availability=availability("WBC"),
        required_data_ok=True,
        feature_complete=True,
        model_available=True,
        calibration_available=True,
    )
    assert ok
    assert reasons == []


def test_production_gate_rejects_research_only_wbc():
    ok, reasons = eligibility_gate(
        availability=availability("WBC"),
        required_data_ok=True,
        feature_complete=True,
        model_available=True,
        calibration_available=True,
        production=True,
    )
    assert not ok
    assert "competition_not_production_eligible" in reasons


def test_production_gate_accepts_pit_safe_npb():
    ok, reasons = eligibility_gate(
        availability=availability("NPB"),
        required_data_ok=True,
        feature_complete=True,
        model_available=True,
        calibration_available=True,
        production=True,
    )
    assert ok
    assert reasons == []


def test_production_gate_fails_closed_for_unknown_competition():
    with pytest.raises(ValueError, match="unknown competition_id/league"):
        eligibility_gate(
            availability=availability("UNKNOWN"),
            required_data_ok=True,
            feature_complete=True,
            model_available=True,
            calibration_available=True,
            production=True,
        )
