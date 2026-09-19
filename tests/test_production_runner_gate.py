import pytest

from data.availability import AvailabilityRecord
from prediction.runner import eligibility_gate, run_production_prediction


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


def test_production_gate_rejects_pit_safe_npb_until_adoption():
    ok, reasons = eligibility_gate(
        availability=availability("NPB"),
        required_data_ok=True,
        feature_complete=True,
        model_available=True,
        calibration_available=True,
        production=True,
    )
    assert not ok
    assert reasons == ["competition_not_production_eligible"]


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


def test_explicit_production_entry_point_cannot_fall_back_to_research_mode():
    future = "2999-01-01T00:00:00+00:00"
    # The wrapper must force production=True even if a caller tries to pass
    # production=False. NPB is currently fail-closed, so the forced production
    # gate must reject the prediction before any filesystem side effect occurs.
    result = run_production_prediction(
            row={
                "required_data_ok": True,
                "feature_complete": True,
                "model_available": True,
                "calibration_available": True,
            },
            availability=AvailabilityRecord(
                event_id="g2",
                league="NPB",
                home_team="HOME",
                away_team="AWAY",
                home_starter="P1",
                away_starter="P2",
                home_starter_announced_at="2998-12-31T23:00:00+00:00",
                away_starter_announced_at="2998-12-31T23:00:00+00:00",
                lineup_status="UNVERIFIABLE",
                lineup_announced_at=None,
                source="test",
                retrieved_at=future,
                prediction_cutoff=future,
            ),
            probability_fn=lambda row: {"home": 0.5, "draw": 0.2, "away": 0.3},
            model_version="test",
            feature_version="test",
            calibration_version="test",
            git_commit="test",
            data_snapshot_id="test",
            log_path="/tmp/unused.jsonl",
            production=False,
        )
    assert result["eligible"] is False
    assert result["reasons"] == ["competition_not_production_eligible"]
