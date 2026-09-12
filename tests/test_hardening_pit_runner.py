import pytest

from data.availability import AvailabilityRecord, prediction_eligible
from prediction.runner import eligibility_gate


def availability(**overrides):
    values = dict(
        event_id="g1",
        league="NPB",
        home_team="読売ジャイアンツ",
        away_team="阪神タイガース",
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
    values.update(overrides)
    return AvailabilityRecord(**values)


def test_retrieval_at_cutoff_is_pit_safe():
    ok, reasons = eligibility_gate(
        availability=availability(),
        required_data_ok=True,
        feature_complete=True,
        model_available=True,
        calibration_available=True,
    )
    assert ok
    assert reasons == []


def test_retrieval_after_cutoff_is_rejected():
    with pytest.raises(ValueError, match="retrieved_at is after prediction cutoff"):
        availability(retrieved_at="2026-01-01T00:06:00+00:00").validate()


def test_starter_without_announcement_time_fails_closed():
    record = availability(home_starter_announced_at=None)
    # Validation accepts an unknown starter name only as an invalid PIT record;
    # production must never silently treat it as confirmed.
    with pytest.raises(ValueError, match="unknown announcement timestamp"):
        record.validate()


def test_unknown_lineup_does_not_unnecessarily_block_prediction():
    ok, reasons = prediction_eligible(availability())
    assert ok
    assert reasons == []


def test_mlb_requires_only_home_and_away_probability_contract():
    record = availability(league="MLB")
    ok, reasons = eligibility_gate(
        availability=record,
        required_data_ok=True,
        feature_complete=True,
        model_available=True,
        calibration_available=True,
    )
    assert ok
    assert reasons == []
