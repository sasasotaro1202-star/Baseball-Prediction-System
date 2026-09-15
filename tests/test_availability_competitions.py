from __future__ import annotations

import pytest

from data.availability import AvailabilityRecord, prediction_eligible


BASE = dict(
    event_id="WBC:123",
    league="WBC",
    home_team="Japan",
    away_team="USA",
    home_starter="Pitcher A",
    away_starter="Pitcher B",
    home_starter_announced_at="2026-01-01T00:00:00+00:00",
    away_starter_announced_at="2026-01-01T00:00:00+00:00",
    lineup_status="UNVERIFIABLE",
    lineup_announced_at=None,
    source="test",
    retrieved_at="2026-01-01T00:05:00+00:00",
    prediction_cutoff="2026-01-01T00:05:00+00:00",
)


def test_registered_research_competition_can_be_validated_without_becoming_production_eligible():
    record = AvailabilityRecord(**BASE)
    record.validate()
    ok, reasons = prediction_eligible(record)
    assert ok is True
    assert reasons == []


def test_unknown_competition_still_fails_closed():
    row = dict(BASE, league="NOT_A_REAL_COMPETITION")
    with pytest.raises(ValueError):
        AvailabilityRecord(**row).validate()
