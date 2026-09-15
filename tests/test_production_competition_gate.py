from __future__ import annotations

from data.availability import AvailabilityRecord, prediction_eligible, production_prediction_eligible
from data.competition_registry import production_eligible


def _record(league: str) -> AvailabilityRecord:
    return AvailabilityRecord(
        event_id=f"gate-{league}",
        league=league,
        home_team="HOME",
        away_team="AWAY",
        home_starter="Starter H",
        away_starter="Starter A",
        home_starter_announced_at="2026-09-15T07:00:00+09:00",
        away_starter_announced_at="2026-09-15T07:00:00+09:00",
        lineup_status="UNCONFIRMED",
        lineup_announced_at=None,
        source="test",
        retrieved_at="2026-09-15T08:00:00+09:00",
        prediction_cutoff="2026-09-15T08:30:00+09:00",
    )


def test_research_only_competition_can_pass_pit_but_not_production():
    record = _record("WBC")
    assert prediction_eligible(record) == (True, [])
    ok, reasons = production_prediction_eligible(record)
    assert not ok
    assert reasons == ["competition_not_production_eligible"]
    assert production_eligible("WBC") is False


def test_npb_remains_production_eligible_when_pit_safe():
    record = _record("NPB")
    assert production_eligible("NPB") is True
    assert production_prediction_eligible(record) == (True, [])
