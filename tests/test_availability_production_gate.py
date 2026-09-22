from data.availability import AvailabilityRecord, production_prediction_eligible


def _record(source: str) -> AvailabilityRecord:
    return AvailabilityRecord(
        event_id="SMOKE-1",
        league="MLB",
        home_team="Home",
        away_team="Away",
        home_starter="Starter H",
        away_starter="Starter A",
        home_starter_announced_at="2026-09-23T09:00:00+00:00",
        away_starter_announced_at="2026-09-23T09:00:00+00:00",
        lineup_status="UNCONFIRMED",
        lineup_announced_at=None,
        source=source,
        retrieved_at="2026-09-23T10:00:00+00:00",
        prediction_cutoff="2026-09-23T12:00:00+00:00",
    )


def test_production_gate_rejects_non_official_source():
    ok, reasons = production_prediction_eligible(_record("https://example.com/"))
    assert not ok
    assert "starter_source_not_official" in reasons


def test_production_gate_preserves_research_only_competition_block():
    ok, reasons = production_prediction_eligible(_record("https://mlb.com/"))
    assert not ok
    assert "competition_not_production_eligible" in reasons


def test_production_gate_rejects_missing_event_start_time():
    record = _record("https://mlb.com/")
    ok, reasons = production_prediction_eligible(record)
    assert not ok
    assert "event_start_time_missing" in reasons


def test_production_gate_rejects_already_started_event():
    base = _record("https://mlb.com/")
    record = AvailabilityRecord(
        **{
            **base.__dict__,
            "event_start_at": "2026-09-23T11:00:00+00:00",
        }
    )
    ok, reasons = production_prediction_eligible(record)
    assert not ok
    assert "event_already_started_or_not_future" in reasons
