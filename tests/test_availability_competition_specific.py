from data.availability import AvailabilityRecord, prediction_eligible

def _record(league, **kwargs):
    base = {
        "event_id": "e1", "league": league, "home_team": "A", "away_team": "B",
        "home_starter": None, "away_starter": None,
        "home_starter_announced_at": None, "away_starter_announced_at": None,
        "lineup_status": "UNVERIFIABLE", "lineup_announced_at": None,
        "source": "https://example.test",
        "retrieved_at": "2026-09-01T07:00:00Z",
        "prediction_cutoff": "2026-09-01T07:30:00Z",
        "event_start_at": "2026-09-01T10:00:00Z",
    }
    base.update(kwargs)
    return AvailabilityRecord(**base)

def test_competitions_without_starter_requirement_can_pass_pregame_gate():
    record = _record("NCAA_D1_BASEBALL")
    ok, reasons = prediction_eligible(record)
    assert ok is True
    assert reasons == []

def test_competitions_requiring_starter_evidence_still_fail_closed():
    record = _record("KBO")
    ok, reasons = prediction_eligible(record)
    assert ok is False
    assert "home_starter_not_confirmed" in reasons
    assert "away_starter_not_confirmed" in reasons