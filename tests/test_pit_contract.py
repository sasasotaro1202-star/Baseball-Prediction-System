from datetime import datetime, timezone

from research.pit_contract import production_starter_eligible, validate_starter_evidence


CUTOFF = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def test_official_starter_before_cutoff_is_eligible():
    record = {
        "status": "official_announced",
        "published_at": "2026-09-19T10:00:00Z",
    }
    assert production_starter_eligible(record, prediction_cutoff=CUTOFF)


def test_probable_pitcher_is_not_confirmed():
    record = {
        "status": "official_probable_only",
        "published_at": "2026-09-19T10:00:00Z",
    }
    assert not production_starter_eligible(record, prediction_cutoff=CUTOFF)


def test_late_announcement_is_rejected():
    record = {
        "status": "official_announced",
        "published_at": "2026-09-19T13:00:00Z",
    }
    ok, reasons = validate_starter_evidence([record], prediction_cutoff=CUTOFF)
    assert not ok
    assert "record_0:starter_available_after_prediction_cutoff" in reasons
