from data.availability import AvailabilityRecord, prediction_eligible
from core.pit_replay import replay


BASE = dict(
    event_id="MLB:123",
    league="MLB",
    home_team="Home",
    away_team="Away",
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


def test_both_verified_announcements_are_eligible():
    ok, reasons = prediction_eligible(AvailabilityRecord(**BASE))
    assert ok is True
    assert reasons == []


def test_observed_starter_without_verified_announcement_is_not_eligible():
    row = dict(BASE, home_starter_announced_at=None)
    # A raw observed starter cannot be silently promoted to an announced starter.
    try:
        AvailabilityRecord(**row).validate()
    except ValueError as exc:
        assert "unknown announcement timestamp" in str(exc)
    else:
        raise AssertionError("unverified starter announcement must fail closed")


def test_announcement_after_cutoff_is_rejected():
    row = dict(BASE, home_starter_announced_at="2026-01-01T00:06:00+00:00")
    try:
        AvailabilityRecord(**row).validate()
    except ValueError as exc:
        assert "after prediction cutoff" in str(exc)
    else:
        raise AssertionError("future announcement must be rejected")


def test_replay_excludes_observations_after_cutoff(tmp_path):
    path = tmp_path / "snapshots.jsonl"
    path.write_text(
        '{"event_id":"MLB:123","league":"MLB","entity_type":"game","entity_id":"123","source":"test","status":"KNOWN","available_at":"2026-01-01T00:04:00+00:00","retrieved_at":"2026-01-01T00:04:00+00:00","prediction_cutoff":"2026-01-01T00:04:00+00:00","payload_hash":"a"}\n'
        '{"event_id":"MLB:123","league":"MLB","entity_type":"game","entity_id":"123","source":"test","status":"KNOWN","available_at":"2026-01-01T00:06:00+00:00","retrieved_at":"2026-01-01T00:06:00+00:00","prediction_cutoff":"2026-01-01T00:06:00+00:00","payload_hash":"b"}\n',
        encoding="utf-8",
    )
    rows = replay(path, cutoff="2026-01-01T00:05:00+00:00")
    assert len(rows) == 1
    assert rows[0]["payload_hash"] == "a"
