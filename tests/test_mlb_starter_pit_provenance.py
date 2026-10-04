from __future__ import annotations

import pytest

from data.availability import AvailabilityRecord, from_mapping
from data.pit_acquisition import _explicit_timestamp


def _record(**overrides):
    base = dict(
        event_id="MLB:1",
        league="MLB",
        home_team="Home",
        away_team="Away",
        home_starter="Starter H",
        away_starter="Starter A",
        home_starter_announced_at="2026-09-01T08:00:00+00:00",
        away_starter_announced_at="2026-09-01T08:10:00+00:00",
        lineup_status="UNVERIFIABLE",
        lineup_announced_at=None,
        source="https://www.mlb.com/probable-pitchers",
        retrieved_at="2026-09-01T09:00:00+00:00",
        prediction_cutoff="2026-09-01T09:00:00+00:00",
        event_start_at="2026-09-01T23:00:00+00:00",
    )
    base.update(overrides)
    return AvailabilityRecord(**base)


def test_explicit_starter_publication_and_availability_are_preserved():
    record = _record(
        home_starter_published_at="2026-09-01T07:55:00+00:00",
        away_starter_published_at="2026-09-01T08:05:00+00:00",
        home_starter_available_at="2026-09-01T07:55:00+00:00",
        away_starter_available_at="2026-09-01T08:05:00+00:00",
    )
    record.validate()
    mapped = from_mapping(record.__dict__)
    assert mapped.home_starter_published_at == record.home_starter_published_at
    assert mapped.away_starter_available_at == record.away_starter_available_at


def test_starter_publication_after_announcement_fails_closed():
    record = _record(
        home_starter_published_at="2026-09-01T08:05:00+00:00",
    )
    with pytest.raises(ValueError, match="publication is after announcement"):
        record.validate()


def test_starter_timestamps_after_retrieval_fail_closed():
    record = _record(
        home_starter_published_at="2026-09-01T09:01:00+00:00",
        home_starter_announced_at="2026-09-01T09:02:00+00:00",
    )
    # AvailabilityRecord validates the announcement boundary before the
    # publication boundary. Both timestamps are therefore rejected closed,
    # with the first invalid PIT condition reported deterministically.
    with pytest.raises(ValueError, match="announcement is after source retrieval"):
        record.validate()

def test_revision_time_after_cutoff_fails_closed():
    record = _record(revision_time="2026-09-01T09:01:00+00:00")
    with pytest.raises(ValueError, match="revision_time is after prediction cutoff"):
        record.validate()


def test_starter_availability_after_cutoff_fails_closed():
    record = _record(
        home_starter_available_at="2026-09-01T09:01:00+00:00",
    )
    with pytest.raises(ValueError, match="availability is after prediction cutoff"):
        record.validate()


def test_available_timestamp_without_announcement_does_not_create_eligibility():
    record = _record(
        home_starter=None,
        home_starter_announced_at=None,
        home_starter_available_at="2026-09-01T07:55:00+00:00",
    )
    with pytest.raises(ValueError, match="starter has unknown announcement timestamp"):
        record.validate()


def test_explicit_timestamp_parser_requires_timezone_and_known_kind():
    payload = {
        "home_starter_announced_at": "2026-09-01T08:00:00Z",
        "home_starter_published_at": "2026-09-01T07:55:00Z",
        "home_starter_available_at": "2026-09-01T07:55:00Z",
    }
    assert _explicit_timestamp(payload, "home", "announcement") == "2026-09-01T08:00:00+00:00"
    assert _explicit_timestamp(payload, "home", "published") == "2026-09-01T07:55:00+00:00"
    assert _explicit_timestamp(payload, "home", "available") == "2026-09-01T07:55:00+00:00"
    assert _explicit_timestamp(payload, "", "revision") is None
    with pytest.raises(ValueError, match="unsupported starter timestamp kind"):
        _explicit_timestamp(payload, "home", "unknown")
