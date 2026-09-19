import pandas as pd
import pytest

from core.pit import assert_no_future_rows, filter_as_of
from data.availability import AvailabilityRecord


def test_filter_as_of_requires_explicit_availability_by_default():
    frame = pd.DataFrame(
        [
            {
                "event_id": "g1",
                "event_time": "2026-06-01T10:00:00Z",
                "available_ts": None,
                "value": 123,
            },
            {
                "event_id": "g2",
                "event_time": "2026-06-01T10:00:00Z",
                "available_ts": "2026-06-01T09:00:00Z",
                "value": 456,
            },
        ]
    )

    result = filter_as_of(frame, "2026-06-01T11:00:00Z")

    assert result.rows_in == 2
    assert result.rows_out == 1
    assert result.frame["event_id"].tolist() == ["g2"]


def test_filter_as_of_strict_event_time_only_allows_missing_availability_before_cutoff():
    frame = pd.DataFrame(
        [
            {
                "event_id": "before",
                "event_time": "2026-06-01T10:00:00Z",
                "available_ts": None,
            },
            {
                "event_id": "at_cutoff",
                "event_time": "2026-06-01T11:00:00Z",
                "available_ts": None,
            },
            {
                "event_id": "after",
                "event_time": "2026-06-01T12:00:00Z",
                "available_ts": None,
            },
        ]
    )

    result = filter_as_of(frame, "2026-06-01T11:00:00Z", strict_event_time=True)

    assert result.frame["event_id"].tolist() == ["before"]


def test_filter_as_of_never_admits_future_availability():
    frame = pd.DataFrame(
        [
            {
                "event_id": "future",
                "event_time": "2026-06-01T10:00:00Z",
                "available_ts": "2026-06-01T12:00:00Z",
            }
        ]
    )

    result = filter_as_of(frame, "2026-06-01T11:00:00Z")

    assert result.rows_out == 0


def test_assert_no_future_rows_fails_closed():
    frame = pd.DataFrame(
        [
            {"event_id": "safe", "available_ts": "2026-06-01T10:00:00Z"},
            {"event_id": "future", "available_ts": "2026-06-01T12:00:00Z"},
        ]
    )

    with pytest.raises(ValueError, match="PIT violation"):
        assert_no_future_rows(frame, "2026-06-01T11:00:00Z")


def test_prediction_eligibility_requires_starter_announcement_timestamps():
    record = AvailabilityRecord(
        event_id="g1",
        league="MLB",
        home_team="HOME",
        away_team="AWAY",
        home_starter="P1",
        away_starter="P2",
        home_starter_announced_at=None,
        away_starter_announced_at="2026-06-01T09:00:00Z",
        lineup_status="UNVERIFIABLE",
        lineup_announced_at=None,
        source="test",
        retrieved_at="2026-06-01T10:00:00Z",
        prediction_cutoff="2026-06-01T10:00:00Z",
    )

    with pytest.raises(ValueError, match="home starter has unknown announcement timestamp"):
        record.validate()
