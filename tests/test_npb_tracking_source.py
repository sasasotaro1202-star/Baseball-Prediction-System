import pandas as pd
import pytest

from research.npb_tracking_source import (
    TrackingPITError,
    lagged_player_summary,
    normalize_tracking_frame,
)


def test_tracking_normalization_and_pit_pass():
    frame = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "event_time": "2026-09-20T10:00:00Z",
                "pitch_speed_kmh": 150,
                "spin_rate_rpm": 2400,
                "available_at": "2026-09-20T10:01:00Z",
                "prediction_time": "2026-09-21T00:00:00Z",
            }
        ]
    )
    out = normalize_tracking_frame(frame, source_id="npb_tracking_research")
    assert out.loc[0, "pitch_speed_kmh"] == 150
    assert out.loc[0, "spin_rate_rpm"] == 2400
    assert out.loc[0, "tracking_pit_status"] == "PASS"


def test_tracking_pit_missing_availability_fails_closed():
    frame = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "event_time": "2026-09-20T10:00:00Z",
                "pitch_speed_kmh": 150,
                "prediction_time": "2026-09-21T00:00:00Z",
            }
        ]
    )
    with pytest.raises(TrackingPITError):
        normalize_tracking_frame(frame, source_id="npb_tracking_research")


def test_tracking_pit_future_value_fails_closed():
    frame = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "event_time": "2026-09-20T10:00:00Z",
                "pitch_speed_kmh": 150,
                "available_at": "2026-09-22T00:00:00Z",
                "prediction_time": "2026-09-21T00:00:00Z",
            }
        ]
    )
    with pytest.raises(TrackingPITError):
        normalize_tracking_frame(frame, source_id="npb_tracking_research")


def test_lagged_player_summary_is_observation_driven():
    rows = []
    for i in range(10):
        rows.append(
            {
                "player_id": "p1",
                "event_time": f"2026-09-{10+i:02d}T00:00:00Z",
                "pitch_speed_kmh": 145 + i,
                "spin_rate_rpm": 2200 + 10 * i,
            }
        )
    out = lagged_player_summary(pd.DataFrame(rows), min_observations=10)
    assert len(out) == 1
    assert out.loc[0, "tracking_observations"] == 10
    assert out.loc[0, "pitch_speed_kmh_last"] == 154
