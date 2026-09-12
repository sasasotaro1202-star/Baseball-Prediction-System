from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core.pit import filter_as_of
from core.pit_snapshot import make_snapshot
from data.availability import AvailabilityRecord, prediction_eligible
from data.market_lines import TotalRunsLine, classify_total
from evaluation.calibration import TemperatureCalibration, fit_temperature


def test_pit_excludes_future_availability():
    df = pd.DataFrame({"event_id": ["a", "b"], "event_time": ["2026-01-01T00:00:00Z"] * 2,
                       "available_ts": ["2025-12-31T23:00:00Z", "2026-01-01T01:00:00Z"]})
    out = filter_as_of(df, "2026-01-01T00:00:00Z")
    assert out.rows_out == 1 and out.frame.iloc[0]["event_id"] == "a"


def test_snapshot_rejects_future_availability():
    try:
        make_snapshot(event_id="g", league="MLB", entity_type="starter", entity_id="p",
                      source="test", payload={}, prediction_cutoff="2026-01-01T00:00:00+00:00",
                      available_at="2026-01-01T00:00:01+00:00")
    except ValueError:
        return
    raise AssertionError("future availability was accepted")


def test_starter_gate_requires_both():
    record = AvailabilityRecord("g", "MLB", "A", "B", "p1", None, "2026-01-01T00:00:00+00:00", None,
                                "UNVERIFIABLE", None, "test", "2025-12-31T23:00:00+00:00", "2026-01-01T00:00:00+00:00")
    ok, reasons = prediction_eligible(record)
    assert not ok and "away_starter_not_confirmed" in reasons


def test_market_line_classification():
    line = TotalRunsLine("g", "MLB", 7.5, "test", "2026-01-01T00:00:00+00:00", "2025-12-31T23:00:00+00:00")
    line.validate()
    assert classify_total(7, 7.5) == "LOW"
    assert classify_total(8, 7.5) == "HIGH"


def test_temperature_calibration_preserves_probability_contract():
    p = np.array([[0.8, 0.2], [0.3, 0.7]])
    q = TemperatureCalibration(1.0).transform(p)
    assert np.allclose(q.sum(axis=1), 1.0)
    assert np.all(q >= 0) and np.all(q <= 1)
