import numpy as np
import pandas as pd
import pytest

from research.temporal_predictability_calibration import (
    calibration_diagnostics,
    calibrate_predictability_walk_forward,
)


def _fixture(n: int = 70):
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    pt = np.array([base + pd.Timedelta(hours=i) for i in range(n)], dtype=object)
    mature = np.array([t + pd.Timedelta(hours=1) for t in pt], dtype=object)
    raw = np.linspace(0.25, 0.85, n)
    correctness = (raw >= 0.55).astype(int)
    return raw, correctness, pt, mature


def test_same_prediction_time_is_excluded():
    raw, y, pt, mature = _fixture(70)
    pt[41] = pt[40]
    mature[41] = pt[41] + pd.Timedelta(hours=1)
    out = calibrate_predictability_walk_forward(
        raw, y, pt, mature, min_calibration=10, max_calibration=25
    )
    assert out["calibration_count"][41] == out["calibration_count"][40]
    assert out["calibration_count"][41] <= 25


def test_immature_outcome_is_excluded_until_confirmation_time():
    raw, y, pt, mature = _fixture(70)
    mature[20] = pt[50]
    out = calibrate_predictability_walk_forward(
        raw, y, pt, mature, min_calibration=10, max_calibration=50
    )
    expected_at_30 = int(
        (
            (pt[:30] < pt[30])
            & (mature[:30] <= pt[30])
        ).sum()
    )
    assert out["calibration_count"][30] == expected_at_30
    assert 20 not in np.flatnonzero(
        (pt[:30] < pt[30]) & (mature[:30] <= pt[30])
    )


def test_history_window_is_bounded():
    raw, y, pt, mature = _fixture(70)
    out = calibrate_predictability_walk_forward(
        raw, y, pt, mature, min_calibration=10, max_calibration=20
    )
    assert int(out["calibration_count"].max()) <= 20


def test_invalid_temporal_order_fails_closed():
    raw, y, pt, mature = _fixture(10)
    mature[3] = pt[3] - pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="cannot precede"):
        calibrate_predictability_walk_forward(raw, y, pt, mature)


def test_unsorted_prediction_times_fail_closed():
    raw, y, pt, mature = _fixture(10)
    pt[4] = pt[3] - pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="monotonically"):
        calibrate_predictability_walk_forward(raw, y, pt, mature)


def test_insufficient_history_uses_explicit_fallback():
    raw, y, pt, mature = _fixture(12)
    out = calibrate_predictability_walk_forward(
        raw, y, pt, mature, min_calibration=10, max_calibration=20, prior_mean=0.6
    )
    assert out["method"][0] == "FALLBACK_PRIOR_MEAN"
    assert out["calibration_count"][0] == 0
    assert np.allclose(out["calibrated_predictability"][:10], 0.6)


def test_deterministic_and_research_only():
    raw, y, pt, mature = _fixture(60)
    a = calibrate_predictability_walk_forward(
        raw, y, pt, mature, min_calibration=10, max_calibration=30
    )
    b = calibrate_predictability_walk_forward(
        raw, y, pt, mature, min_calibration=10, max_calibration=30
    )
    assert np.array_equal(a["calibration_count"], b["calibration_count"])
    assert np.allclose(a["calibrated_predictability"], b["calibrated_predictability"])
    assert a["contract"]["production_changed"] is False
    assert a["contract"]["promotion_allowed"] is False


def test_diagnostics_are_finite():
    raw, y, pt, mature = _fixture(40)
    out = calibrate_predictability_walk_forward(
        raw, y, pt, mature, min_calibration=10, max_calibration=20
    )
    diag = calibration_diagnostics(y, raw, out["calibrated_predictability"])
    assert all(np.isfinite(float(v)) for v in diag.values())
