import numpy as np
import pandas as pd
import pytest

from research.temporal_conformal_prediction_sets import (
    temporal_prediction_set_metrics,
    validate_maturity_contract,
    walk_forward_split_conformal,
)


def _fixture(n=80):
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    pt = np.array([base + pd.Timedelta(hours=i) for i in range(n)], dtype=object)
    mature = np.array([t + pd.Timedelta(hours=1) for t in pt], dtype=object)
    y = np.array([0, 1, 2] * ((n + 2) // 3))[:n]
    p = np.tile(
        np.array([[.80, .15, .05], [.10, .80, .10], [.05, .10, .85]], dtype=float),
        ((n + 2) // 3, 1),
    )[:n]
    return y, p, pt, mature


def test_maturity_contract_allows_tied_prediction_times():
    y, p, pt, mature = _fixture(6)
    pt[1] = pt[0]
    mature[1] = pt[1] + pd.Timedelta(hours=1)
    out = validate_maturity_contract(pt, mature)
    assert out["status"] == "PASS"


def test_maturity_contract_rejects_unsorted_or_impossible_timestamps():
    y, p, pt, mature = _fixture(6)
    bad = pt.copy()
    bad[2] = bad[1] - pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="monotonically"):
        validate_maturity_contract(bad, mature)
    impossible = mature.copy()
    impossible[2] = pt[2] - pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="cannot precede"):
        validate_maturity_contract(pt, impossible)


def test_walk_forward_excludes_same_time_and_immature_outcomes():
    y, p, pt, mature = _fixture(60)
    # One prior prediction has a late-confirmed outcome and must not enter case 40.
    mature[39] = pt[50]
    out = walk_forward_split_conformal(
        y, p, pt, mature, alpha=.10,
        class_names=("home", "draw", "away"),
        min_calibration=30,
    )
    assert out["calibration_count"][30] == 29
    assert out["action"][30] == "ABSTAIN"
    assert out["calibration_count"][31] == 30
    assert out["action"][31] == "SINGLE"


def test_walk_forward_is_maturity_gated_not_retrospective():
    y, p, pt, mature = _fixture(80)
    # Make the most recent 40 outcomes unavailable until well after the target range.
    mature[0:40] = pt[79] + pd.Timedelta(hours=1)
    out = walk_forward_split_conformal(
        y, p, pt, mature, alpha=.10,
        class_names=("home", "draw", "away"),
        min_calibration=10,
    )
    # No target before row 50 may use those immature outcomes.
    assert out["calibration_count"][20] == 0
    assert out["calibration_count"][50] == 10
    assert out["maturity_contract"]["same_prediction_time_excluded"] is True


def test_rolling_calibration_window_is_bounded():
    y, p, pt, mature = _fixture(70)
    out = walk_forward_split_conformal(
        y, p, pt, mature, alpha=.10,
        class_names=("home", "draw", "away"),
        min_calibration=10,
        max_calibration=20,
    )
    assert out["calibration_count"][25] == 20
    assert out["calibration_count"].max() <= 20


def test_temporal_metrics_use_only_eligible_rows():
    y, p, pt, mature = _fixture(60)
    out = walk_forward_split_conformal(
        y, p, pt, mature, alpha=.10,
        class_names=("home", "draw", "away"),
        min_calibration=30,
    )
    metrics = temporal_prediction_set_metrics(out, y)
    assert metrics["total_rows"] == 60
    assert metrics["eligible_rows"] == 30
    assert metrics["coverage_eligible"] >= 2 / 3
    assert 0.0 <= metrics["maturity_gated_fraction"] <= 1.0
