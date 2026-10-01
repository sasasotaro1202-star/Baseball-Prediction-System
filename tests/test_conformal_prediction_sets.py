import numpy as np
import pytest

from research.conformal_prediction_sets import (
    decision_from_prediction_set,
    group_split_conformal_prediction_sets,
    prediction_set_metrics,
    split_conformal_prediction_sets,
    validate_chronological_calibration,
    validate_prediction_set_output,
)


def _temporal_fixture(calibration_rows: int, test_rows: int):
    cal_pred = [
        f"2026-01-{1 + i // 24:02d}T{(i % 24):02d}:00:00Z"
        for i in range(calibration_rows)
    ]
    test_pred = [
        f"2026-03-{1 + i // 24:02d}T{(i % 24):02d}:00:00Z"
        for i in range(test_rows)
    ]
    cal_available = [
        f"2025-12-{28 + (i // 24):02d}T{(i % 24):02d}:00:00Z"
        for i in range(calibration_rows)
    ]
    cal_outcome = [
        f"2026-02-{1 + i // 24:02d}T{(i % 24):02d}:00:00Z"
        for i in range(calibration_rows)
    ]
    test_available = [
        f"2026-02-{28 + (i // 24):02d}T{(i % 24):02d}:00:00Z"
        for i in range(test_rows)
    ]
    return cal_pred, test_pred, cal_available, cal_outcome, test_available


def _fixture():
    y = np.array([0, 0, 1, 1, 2, 2] * 8)
    p = np.array([
        [0.90, 0.08, 0.02],
        [0.82, 0.15, 0.03],
        [0.05, 0.90, 0.05],
        [0.10, 0.84, 0.06],
        [0.06, 0.12, 0.82],
        [0.12, 0.10, 0.78],
    ] * 8, dtype=float)
    p_test = np.array([
        [0.90, 0.07, 0.03],
        [0.25, 0.50, 0.25],
        [0.02, 0.03, 0.95],
    ], dtype=float)
    return y, p, p_test


def test_split_conformal_sets_are_deterministic_and_cover_expected_labels():
    y, p, p_test = _fixture()
    times = _temporal_fixture(len(y), len(p_test))
    out = split_conformal_prediction_sets(
        y,
        p,
        p_test,
        alpha=0.10,
        class_names=("home", "draw", "away"),
        calibration_prediction_times=times[0],
        test_prediction_times=times[1],
        calibration_available_at=times[2],
        calibration_outcome_available_at=times[3],
        test_available_at=times[4],
    )
    assert out["method"] == "split_conformal"
    assert out["finite_sample_correction"] is True
    assert out["class_names"] == ["home", "draw", "away"]
    assert len(out["prediction_sets"]) == len(p_test)
    assert out["prediction_sets"][0] == ["home"]
    assert out["prediction_sets"][2] == ["away"]
    metrics = prediction_set_metrics(out, [0, 1, 2])
    assert metrics["set_coverage"] >= 2 / 3
    assert 0.0 <= metrics["mean_set_size"] <= 3.0


def test_group_conformal_uses_global_fallback_for_small_groups():
    y, p, p_test = _fixture()
    groups = np.array(["NPB"] * len(y))
    times = _temporal_fixture(len(y), len(p_test))
    out = group_split_conformal_prediction_sets(
        y,
        p,
        groups,
        p_test,
        ["NPB", "MLB", "NPB"],
        alpha=0.10,
        class_names=("home", "draw", "away"),
        min_group_size=30,
        calibration_prediction_times=times[0],
        test_prediction_times=times[1],
        calibration_available_at=times[2],
        calibration_outcome_available_at=times[3],
        test_available_at=times[4],
    )
    assert out["group_fallback"].tolist() == [False, True, False]
    assert 0.0 < out["fallback_rate"] < 1.0


def test_temporal_contract_fails_closed_on_overlap_future_outcome_and_bad_availability():
    with pytest.raises(ValueError):
        validate_chronological_calibration(
            ["2026-01-02T00:00:00Z"],
            ["2026-01-02T00:00:00Z"],
            calibration_available_at=["2026-01-01T23:00:00Z"],
            calibration_outcome_available_at=["2026-01-02T01:00:00Z"],
            test_available_at=["2026-01-01T23:00:00Z"],
        )
    with pytest.raises(ValueError):
        validate_chronological_calibration(
            ["2026-01-01T00:00:00Z"],
            ["2026-01-02T00:00:00Z"],
            calibration_available_at=["2026-01-01T01:00:00Z"],
            calibration_outcome_available_at=["2026-01-03T01:00:00Z"],
            test_available_at=["2026-01-01T23:00:00Z"],
        )
    with pytest.raises(ValueError):
        validate_chronological_calibration(
            ["2026-01-01T00:00:00Z"],
            ["2026-01-02T00:00:00Z"],
            calibration_available_at=["2025-12-31T23:00:00Z"],
            calibration_outcome_available_at=["2025-12-31T22:00:00Z"],
            test_available_at=["2026-01-01T23:00:00Z"],
        )
    good = validate_chronological_calibration(
        ["2026-01-01T00:00:00Z"],
        ["2026-01-02T00:00:00Z"],
        calibration_available_at=["2025-12-31T23:00:00Z"],
        calibration_outcome_available_at=["2026-01-01T12:00:00Z"],
        test_available_at=["2026-01-01T23:00:00Z"],
    )
    assert good["status"] == "PASS"


def test_split_conformal_requires_temporal_evidence_at_the_calculation_boundary():
    y, p, p_test = _fixture()
    with pytest.raises(TypeError):
        split_conformal_prediction_sets(
            y, p, p_test, alpha=0.10, class_names=("home", "draw", "away")
        )


def test_split_conformal_rejects_future_calibration_outcomes():
    y, p, p_test = _fixture()
    times = _temporal_fixture(len(y), len(p_test))
    future_outcomes = list(times[3])
    future_outcomes[-1] = "2026-03-02T00:00:00Z"
    with pytest.raises(ValueError, match="outcomes"):
        split_conformal_prediction_sets(
            y,
            p,
            p_test,
            alpha=0.10,
            class_names=("home", "draw", "away"),
            calibration_prediction_times=times[0],
            test_prediction_times=times[1],
            calibration_available_at=times[2],
            calibration_outcome_available_at=future_outcomes,
            test_available_at=times[4],
        )


def test_prediction_set_output_contract_is_fail_closed():
    validate_prediction_set_output(["home"], class_names=("home", "draw", "away"))
    with pytest.raises(ValueError):
        validate_prediction_set_output([], class_names=("home", "draw", "away"))
    with pytest.raises(ValueError):
        validate_prediction_set_output(["home", "home"], class_names=("home", "draw", "away"))
    with pytest.raises(ValueError):
        validate_prediction_set_output(["tie"], class_names=("home", "draw", "away"))
    assert decision_from_prediction_set([])["action"] == "ABSTAIN"
    assert decision_from_prediction_set(["home"])["action"] == "SINGLE"
    assert decision_from_prediction_set(["home", "draw"])["action"] == "SET"
