import pytest

from data.market_lines import TotalRunsLine, classify_total, low_high_threshold
from prediction.prediction_log import PredictionRecord, append_prediction, make_prediction_id, validate_prediction


def line(**overrides):
    values = dict(
        event_id="g1", league="MLB", line=7.5, source="book",
        observed_at="2026-01-01T00:00:00+00:00",
        available_at="2026-01-01T00:01:00+00:00", status="KNOWN",
    )
    values.update(overrides)
    return TotalRunsLine(**values)


def record(**overrides):
    values = dict(
        prediction_id=make_prediction_id("g1", "2026-01-01T00:05:00+00:00", "m1", "abc"),
        event_id="g1", league="NPB",
        prediction_cutoff="2026-01-01T00:05:00+00:00",
        prediction_created_at="2026-01-01T00:05:01+00:00",
        home_team="読売ジャイアンツ", away_team="阪神タイガース",
        home_starter="P1", away_starter="P2",
        probabilities={"home": 0.5, "draw": 0.2, "away": 0.3},
        score_candidates=[{"score": "3-2", "probability": 0.1}],
        low_probability=0.6, high_probability=0.4, total_runs_line=7.5,
        confidence=0.8, volatility=0.2,
        model_version="m1", feature_version="f1", calibration_version="c1",
        git_commit="abc", data_snapshot_id="snap1",
    )
    values.update(overrides)
    return PredictionRecord(**values)


def test_market_line_rejects_naive_timestamp():
    with pytest.raises(ValueError, match="timezone-aware"):
        line(observed_at="2026-01-01T00:00:00").validate()


def test_market_line_requires_available_at_for_known_line():
    with pytest.raises(ValueError, match="available_at"):
        line(available_at=None).validate()


def test_market_line_pit_requires_both_observed_and_available_before_cutoff():
    assert line().pit_usable("2026-01-01T00:05:00+00:00")
    assert not line(observed_at="2026-01-01T00:06:00+00:00").pit_usable("2026-01-01T00:05:00+00:00")
    assert not line(available_at="2026-01-01T00:06:00+00:00").pit_usable("2026-01-01T00:05:00+00:00")


def test_integer_total_has_push_not_low_or_high():
    assert classify_total(7, 7) == "PUSH"
    assert classify_total(6, 7) == "LOW"
    assert classify_total(8, 7) == "HIGH"
    assert low_high_threshold(7.5) == (7.0, 8.0)


def test_prediction_rejects_non_finite_probability():
    with pytest.raises(ValueError, match="finite"):
        validate_prediction(record(probabilities={"home": float("nan"), "draw": 0.2, "away": 0.8}))


def test_prediction_rejects_low_high_mismatch():
    with pytest.raises(ValueError, match="sum to 1"):
        validate_prediction(record(low_probability=0.7, high_probability=0.4))


def test_prediction_ledger_is_idempotent(tmp_path):
    path = tmp_path / "predictions.jsonl"
    r = record()
    append_prediction(r, path)
    append_prediction(r, path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_prediction_rejects_creation_before_cutoff():
    with pytest.raises(ValueError, match="cannot precede"):
        validate_prediction(record(prediction_created_at="2026-01-01T00:04:59+00:00"))
