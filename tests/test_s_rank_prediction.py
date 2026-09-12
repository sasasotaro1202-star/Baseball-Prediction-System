from prediction.prediction_log import PredictionRecord, make_prediction_id, validate_prediction
from prediction.runner import eligibility_gate
from data.availability import AvailabilityRecord


def _availability():
    return AvailabilityRecord(
        event_id="g1", league="NPB", home_team="A", away_team="B",
        home_starter="p1", away_starter="p2",
        home_starter_announced_at="2026-01-01T00:00:00+00:00",
        away_starter_announced_at="2026-01-01T00:00:00+00:00",
        lineup_status="UNVERIFIABLE", lineup_announced_at=None, source="test",
        retrieved_at="2026-01-01T00:05:00+00:00",
        prediction_cutoff="2026-01-01T00:05:00+00:00",
    )


def test_prediction_id_is_deterministic():
    a = make_prediction_id("g1", "2026-01-01T00:05:00+00:00", "m1", "abc")
    b = make_prediction_id("g1", "2026-01-01T00:05:00+00:00", "m1", "abc")
    assert a == b


def test_runner_gate_accepts_confirmed_starters():
    ok, reasons = eligibility_gate(availability=_availability(), required_data_ok=True,
                                   feature_complete=True, model_available=True,
                                   calibration_available=True)
    assert ok and reasons == []


def test_npb_prediction_contract_requires_three_probabilities():
    record = PredictionRecord(
        prediction_id="x", event_id="g1", league="NPB",
        prediction_cutoff="2026-01-01T00:05:00+00:00",
        prediction_created_at="2026-01-01T00:05:01+00:00",
        home_team="A", away_team="B", home_starter="p1", away_starter="p2",
        probabilities={"home": .5, "draw": .2, "away": .3}, score_candidates=[],
        low_probability=None, high_probability=None, total_runs_line=None,
        confidence=None, volatility=None, model_version="m1", feature_version="f1",
        calibration_version="c1", git_commit="abc", data_snapshot_id="s1",
    )
    validate_prediction(record)
