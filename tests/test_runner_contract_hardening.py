import pytest
from prediction.runner import _validate_final_probabilities, _validate_score_candidates


def test_probability_mapping_rejects_nan_and_inf():
    with pytest.raises(ValueError):
        _validate_final_probabilities({"home": float("nan"), "away": 1.0}, "MLB")
    with pytest.raises(ValueError):
        _validate_final_probabilities({"home": float("inf"), "away": 0.0}, "MLB")


def test_probability_contract_rejects_unknown_keys():
    with pytest.raises(ValueError):
        _validate_final_probabilities({"home": 0.5, "away": 0.4, "draw": 0.1}, "MLB")


def test_score_candidates_reject_invalid_probability():
    with pytest.raises(ValueError):
        _validate_score_candidates([{"score": "3-2", "probability": float("nan")}])


def test_score_candidates_reject_malformed_item():
    with pytest.raises(ValueError):
        _validate_score_candidates([{"score": "3-2"}])



def test_prediction_record_preserves_competition_identity():
    from prediction.prediction_log import PredictionRecord, validate_prediction

    record = PredictionRecord(
        prediction_id="p1",
        event_id="WBC:1",
        league="WBC",
        competition_key="WBC:wbc_regular:regular_season",
        competition_stage="regular_season",
        prediction_cutoff="2026-01-01T00:00:00+00:00",
        prediction_created_at="2026-01-01T00:01:00+00:00",
        home_team="Japan",
        away_team="USA",
        home_starter="A",
        away_starter="B",
        probabilities={"home": 0.6, "away": 0.4},
        score_candidates=[{"score": "3-2", "probability": 0.25}] * 4,
        low_probability=None,
        high_probability=None,
        total_runs_line=None,
        confidence=None,
        volatility=None,
        model_version="m",
        feature_version="f",
        calibration_version="c",
        git_commit="g",
        data_snapshot_id="d",
    )
    # Current production league contracts remain intentionally NPB/MLB-specific;
    # a non-production research record may be serialized separately, but this
    # test only verifies the identity fields are preserved by the dataclass.
    assert record.competition_key == "WBC:wbc_regular:regular_season"
    assert record.competition_stage == "regular_season"
