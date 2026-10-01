from pathlib import Path

import pytest

from prediction.prediction_log import PredictionRecord, record_from_mapping, validate_prediction


def _record(**overrides):
    base = dict(
        prediction_id="test-v18-001",
        event_id="game-001",
        league="NPB",
        prediction_cutoff="2026-09-30T09:00:00+00:00",
        prediction_created_at="2026-09-30T09:01:00+00:00",
        home_team="HOME",
        away_team="AWAY",
        home_starter="PITCHER-H",
        away_starter="PITCHER-A",
        probabilities={"home": 0.55, "draw": 0.10, "away": 0.35},
        score_candidates=[
            {"score": "2-1", "probability": 0.25},
            {"score": "1-1", "probability": 0.20},
            {"score": "1-2", "probability": 0.18},
            {"score": "2-2", "probability": 0.10},
        ],
        low_probability=0.62,
        high_probability=0.38,
        total_runs_line=6.5,
        confidence=0.55,
        volatility=0.20,
        model_version="model-v1",
        feature_version="features-v1",
        calibration_version="cal-v1",
        git_commit="abc123",
        data_snapshot_id="snapshot-1",
    )
    base.update(overrides)
    return PredictionRecord(**base)


def test_prediction_record_roundtrips_v18_metadata(tmp_path):
    intelligence = {
        "prediction_time": "2026-09-30T09:00:00+00:00",
        "valid_until": "2026-09-30T10:00:00+00:00",
        "pit_status": "PASS",
        "provenance": {"source": "unit-test", "snapshot": "snapshot-1"},
        "confidence": 0.55,
        "predictability": 0.72,
        "uncertainty": 0.28,
        "disagreement": 0.14,
        "ood": 0.05,
        "failure_risk": 0.11,
        "freshness": 0.91,
        "information_value": 0.17,
        "forecast_lifetime": 1.0,
        "action": "MAINTAIN",
        "update_need": "LOW",
    }
    record = _record(
        prediction_set=["home"],
        prediction_set_alpha=0.10,
        prediction_set_method="split_conformal",
        prediction_set_action="SINGLE",
        prediction_intelligence=intelligence,
    )
    ledger = tmp_path / "predictions.jsonl"
    ledger.write_text(__import__("json").dumps(record.__dict__, ensure_ascii=False) + "\n", encoding="utf-8")
    loaded = record_from_mapping(__import__("json").loads(ledger.read_text(encoding="utf-8")))
    assert loaded.prediction_set == ["home"]
    assert loaded.prediction_set_method == "split_conformal"
    assert loaded.prediction_intelligence["pit_status"] == "PASS"
    assert loaded.prediction_intelligence["predictability"] == pytest.approx(0.72)


def test_prediction_set_requires_complete_metadata():
    with pytest.raises(ValueError, match="prediction-set metadata"):
        validate_prediction(_record(prediction_set_alpha=0.10))


def test_prediction_set_rejects_mismatched_action():
    with pytest.raises(ValueError, match="does not match"):
        validate_prediction(
            _record(
                prediction_set=["home", "draw"],
                prediction_set_alpha=0.10,
                prediction_set_method="split_conformal",
                prediction_set_action="SINGLE",
            )
        )


def test_prediction_intelligence_is_fail_closed_on_pit_or_time():
    with pytest.raises(ValueError, match="pit_status"):
        validate_prediction(
            _record(
                prediction_intelligence={
                    "prediction_time": "2026-09-30T09:00:00+00:00",
                    "pit_status": "FAIL",
                    "provenance": {"source": "unit-test"},
                }
            )
        )
    with pytest.raises(ValueError, match="prediction_time"):
        validate_prediction(
            _record(
                prediction_intelligence={
                    "prediction_time": "2026-09-30T09:05:00+00:00",
                    "pit_status": "PASS",
                    "provenance": {"source": "unit-test"},
                }
            )
        )


def test_prediction_set_can_explicitly_abstain():
    record = _record(
        prediction_set=[],
        prediction_set_alpha=0.10,
        prediction_set_method="split_conformal",
        prediction_set_action="ABSTAIN",
    )
    assert record.prediction_set == []
    assert record.prediction_set_action == "ABSTAIN"
