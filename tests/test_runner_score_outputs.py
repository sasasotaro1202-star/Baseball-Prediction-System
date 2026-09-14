from pathlib import Path

import pytest

from data.availability import AvailabilityRecord
from prediction.runner import run_prediction


def _availability(league="NPB"):
    return AvailabilityRecord(
        event_id="score-test-1",
        league=league,
        home_team="Home",
        away_team="Away",
        home_starter="P1",
        away_starter="P2",
        home_starter_announced_at="2020-01-01T00:00:00+00:00",
        away_starter_announced_at="2020-01-01T00:00:00+00:00",
        lineup_status="UNVERIFIABLE",
        lineup_announced_at=None,
        source="test",
        retrieved_at="2020-01-01T00:05:00+00:00",
        prediction_cutoff="2020-01-01T00:05:00+00:00",
    )


def test_runner_generates_game_specific_score_and_low_high(tmp_path: Path):
    result = run_prediction(
        row={"home_run_lambda": 4.2, "away_run_lambda": 3.1},
        availability=_availability(),
        probability_fn=lambda row: {"home": 0.5, "away": 0.2, "draw": 0.3},
        model_version="m1",
        feature_version="f1",
        calibration_version="c1",
        git_commit="abc123",
        data_snapshot_id="snap1",
        log_path=str(tmp_path / "predictions.jsonl"),
    )
    assert result["eligible"] is True
    record = result["prediction"]
    assert len(record.score_candidates) == 4
    assert all(x["score"] != "その他" for x in record.score_candidates)
    assert record.low_probability is not None
    assert record.high_probability is not None
    assert abs(record.low_probability + record.high_probability - 1.0) < 1e-12


def test_runner_rejects_caller_override_of_generated_outputs(tmp_path: Path):
    with pytest.raises(ValueError, match="supplied score_candidates"):
        run_prediction(
            row={
                "home_run_lambda": 4.2,
                "away_run_lambda": 3.1,
                "score_candidates": [
                    {"score": "0-0", "probability": 1.0},
                    {"score": "0-1", "probability": 0.0},
                    {"score": "1-0", "probability": 0.0},
                    {"score": "1-1", "probability": 0.0},
                ],
            },
            availability=_availability(),
            probability_fn=lambda row: {"home": 0.5, "away": 0.2, "draw": 0.3},
            model_version="m1",
            feature_version="f1",
            calibration_version="c1",
            git_commit="abc124",
            data_snapshot_id="snap1",
            log_path=str(tmp_path / "predictions.jsonl"),
        )


def test_runner_requires_four_candidates_when_supplied_without_lambdas(tmp_path: Path):
    with pytest.raises(ValueError, match="exactly four"):
        run_prediction(
            row={
                "score_candidates": [{"score": "1-0", "probability": 0.2}],
                "low_probability": 0.6,
                "high_probability": 0.4,
            },
            availability=_availability(),
            probability_fn=lambda row: {"home": 0.5, "away": 0.2, "draw": 0.3},
            model_version="m1",
            feature_version="f1",
            calibration_version="c1",
            git_commit="abc125",
            data_snapshot_id="snap1",
            log_path=str(tmp_path / "predictions.jsonl"),
        )


def test_runner_keeps_mlb_two_class_win_loss_contract(tmp_path: Path):
    result = run_prediction(
        row={"home_run_lambda": 3.0, "away_run_lambda": 2.5},
        availability=_availability("MLB"),
        probability_fn=lambda row: {"home": 0.57, "away": 0.43},
        model_version="m1",
        feature_version="f1",
        calibration_version="c1",
        git_commit="abc126",
        data_snapshot_id="snap1",
        log_path=str(tmp_path / "mlb.jsonl"),
    )
    assert result["eligible"] is True
    assert set(result["prediction"].probabilities) == {"home", "away"}
    assert len(result["prediction"].score_candidates) == 4
    assert result["prediction"].low_probability is not None
    assert result["prediction"].high_probability is not None
