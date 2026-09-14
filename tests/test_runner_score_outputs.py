from pathlib import Path

from data.availability import AvailabilityRecord
from prediction.runner import run_prediction


def _availability():
    return AvailabilityRecord(
        event_id="score-test-1",
        league="NPB",
        home_team="読売ジャイアンツ",
        away_team="阪神タイガース",
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


def test_runner_does_not_accept_non_four_score_candidates(tmp_path: Path):
    try:
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
            git_commit="abc124",
            data_snapshot_id="snap1",
            log_path=str(tmp_path / "predictions.jsonl"),
        )
    except ValueError as exc:
        assert "exactly four" in str(exc)
    else:
        raise AssertionError("runner accepted an invalid production score contract")
