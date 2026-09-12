from research.validation_pipeline import run_validation_pipeline, can_promote


def _kwargs():
    return dict(
        candidate_id="candidate-score-hilo-001",
        development_metrics={"LogLoss": 0.61, "rows": 500},
        holdout_baseline={"LogLoss": 0.60, "Brier": 0.20, "Accuracy": 0.70, "rows": 300},
        holdout_candidate={"LogLoss": 0.58, "Brier": 0.19, "Accuracy": 0.705, "rows": 300},
        validation_windows=2,
        calibration_ok=True,
        no_future_target_data=True,
        reproducible=True,
        holdout_score_baseline={"ScoreMAE": 2.0},
        holdout_score_candidate={"ScoreMAE": 1.9},
        holdout_hilo_baseline={"LogLoss": 0.60, "Brier": 0.20, "Accuracy": 0.70},
        holdout_hilo_candidate={"LogLoss": 0.58, "Brier": 0.19, "Accuracy": 0.705},
    )


def test_pipeline_adopts_only_after_locked_holdout():
    record = run_validation_pipeline(**_kwargs())
    assert record.development["stage"] == "candidate_locked"
    assert record.development["holdout_evaluated"] is False
    assert record.stage == "locked_holdout_evaluated"
    assert record.decision == "ADOPT"
    assert can_promote(record)


def test_pipeline_rejects_when_score_not_evaluated():
    kw = _kwargs()
    kw["holdout_score_baseline"] = None
    kw["holdout_score_candidate"] = None
    record = run_validation_pipeline(**kw)
    assert record.decision == "REJECT"
    assert "score_target_not_evaluated" in record.locked_holdout["reasons"]
