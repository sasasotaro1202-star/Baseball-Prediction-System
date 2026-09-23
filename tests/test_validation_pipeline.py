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
        holdout_uncertainty={
            "improvement_ci95": {"LogLoss": [0.005, 0.04]},
            "p_improvement_positive": {"LogLoss": 0.99},
        },
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


def test_pipeline_rejects_missing_uncertainty():
    kw = _kwargs()
    kw["holdout_uncertainty"] = None
    record = run_validation_pipeline(**kw)
    assert record.decision == "REJECT"
    assert "uncertainty_check_missing" in record.locked_holdout["reasons"]


def test_pipeline_forces_mlb_starter_pit_evidence():
    kw = _kwargs()
    kw["candidate_id"] = "mlb-candidate-001"
    kw["holdout_uncertainty"] = {
        "improvement_ci95": {"LogLoss": [0.005, 0.04]},
        "p_improvement_positive": {"LogLoss": 0.99},
    }
    kw["league"] = "MLB"
    kw["pit_starter_evidence_ok"] = False
    kw["holdout_pit_starter_evidence_ok"] = False
    record = run_validation_pipeline(**kw)
    assert record.decision == "REJECT"
    assert "development_starter_pit_evidence_not_verified" in record.locked_holdout["reasons"]
    assert "holdout_starter_pit_evidence_not_verified" in record.locked_holdout["reasons"]


def test_pipeline_accepts_mlb_when_both_starter_pit_evidence_are_verified():
    kw = _kwargs()
    kw["candidate_id"] = "mlb-candidate-pit-verified"
    kw["league"] = "MLB"
    kw["pit_starter_evidence_ok"] = True
    kw["holdout_pit_starter_evidence_ok"] = True
    record = run_validation_pipeline(**kw)
    assert record.decision == "ADOPT"
    assert can_promote(record)
