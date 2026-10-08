from research.adoption_gate import GatePolicy, evaluate_locked_holdout


def _metrics():
    return {"rows": 250, "LogLoss": 0.90, "Brier": 0.18, "Accuracy": 0.60, "DrawRecall": 0.60, "DrawProbabilityMAE": 0.10}


def test_mlb_holdout_pit_evidence_is_required():
    result = evaluate_locked_holdout(
        _metrics(),
        {"rows": 250, "LogLoss": 0.87, "Brier": 0.175, "Accuracy": 0.61, "DrawRecall": 0.60, "DrawProbabilityMAE": 0.10},
        policy=GatePolicy(
            require_score_check=False,
            require_hilo_check=False,
            require_npb_three_way_check=False,
            require_uncertainty_check=False,
        ),
        validation_windows=2,
        calibration_ok=True,
        no_future_target_data=True,
        reproducible=True,
        pit_starter_evidence_ok=True,
        holdout_pit_starter_evidence_ok=False,
        league="MLB",
    )
    assert result["decision"] == "REJECT"
    assert "holdout_starter_pit_evidence_not_verified" in result["reasons"]


def test_all_leagues_require_starter_pit_evidence_by_default():
    result = evaluate_locked_holdout(
        _metrics(),
        {"rows": 250, "LogLoss": 0.87, "Brier": 0.175, "Accuracy": 0.61, "DrawRecall": 0.60, "DrawProbabilityMAE": 0.10},
        policy=GatePolicy(
            require_score_check=False,
            require_hilo_check=False,
            require_npb_three_way_check=False,
            require_uncertainty_check=False,
            require_pit_starter_evidence=False,
            require_evaluation_period_stability=False,
        ),
        validation_windows=2,
        calibration_ok=True,
        no_future_target_data=True,
        reproducible=True,
        pit_starter_evidence_ok=False,
        holdout_pit_starter_evidence_ok=False,
        league="NPB",
    )
    assert result["decision"] == "REJECT"
    assert "development_starter_pit_evidence_not_verified" in result["reasons"]
    assert "holdout_starter_pit_evidence_not_verified" in result["reasons"]
