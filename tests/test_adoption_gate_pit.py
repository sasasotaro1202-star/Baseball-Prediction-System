from research.adoption_gate import GatePolicy, evaluate_locked_holdout


def _metrics():
    return {"rows": 250, "LogLoss": 0.90, "Brier": 0.18, "Accuracy": 0.60}


def test_mlb_holdout_pit_evidence_is_required():
    result = evaluate_locked_holdout(
        _metrics(),
        {"rows": 250, "LogLoss": 0.88, "Brier": 0.175, "Accuracy": 0.61},
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


def test_non_mlb_gate_does_not_require_holdout_pit_by_default():
    result = evaluate_locked_holdout(
        _metrics(),
        {"rows": 250, "LogLoss": 0.88, "Brier": 0.175, "Accuracy": 0.61},
        policy=GatePolicy(
            require_score_check=False,
            require_hilo_check=False,
            require_npb_three_way_check=False,
            require_uncertainty_check=False,
            require_pit_starter_evidence=False,
        ),
        validation_windows=2,
        calibration_ok=True,
        no_future_target_data=True,
        reproducible=True,
        league="NPB",
    )
    assert result["decision"] == "ADOPT"
