from research.adoption_gate import GatePolicy, evaluate_locked_holdout


def _passing_metrics():
    return (
        {"rows": 400, "LogLoss": 0.80, "Brier": 0.45, "Accuracy": 0.55,
         "DrawRecall": 0.20, "DrawProbabilityMAE": 0.18},
        {"rows": 400, "LogLoss": 0.78, "Brier": 0.44, "Accuracy": 0.56,
         "DrawRecall": 0.20, "DrawProbabilityMAE": 0.18},
    )


def test_starter_pit_gate_is_optional_for_non_starter_candidate():
    baseline, candidate = _passing_metrics()
    result = evaluate_locked_holdout(
        baseline, candidate,
        policy=GatePolicy(require_pit_starter_evidence=False),
        validation_windows=2,
        calibration_ok=True,
        no_future_target_data=True,
        reproducible=True,
        pit_starter_evidence_ok=False,
        baseline_score={"ScoreMAE": 1.5},
        candidate_score={"ScoreMAE": 1.5},
        baseline_hilo={"LogLoss": 0.60, "Brier": 0.24, "Accuracy": 0.60},
        candidate_hilo={"LogLoss": 0.60, "Brier": 0.24, "Accuracy": 0.60},
        league="NPB",
    )
    assert "starter_pit_evidence_not_verified" not in result["reasons"]


def test_starter_pit_gate_still_blocks_when_required():
    baseline, candidate = _passing_metrics()
    result = evaluate_locked_holdout(
        baseline, candidate,
        policy=GatePolicy(require_pit_starter_evidence=True),
        validation_windows=2,
        calibration_ok=True,
        no_future_target_data=True,
        reproducible=True,
        pit_starter_evidence_ok=False,
        baseline_score={"ScoreMAE": 1.5},
        candidate_score={"ScoreMAE": 1.5},
        baseline_hilo={"LogLoss": 0.60, "Brier": 0.24, "Accuracy": 0.60},
        candidate_hilo={"LogLoss": 0.60, "Brier": 0.24, "Accuracy": 0.60},
        league="NPB",
    )
    assert "starter_pit_evidence_not_verified" in result["reasons"]
