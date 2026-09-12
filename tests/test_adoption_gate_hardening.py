from research.adoption_gate import evaluate_locked_holdout


def test_missing_primary_metric_cannot_be_adopted():
    result = evaluate_locked_holdout(
        {"rows": 300, "LogLoss": 0.60, "Brier": 0.20},
        {"rows": 300, "LogLoss": 0.58, "Brier": 0.19, "Accuracy": 0.60},
        validation_windows=2,
        calibration_ok=True,
        no_future_target_data=True,
        reproducible=True,
    )
    assert result["decision"] == "REJECT"
    assert any("primary_metrics_missing_or_nonfinite" in x for x in result["reasons"])


def test_complete_but_missing_score_evidence_cannot_be_adopted():
    result = evaluate_locked_holdout(
        {"rows": 300, "LogLoss": 0.60, "Brier": 0.20, "Accuracy": 0.60},
        {"rows": 300, "LogLoss": 0.58, "Brier": 0.19, "Accuracy": 0.61},
        validation_windows=2,
        calibration_ok=True,
        no_future_target_data=True,
        reproducible=True,
        league="MLB",
    )
    assert result["decision"] == "REJECT"
    assert "score_target_not_evaluated" in result["reasons"]
    assert "hilo_target_not_evaluated" in result["reasons"]
