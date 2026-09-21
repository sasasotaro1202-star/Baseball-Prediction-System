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


def test_holdout_gate_blocks_baseline_candidate_row_mismatch():
    baseline = {"rows": 300, "LogLoss": 0.70, "Brier": 0.50, "Accuracy": 0.55}
    candidate = {"rows": 299, "LogLoss": 0.68, "Brier": 0.49, "Accuracy": 0.56}
    out = evaluate_locked_holdout(
        baseline, candidate,
        validation_windows=2,
        calibration_ok=True,
        no_future_target_data=True,
        reproducible=True,
        baseline_score={"ScoreMAE": 2.0},
        candidate_score={"ScoreMAE": 2.0},
        baseline_hilo={"LogLoss": 0.6, "Brier": 0.4, "Accuracy": 0.6},
        candidate_hilo={"LogLoss": 0.6, "Brier": 0.4, "Accuracy": 0.6},
        league="MLB",
    )
    assert out["decision"] == "REJECT"
    assert "baseline_candidate_row_mismatch" in out["reasons"]


def test_missing_baseline_rows_cannot_bypass_alignment_check():
    result = evaluate_locked_holdout(
        {"LogLoss": 0.70, "Brier": 0.50, "Accuracy": 0.55},
        {"rows": 300, "LogLoss": 0.68, "Brier": 0.49, "Accuracy": 0.56},
        validation_windows=2,
        calibration_ok=True,
        no_future_target_data=True,
        reproducible=True,
        baseline_score={"ScoreMAE": 2.0},
        candidate_score={"ScoreMAE": 2.0},
        baseline_hilo={"LogLoss": 0.6, "Brier": 0.4, "Accuracy": 0.6},
        candidate_hilo={"LogLoss": 0.6, "Brier": 0.4, "Accuracy": 0.6},
        league="MLB",
    )
    assert result["decision"] == "REJECT"
    assert "invalid_metric_row_count" in result["reasons"]


def test_candidate_lock_rejects_malformed_row_count():
    from research.adoption_gate import candidate_lock
    import pytest
    with pytest.raises(ValueError):
        candidate_lock(development_metrics={"rows": "not-a-number"}, candidate_id="x")
