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


def test_uncertainty_gate_requires_positive_robust_logloss_improvement():
    from research.adoption_gate import GatePolicy

    result = evaluate_locked_holdout(
        {"rows": 300, "LogLoss": 0.70, "Brier": 0.49, "Accuracy": 0.55},
        {"rows": 300, "LogLoss": 0.675, "Brier": 0.48, "Accuracy": 0.56},
        policy=GatePolicy(
            require_uncertainty_check=True,
            min_positive_improvement_probability=0.95,
        ),
        validation_windows=2,
        calibration_ok=True,
        no_future_target_data=True,
        reproducible=True,
        baseline_score={"ScoreMAE": 2.0},
        candidate_score={"ScoreMAE": 1.9},
        baseline_hilo={"LogLoss": 0.6, "Brier": 0.4, "Accuracy": 0.6},
        candidate_hilo={"LogLoss": 0.59, "Brier": 0.39, "Accuracy": 0.61},
        league="MLB",
        pit_starter_evidence_ok=True,
        holdout_pit_starter_evidence_ok=True,
        holdout_uncertainty={
            "improvement_ci95": {"LogLoss": [-0.01, 0.04]},
            "p_improvement_positive": {"LogLoss": 0.90},
        },
    )
    assert result["decision"] == "REJECT"
    assert "logloss_improvement_uncertainty_ci_failed" in result["reasons"]
    assert "logloss_improvement_probability_failed" in result["reasons"]


def test_uncertainty_gate_accepts_robust_holdout_signal_when_other_gates_pass():
    from research.adoption_gate import GatePolicy

    result = evaluate_locked_holdout(
        {"rows": 300, "LogLoss": 0.70, "Brier": 0.50, "Accuracy": 0.55},
        {"rows": 300, "LogLoss": 0.675, "Brier": 0.49, "Accuracy": 0.56},
        policy=GatePolicy(
            require_uncertainty_check=True,
            min_positive_improvement_probability=0.95,
        ),
        validation_windows=2,
        calibration_ok=True,
        no_future_target_data=True,
        reproducible=True,
        baseline_score={"ScoreMAE": 2.0},
        candidate_score={"ScoreMAE": 1.9},
        baseline_hilo={"LogLoss": 0.6, "Brier": 0.4, "Accuracy": 0.6},
        candidate_hilo={"LogLoss": 0.59, "Brier": 0.39, "Accuracy": 0.61},
        league="MLB",
        pit_starter_evidence_ok=True,
        holdout_pit_starter_evidence_ok=True,
        holdout_uncertainty={
            "improvement_ci95": {"LogLoss": [0.005, 0.04]},
            "p_improvement_positive": {"LogLoss": 0.97},
        },
        evaluation_periods=[
            {"baseline_LogLoss": 0.71, "candidate_LogLoss": 0.69},
            {"baseline_LogLoss": 0.69, "candidate_LogLoss": 0.68},
        ],
    )
    assert result["decision"] == "ADOPT"



def test_default_primary_gate_rejects_sub_three_percent_logloss_gain():
    from research.adoption_gate import GatePolicy

    result = evaluate_locked_holdout(
        {"rows": 300, "LogLoss": 0.60, "Brier": 0.20, "Accuracy": 0.60},
        {"rows": 300, "LogLoss": 0.59, "Brier": 0.19, "Accuracy": 0.61},
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
    )
    assert result["decision"] == "REJECT"
    assert "logloss_improvement_below_gate" in result["reasons"]


def test_default_auxiliary_brier_gate_requires_one_percent_relative_gain():
    from research.adoption_gate import GatePolicy

    result = evaluate_locked_holdout(
        {"rows": 300, "LogLoss": 0.60, "Brier": 0.20, "Accuracy": 0.60},
        {"rows": 300, "LogLoss": 0.58, "Brier": 0.199, "Accuracy": 0.61},
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
    )
    assert result["decision"] == "REJECT"
    assert "brier_improvement_below_gate" in result["reasons"]

def test_adoption_gate_cannot_be_weakened_by_disabling_uncertainty_check():
    from research.adoption_gate import GatePolicy

    result = evaluate_locked_holdout(
        {"rows": 300, "LogLoss": 0.70, "Brier": 0.50, "Accuracy": 0.55},
        {"rows": 300, "LogLoss": 0.67, "Brier": 0.49, "Accuracy": 0.56},
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
        holdout_pit_starter_evidence_ok=True,
        evaluation_periods=[
            {"baseline_LogLoss": 0.70, "candidate_LogLoss": 0.68},
            {"baseline_LogLoss": 0.69, "candidate_LogLoss": 0.68},
        ],
    )
    assert result["decision"] == "REJECT"
    assert "uncertainty_check_missing" in result["reasons"]
