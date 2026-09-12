import pytest

from research.adoption_gate import evaluate_locked_holdout
from research.candidates import CandidateSpec, lock_candidate


def _base():
    return {
        "rows": 300,
        "Accuracy": 0.55,
        "LogLoss": 0.95,
        "Brier": 0.60,
        "DrawRecall": 0.30,
        "DrawProbabilityMAE": 0.12,
    }


def test_npb_adoption_requires_draw_metrics():
    base = _base()
    cand = dict(base)
    cand["LogLoss"] = 0.90
    result = evaluate_locked_holdout(
        base,
        cand,
        league="NPB",
        validation_windows=2,
        calibration_ok=True,
        no_future_target_data=True,
        reproducible=True,
        baseline_score={"ScoreMAE": 2.0},
        candidate_score={"ScoreMAE": 2.0},
        baseline_hilo={"LogLoss": 0.68, "Brier": 0.24, "Accuracy": 0.65},
        candidate_hilo={"LogLoss": 0.68, "Brier": 0.24, "Accuracy": 0.65},
    )
    assert result["decision"] == "ADOPT"
    assert "npb_three_way" in result["targets"]


def test_npb_adoption_rejects_draw_recall_regression():
    base = _base()
    cand = dict(base)
    cand.update({"LogLoss": 0.90, "DrawRecall": 0.20})
    result = evaluate_locked_holdout(
        base,
        cand,
        league="NPB",
        validation_windows=2,
        calibration_ok=True,
        no_future_target_data=True,
        reproducible=True,
        baseline_score={"ScoreMAE": 2.0},
        candidate_score={"ScoreMAE": 2.0},
        baseline_hilo={"LogLoss": 0.68, "Brier": 0.24, "Accuracy": 0.65},
        candidate_hilo={"LogLoss": 0.68, "Brier": 0.24, "Accuracy": 0.65},
    )
    assert result["decision"] == "REJECT"
    assert "draw_recall_regression" in result["reasons"]


def test_candidate_lock_never_marks_holdout_as_evaluated():
    spec = CandidateSpec(
        candidate_id="cand-test",
        league="NPB",
        objective="win",
        model_version="candidate-model",
        feature_version="features-v1",
        development_metrics={"rows": 300, "LogLoss": 0.9, "Brier": 0.5, "Accuracy": 0.6},
        selection_reason="development-only",
        git_commit="abc",
        dataset_hash="def",
    )
    locked = lock_candidate(spec)
    assert locked["stage"] == "candidate_locked"
    assert locked["holdout_evaluated"] is False
