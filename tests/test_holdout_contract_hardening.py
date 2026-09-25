import pytest

from research.adoption_gate import candidate_lock
from research.candidates import CandidateSpec, candidate_fingerprint
from research.validation_pipeline import run_validation_pipeline


def _spec(**overrides):
    base = dict(
        candidate_id="cand-test",
        league="NPB",
        objective="win",
        model_version="IndividuallyCalibratedBlend",
        feature_version="features-v1",
        development_metrics={"rows": 250, "LogLoss": 0.68, "Brier": 0.22, "Accuracy": 0.64},
        selection_reason="Development OOS only",
        git_commit="abc123",
        dataset_hash="dataset-a",
    )
    base.update(overrides)
    return CandidateSpec(**base)


def test_candidate_fingerprint_changes_with_locked_selection_inputs():
    a = _spec()
    b = _spec(development_metrics={"rows": 250, "LogLoss": 0.679, "Brier": 0.22, "Accuracy": 0.64})
    c = _spec(dataset_hash="dataset-b")
    assert candidate_fingerprint(a) != candidate_fingerprint(b)
    assert candidate_fingerprint(a) != candidate_fingerprint(c)


def test_candidate_lock_is_explicitly_pre_holdout():
    payload = candidate_lock(development_metrics={"rows": 250}, candidate_id="cand-1")
    assert payload["stage"] == "candidate_locked"
    assert payload["holdout_evaluated"] is False
    assert payload["holdout_access"] == "forbidden_during_selection"


def test_validation_pipeline_requires_independent_uncertainty_evidence():
    common = dict(
        candidate_id="cand-uncertainty",
        development_metrics={"rows": 250},
        holdout_baseline={"rows": 250, "LogLoss": 0.70, "Brier": 0.25, "Accuracy": 0.60, "DrawRecall": 0.20, "DrawProbabilityMAE": 0.01},
        holdout_candidate={"rows": 250, "LogLoss": 0.68, "Brier": 0.24, "Accuracy": 0.61, "DrawRecall": 0.20, "DrawProbabilityMAE": 0.01},
        validation_windows=2,
        calibration_ok=True,
        no_future_target_data=True,
        reproducible=True,
        league="NPB",
    )
    rejected = run_validation_pipeline(**common)
    assert rejected.decision == "REJECT"
    assert "uncertainty_check_missing" in rejected.locked_holdout["reasons"]

    evidence = {
        "improvement_ci95": {"LogLoss": [0.01, 0.05]},
        "p_improvement_positive": {"LogLoss": 0.99},
    }
    accepted = run_validation_pipeline(**common, holdout_uncertainty=evidence)
    assert accepted.decision == "ADOPT"


def test_validation_pipeline_does_not_allow_missing_primary_metrics_to_promote():
    with pytest.raises(ValueError):
        run_validation_pipeline(
            candidate_id="cand-missing",
            development_metrics={"rows": 250},
            holdout_baseline={"rows": 250, "LogLoss": 0.70, "Brier": 0.25},
            holdout_candidate={"rows": 250, "LogLoss": 0.68, "Brier": 0.24},
            validation_windows=2,
            calibration_ok=True,
            no_future_target_data=True,
            reproducible=True,
            league="NPB",
            holdout_uncertainty={
                "improvement_ci95": {"LogLoss": [0.01, 0.05]},
                "p_improvement_positive": {"LogLoss": 0.99},
            },
        )
