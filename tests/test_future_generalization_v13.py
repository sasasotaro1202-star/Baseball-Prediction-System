import numpy as np
import pytest

from research.future_generalization_v13 import (
    FutureFailureEstimator,
    PredictionAction,
    decide_revision,
    error_correlation,
    model_disagreement,
    predictability_score,
    routing_weights,
    run_e2e_case,
    select_prediction_policy,
    validate_pit,
)


def _synthetic():
    rng = np.random.default_rng(11)
    n, k = 240, 3
    y = rng.integers(0, k, size=n)
    models, losses = {}, {}
    for i, noise in enumerate((0.02, 0.04, 0.06)):
        raw = np.eye(k)[y] * 0.62 + 0.19
        p = _normalize(raw + rng.normal(0.0, noise, size=(n, k)))
        models[f"M{i}"] = p
        losses[f"M{i}"] = -np.log(np.clip(p[np.arange(n), y], 1e-12, 1.0))
    return y, models, losses


def _normalize(x):
    x = np.asarray(x, dtype=float)
    return x / x.sum(axis=1, keepdims=True)


def test_disagreement_and_error_correlation():
    y, models, _ = _synthetic()
    d = model_disagreement(models)
    e = error_correlation(y, models)
    assert d["model_count"] == 3
    assert len(d["pairwise"]) == 3
    assert np.all(np.isfinite(d["mean_probability"]))
    assert np.all((d["class_agreement"] >= 0) & (d["class_agreement"] <= 1))
    assert d["mean_pairwise_js"] >= 0
    assert set(e["mean_error_rate"]) == set(models)


def test_predictability_is_bounded_and_separate():
    _, models, _ = _synthetic()
    p = np.mean(np.stack(list(models.values()), axis=1), axis=1)
    out = predictability_score(p[-1:], history_probs=p)
    assert 0 <= out["predictability"] <= 1
    assert 0 <= out["confidence"] <= 1


def test_future_failure_is_finite():
    _, models, losses = _synthetic()
    f = FutureFailureEstimator(horizon=8).fit(losses).predict(losses)
    assert set(f) == set(models)
    for row in f.values():
        assert 0 <= row["failure_risk"] <= 1
        assert 1 <= row["time_to_failure_periods"] <= 32


def test_routing_is_soft():
    weights = routing_weights(
        {"M0": 0.45, "M1": 0.55, "M2": 0.70},
        failure_risk={"M0": 0.1, "M1": 0.2, "M2": 0.6},
        predictability=0.25,
        disagreement=0.5,
    )
    assert set(weights) == {"M0", "M1", "M2"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert min(weights.values()) > 0


def test_policy_branches():
    assert select_prediction_policy(predictability=0.1, ood_score=0.1, failure_risk_max=0.2) == PredictionAction.SCENARIO.value
    assert select_prediction_policy(predictability=0.8, ood_score=0.9, failure_risk_max=0.2) == PredictionAction.ABSTAIN.value
    assert select_prediction_policy(predictability=0.8, ood_score=0.1, failure_risk_max=0.8) == PredictionAction.DEEP_COMPUTE.value


def test_pit_is_fail_closed():
    now = "2026-09-26T12:00:00Z"
    assert validate_pit(prediction_time=now, available_at=[now])["status"] == "PASS"
    assert validate_pit(prediction_time=now, available_at=[None])["status"] == "FAIL"
    assert validate_pit(prediction_time=now, available_at=["2026-09-26T12:00:01Z"])["status"] == "FAIL"
    assert validate_pit(prediction_time=now, available_at=["not-a-timestamp"])["status"] == "FAIL"


def test_revision_hysteresis():
    old = np.array([[0.34, 0.33, 0.33]])
    assert decide_revision(old, old.copy())["action"] == PredictionAction.MAINTAIN.value
    shifted = decide_revision(old, np.array([[0.55, 0.25, 0.20]]))
    assert shifted["action"] == PredictionAction.MAJOR_REVISION.value


def test_e2e_contract_and_ledger():
    y, models, losses = _synthetic()
    result = run_e2e_case(
        models,
        y,
        prediction_time="2026-09-26T12:00:00Z",
        model_loss_history=losses,
        data_quality=0.92,
        drift_score=0.12,
        ood_score=0.10,
        information_candidates=[
            {"name": "starter", "expected_gain": 0.09, "cost": 0.02, "failure_risk": 0.01},
            {"name": "weather", "expected_gain": 0.02, "cost": 0.02, "failure_risk": 0.05},
        ],
    )
    assert result["status"] == "READY"
    assert result["contract"]["pit_status"] == "PASS"
    assert abs(sum(result["routing"].values()) - 1.0) < 1e-9
    assert len(result["ledger"]["probabilities"]) == 3
    assert result["ledger"]["prediction_time"] == result["contract"]["prediction_time"]
    assert set(result["diagnostic_metrics"]) == {"Accuracy", "LogLoss", "Brier", "ECE"}
