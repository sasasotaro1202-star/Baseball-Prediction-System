import numpy as np
import pandas as pd
import pytest

from research.ultimate_v13_operational_controls import (
    build_forecast_contract,
    error_attribution,
    experiment_record,
    failure_memory,
    fallback_policy,
    output_format,
    prediction_freshness,
    revision_value,
    rollback_target,
    router_stability,
    scenario_forecast,
    strategy_failure_rate,
)


def test_output_formats_and_scenarios():
    assert output_format(predictability=.9, uncertainty=.1, ood=.1) == "SINGLE_PROBABILITY"
    assert output_format(predictability=.1, uncertainty=.9, ood=.1) == "SCENARIO"
    assert output_format(predictability=.9, uncertainty=.1, ood=.95) == "ABSTAIN"
    out = scenario_forecast([.6, .2, .2], scenario_shifts={"base":0.0, "shock":-.1})
    assert out["scenario_count"] == 2
    assert all(abs(x - 1.0) < 1e-9 for x in out["probability_sum_check"])


def test_prediction_freshness_and_revision():
    f = prediction_freshness(
        prediction_time="2026-01-01T00:00:00Z",
        now="2026-01-01T00:10:00Z",
        valid_until="2026-01-01T01:00:00Z",
    )
    assert f["valid"] is True and 0 < f["freshness"] < 1
    v = revision_value([.6,.4], [.7,.3], 0)
    assert v["outcome_matured"] is True
    assert v["revision_better"] is True


def test_error_router_and_fallback_controls():
    e = error_attribution(
        data_quality=.95, label_quality=.98, drift=.1, disagreement=.2,
        predictability=.4, calibration_error=.05, ood=.05,
        router_instability=.1, information_shock=.2,
    )
    assert e["causal_claim"] is False
    r = router_stability([{"A":.6,"B":.4},{"A":.55,"B":.45},{"A":.58,"B":.42}])
    assert 0 <= r["collapse"] <= 1
    assert fallback_policy(health={"router":False})["action"] == "VERIFIED_BASELINE"
    assert fallback_policy(health={"router":True}, kill_switch=True)["reason"] == "KILL_SWITCH"


def test_failure_memory_and_strategy_rate_are_pit_safe():
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    hist = pd.DataFrame({
        "prediction_time":[base, base+pd.Timedelta(hours=1), base+pd.Timedelta(hours=3), base+pd.Timedelta(hours=5)],
        "available_at":[base-pd.Timedelta(minutes=5), base+pd.Timedelta(minutes=30), base+pd.Timedelta(hours=3), base+pd.Timedelta(hours=5)],
        "failure":[True, True, False, True],
        "strategy":["A","A","B","A"],
        "success":[False, True, True, False],
        "x":[0.1,0.2,0.11,0.09],
    })
    q = failure_memory(
        hist, {"x":.1}, prediction_time=base+pd.Timedelta(hours=2), feature_cols=["x"], k=3
    )
    assert q["pit_filtered_rows"] == 2
    rates = strategy_failure_rate(
        hist, prediction_time=base+pd.Timedelta(hours=4)
    )
    assert set(rates) == {"A","B"}

    invalid = hist.copy()
    invalid.loc[0, "available_at"] = "not-a-time"
    blocked = failure_memory(
        invalid, {"x": .1}, prediction_time=base+pd.Timedelta(hours=2), feature_cols=["x"], k=3
    )
    assert blocked["status"] == "BLOCKED"

    bad_strategy = hist.copy()
    bad_strategy.loc[0, "available_at"] = "not-a-time"
    with pytest.raises(ValueError):
        strategy_failure_rate(bad_strategy, prediction_time=base+pd.Timedelta(hours=4))


def test_contract_rollback_and_experiment_registry():
    c = build_forecast_contract(
        prediction_time="2026-01-01T00:00:00Z",
        valid_until="2026-01-01T01:00:00Z",
        model_version="v13",
        strategy="DEEP_COMPUTE",
        output_format_name="PROBABILITY_INTERVAL",
        confidence=.7, predictability=.5, uncertainty=.3, failure_risk=.2,
        ood_score=.1, pit_status="PASS", data_snapshot_id="snap-1",
    )
    assert c["pit_status"] == "PASS" and c["data_snapshot_id"] == "snap-1"
    rb = rollback_target("v13", ["v11","v12","v13"])
    assert rb == {"rollback_allowed":True,"target_version":"v12"}
    record = experiment_record(
        experiment_id="v13-op-001", hypothesis="operational controls are PIT-safe",
        commit="abc", dataset="synthetic", feature_version="f1", model_version="v13",
        parameters={"seed":13}, train_period="2025", validation_period="2025",
        oos_period="2026-synth", metrics={"LogLoss":.68}, decision="HOLD",
    )
    assert record["experiment_id"] == "v13-op-001"
    with pytest.raises(ValueError):
        build_forecast_contract(
            prediction_time="2026-01-01T00:00:00Z",
            valid_until="2026-01-01T01:00:00Z",
            model_version="v13", strategy="x", output_format_name="x",
            confidence=.5, predictability=.5, uncertainty=.5, failure_risk=.5,
            ood_score=.1, pit_status="FAIL", data_snapshot_id="s",
        )
