import pytest

from research.ultimate_v18_intelligence import (
    InformationCandidate,
    build_decision_object,
    forecast_action,
    forecast_lifetime,
    information_value,
    pit_gate,
    predictability_index,
    scope_candidate,
    operational_utility,
    resource_priority,
    silent_degradation,
    validation_depth,
)


def test_pit_gate_is_fail_closed_for_future_and_unknown():
    assert pit_gate(
        prediction_time="2026-09-28T00:00:00Z", available_at=None
    )["status"] == "FAIL"
    assert (
        pit_gate(
            prediction_time="2026-09-28T00:00:00Z",
            available_at="2026-09-28T00:00:01Z",
        )["reason"]
        == "available_after_prediction"
    )
    assert (
        pit_gate(
            prediction_time="2026-09-28T00:00:00Z",
            available_at="2026-09-27T23:59:59Z",
        )["status"]
        == "PASS"
    )


def test_decision_object_separates_confidence_and_predictability():
    obj = build_decision_object(
        case_id="g1",
        prediction_time="2026-09-28T00:00:00Z",
        valid_until="2026-09-28T01:00:00Z",
        target="win",
        horizon="pregame",
        granularity="game",
        result="HOME",
        probability=[0.7, 0.3],
        distribution=None,
        predictability=0.2,
        confidence=0.9,
        uncertainty=0.7,
        current_state="known",
        future_state="uncertain",
        current_regime="normal",
        future_regime="unknown",
        trajectory=[],
        scenario=[],
        branch=[],
        disagreement=0.8,
        error_correlation=0.2,
        future_failure=0.4,
        time_to_failure_seconds=3600,
        ood=0.3,
        novelty=0.2,
        tail_risk=0.4,
        reversal_risk=0.5,
        update_need=0.6,
        next_update_time="2026-09-28T00:30:00Z",
        information_value_score=0.1,
        model="ensemble",
        strategy="research",
        compute="moderate",
        output="probability",
        action="SCENARIO",
        pit_status="PASS",
        provenance={"source": "unit-test"},
        available_at="2026-09-27T23:59:00Z",
    )
    assert obj["research_only"] is True
    assert obj["confidence_predictability_gap"] == pytest.approx(0.7)
    assert obj["high_confidence_low_predictability"] is True
    assert sum(obj["probability"]) == pytest.approx(1.0)


def test_decision_object_rejects_invalid_probability_and_pit():
    with pytest.raises(ValueError, match="must sum to 1"):
        build_decision_object(
            case_id="bad",
            prediction_time="2026-09-28T00:00:00Z",
            valid_until="2026-09-28T01:00:00Z",
            target="win",
            horizon="pregame",
            granularity="game",
            result="HOME",
            probability=[0.7, 0.7],
            distribution=None,
            predictability=0.5,
            confidence=0.5,
            uncertainty=0.5,
            current_state="x",
            future_state="y",
            current_regime="x",
            future_regime="y",
            trajectory=[],
            scenario=[],
            branch=[],
            disagreement=0.2,
            error_correlation=0.2,
            future_failure=0.2,
            time_to_failure_seconds=100,
            ood=0.1,
            novelty=0.1,
            tail_risk=0.1,
            reversal_risk=0.1,
            update_need=0.1,
            next_update_time="2026-09-28T00:30:00Z",
            information_value_score=0,
            model="m",
            strategy="s",
            compute="c",
            output="o",
            action="MAINTAIN",
            pit_status="PASS",
            provenance={},
            available_at="2026-09-27T23:59:00Z",
        )
    with pytest.raises(ValueError, match="blocked by PIT"):
        build_decision_object(
            case_id="future",
            prediction_time="2026-09-28T00:00:00Z",
            valid_until="2026-09-28T01:00:00Z",
            target="win",
            horizon="pregame",
            granularity="game",
            result="HOME",
            probability=[0.7, 0.3],
            distribution=None,
            predictability=0.5,
            confidence=0.5,
            uncertainty=0.5,
            current_state="x",
            future_state="y",
            current_regime="x",
            future_regime="y",
            trajectory=[],
            scenario=[],
            branch=[],
            disagreement=0.2,
            error_correlation=0.2,
            future_failure=0.2,
            time_to_failure_seconds=100,
            ood=0.1,
            novelty=0.1,
            tail_risk=0.1,
            reversal_risk=0.1,
            update_need=0.1,
            next_update_time="2026-09-28T00:30:00Z",
            information_value_score=0,
            model="m",
            strategy="s",
            compute="c",
            output="o",
            action="MAINTAIN",
            pit_status="PASS",
            provenance={},
            available_at=None,
        )


def test_information_value_is_pit_ranked():
    pt = "2026-09-28T00:00:00Z"
    out = information_value(
        [
            InformationCandidate(
                "low",
                expected_error_reduction=0.05,
                reliability=0.7,
            ),
            InformationCandidate(
                "high",
                expected_error_reduction=0.8,
                reliability=0.95,
                latency_seconds=5,
            ),
            InformationCandidate(
                "future",
                expected_error_reduction=0.99,
                available_at="2026-09-28T00:00:01Z",
            ),
        ],
        prediction_time=pt,
        latency_budget_seconds=30,
    )
    assert out["selected"] == "high"
    assert next(x for x in out["candidates"] if x["name"] == "future")[
        "pit_ok"
    ] is False


def test_action_controller_prioritizes_safety_then_information():
    assert (
        forecast_action(
            freshness=0.8,
            update_need=0.1,
            uncertainty=0.1,
            disagreement=0.1,
            ood=0.1,
            failure_risk=0.1,
            kill_switch=True,
        )["action"]
        == "FALLBACK"
    )
    assert (
        forecast_action(
            freshness=0.9,
            update_need=0.1,
            uncertainty=0.1,
            disagreement=0.1,
            ood=0.1,
            failure_risk=0.1,
            information_value_score=0.2,
        )["action"]
        == "ACQUIRE"
    )
    assert (
        forecast_action(
            freshness=0.9,
            update_need=0.1,
            uncertainty=0.9,
            disagreement=0.8,
            ood=0.1,
            failure_risk=0.1,
        )["action"]
        == "SCENARIO"
    )
    assert (
        forecast_action(
            freshness=0.1,
            update_need=0.8,
            uncertainty=0.2,
            disagreement=0.1,
            ood=0.1,
            failure_risk=0.1,
        )["action"]
        == "RECOMPUTE"
    )


def test_lifetime_decays_over_time():
    early = forecast_lifetime(
        prediction_time="2026-09-28T00:00:00Z",
        valid_until="2026-09-28T01:00:00Z",
        now="2026-09-28T00:10:00Z",
        freshness=1,
        value_decay=0.1,
        revision_probability=0.1,
    )
    late = forecast_lifetime(
        prediction_time="2026-09-28T00:00:00Z",
        valid_until="2026-09-28T01:00:00Z",
        now="2026-09-28T00:50:00Z",
        freshness=1,
        value_decay=0.1,
        revision_probability=0.1,
    )
    assert early["effective_lifetime"] > late["effective_lifetime"]


def test_scope_utility_and_stage_are_explicit():
    out = scope_candidate(
        candidate_id="x",
        target="new_case",
        production_value=0.8,
        learning_value=0.9,
        coverage_value=0.5,
        novelty=0.7,
        pit_risk=0.05,
        data_risk=0.1,
        false_discovery_risk=0.1,
        oos_risk=0.2,
        operational_risk=0.1,
        cost=0.05,
        dependency_risk=0.05,
        stage="PIT_VALIDATED",
        next_test="shadow",
    )
    assert out["stage"] == "PIT_VALIDATED"
    assert out["research_only"] is True


def test_predictability_index_has_seven_dimensions():
    out = predictability_index(
        {
            k: 0.5
            for k in (
                "data",
                "information",
                "temporal",
                "local",
                "global",
                "regime",
                "future",
            )
        }
    )
    assert out["score"] == pytest.approx(0.5)
    assert out["interpretation"] != "model_confidence"


def test_operational_utility_is_safety_dominant():
    safe = operational_utility(
        accuracy=.8, reliability=.8, latency=.8, coverage=.8,
        cost_efficiency=.8, safety=1.0,
    )
    unsafe = operational_utility(
        accuracy=.99, reliability=.99, latency=.99, coverage=.99,
        cost_efficiency=.99, safety=.5,
    )
    assert safe["utility"] > unsafe["utility"]
    assert safe["research_only"] is True


def test_validation_depth_never_drops_pit_or_safety():
    out = validation_depth(
        case_importance=.9, failure_risk=.9, ood=.2, rare_case=.2, operational_impact=.8
    )
    assert out["depth"] == "DEEP"
    assert out["pit_required"] is True
    assert out["safety_required"] is True


def test_resource_priority_surfaces_unknown_and_failure_frontier():
    out = resource_priority(
        production_criticality=.2, information_value=.8, failure_risk=.9,
        unknown_frontier=.9, learning_value=.7, operational_risk=.5,
        cost=.1, dependency_risk=.1,
    )
    assert out["priority"] > 2.0
    assert out["research_only"] is True


def test_silent_degradation_detects_multi_axis_shift():
    out = silent_degradation(
        {
            "data_quality": .95, "latency": .9, "calibration": .9,
            "disagreement": .2, "predictability": .8, "coverage": .9,
        },
        {
            "data_quality": .80, "latency": .7, "calibration": .75,
            "disagreement": .3, "predictability": .6, "coverage": .88,
        },
        lower_is_better=("latency", "disagreement"),
        warn_delta=.08,
    )
    assert out["status"] == "ALERT"
    assert "data_quality" in out["degraded_metrics"]
    assert "calibration" in out["degraded_metrics"]
