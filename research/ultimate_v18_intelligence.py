"""Research-only Predictive Intelligence v18 control primitives.

This module deliberately does not select or mutate the production Champion.
It turns the v18 specification into auditable, deterministic policy objects for
Decision Object construction, predictability, information value, forecast
lifetime/action selection, and scope-frontier ranking.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
import math


SCHEMA_VERSION = "v18-research-1"


def _dt(value: Any, *, name: str) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be ISO-8601") from exc
    if dt.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return dt.astimezone(timezone.utc)


def _unit(value: Any, *, name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(x) or not 0.0 <= x <= 1.0:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return x


def _finite(value: Any, *, name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    return x


def _normalize_probability(value: Any, *, name: str = "probability") -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        keys = list(value.keys())
        vals = [_finite(value[k], name=f"{name}[{k!r}]") for k in keys]
        if any(v < 0.0 for v in vals):
            raise ValueError(f"{name} contains negative probability")
        total = math.fsum(vals)
        if not (0.999999 <= total <= 1.000001):
            raise ValueError(f"{name} must sum to 1 within tolerance; got {total}")
        return {str(k): float(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        vals = [_finite(v, name=f"{name}[{i}]") for i, v in enumerate(value)]
        if any(v < 0.0 for v in vals):
            raise ValueError(f"{name} contains negative probability")
        total = math.fsum(vals)
        if not (0.999999 <= total <= 1.000001):
            raise ValueError(f"{name} must sum to 1 within tolerance; got {total}")
        return [float(v) for v in vals]
    raise ValueError(f"{name} must be a mapping or sequence")


def pit_gate(
    *,
    prediction_time: Any,
    available_at: Any | None = None,
    pit_status: str = "PASS",
) -> dict[str, Any]:
    """Fail closed on unknown/future information."""
    pt = _dt(prediction_time, name="prediction_time")
    if str(pit_status).upper() != "PASS":
        return {
            "status": "FAIL",
            "reason": "pit_status_not_pass",
            "prediction_time": pt.isoformat(),
        }
    if available_at is None:
        return {
            "status": "FAIL",
            "reason": "available_at_unknown",
            "prediction_time": pt.isoformat(),
        }
    at = _dt(available_at, name="available_at")
    if at > pt:
        return {
            "status": "FAIL",
            "reason": "available_after_prediction",
            "prediction_time": pt.isoformat(),
            "available_at": at.isoformat(),
        }
    return {
        "status": "PASS",
        "prediction_time": pt.isoformat(),
        "available_at": at.isoformat(),
    }


def predictability_index(indicators: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate normalized predictability indicators; higher means easier to forecast."""
    required = ("data", "information", "temporal", "local", "global", "regime", "future")
    values = {k: _unit(indicators[k], name=f"indicators.{k}") for k in required}
    score = math.fsum(values.values()) / len(values)
    return {
        "score": score,
        "components": values,
        "interpretation": "object_predictability_not_model_confidence",
    }


def forecast_lifetime(
    *,
    prediction_time: Any,
    valid_until: Any,
    now: Any,
    freshness: Any,
    value_decay: Any,
    revision_probability: Any,
) -> dict[str, Any]:
    pt = _dt(prediction_time, name="prediction_time")
    vu = _dt(valid_until, name="valid_until")
    nt = _dt(now, name="now")
    if vu <= pt:
        raise ValueError("valid_until must be after prediction_time")
    fresh = _unit(freshness, name="freshness")
    decay = _unit(value_decay, name="value_decay")
    revision = _unit(revision_probability, name="revision_probability")
    total = (vu - pt).total_seconds()
    remaining = max(0.0, (vu - nt).total_seconds())
    remaining_fraction = min(1.0, remaining / total)
    effective = fresh * (1.0 - decay) * remaining_fraction * (1.0 - revision)
    return {
        "valid": nt < vu,
        "prediction_time": pt.isoformat(),
        "valid_until": vu.isoformat(),
        "remaining_seconds": remaining,
        "remaining_fraction": remaining_fraction,
        "effective_lifetime": effective,
    }


@dataclass(frozen=True)
class InformationCandidate:
    name: str
    expected_error_reduction: float = 0.0
    expected_uncertainty_reduction: float = 0.0
    expected_decision_improvement: float = 0.0
    freshness: float = 1.0
    reliability: float = 1.0
    latency_seconds: float = 0.0
    cost: float = 0.0
    failure_probability: float = 0.0
    pit_suitable: bool = True
    available_at: str | None = None


def information_value(
    candidates: Sequence[InformationCandidate],
    *,
    prediction_time: Any,
    latency_budget_seconds: float = 30.0,
) -> dict[str, Any]:
    pt = _dt(prediction_time, name="prediction_time")
    scored: list[dict[str, Any]] = []
    for item in candidates:
        gain = (
            _unit(item.expected_error_reduction, name="expected_error_reduction")
            + _unit(item.expected_uncertainty_reduction, name="expected_uncertainty_reduction")
            + _unit(item.expected_decision_improvement, name="expected_decision_improvement")
        ) / 3.0
        freshness = _unit(item.freshness, name="freshness")
        reliability = _unit(item.reliability, name="reliability")
        failure = _unit(item.failure_probability, name="failure_probability")
        latency = max(0.0, _finite(item.latency_seconds, name="latency_seconds"))
        cost = max(0.0, _finite(item.cost, name="cost"))
        pit_ok = bool(item.pit_suitable)
        if item.available_at is not None:
            pit_ok = pit_ok and _dt(item.available_at, name="available_at") <= pt
        latency_factor = (
            0.0
            if latency > latency_budget_seconds
            else max(0.0, 1.0 - latency / max(latency_budget_seconds, 1.0))
        )
        score = gain * freshness * reliability * (1.0 - failure) * latency_factor - cost
        if not pit_ok:
            score = float("-inf")
        scored.append(
            {
                "name": item.name,
                "score": score,
                "pit_ok": pit_ok,
                "gain": gain,
                "latency_seconds": latency,
                "cost": cost,
                "failure_probability": failure,
            }
        )
    ranked = sorted(
        scored,
        key=lambda x: (x["score"], x["pit_ok"], x["name"]),
        reverse=True,
    )
    selected = (
        ranked[0]["name"]
        if ranked and math.isfinite(ranked[0]["score"])
        else None
    )
    return {
        "prediction_time": pt.isoformat(),
        "candidates": ranked,
        "selected": selected,
        "research_only": True,
    }


def forecast_action(
    *,
    freshness: float,
    update_need: float,
    uncertainty: float,
    disagreement: float,
    ood: float,
    failure_risk: float,
    information_value_score: float = 0.0,
    deadline_seconds: float = 3600.0,
    source_health: bool = True,
    router_health: bool = True,
    fallback_ready: bool = True,
    kill_switch: bool = False,
) -> dict[str, Any]:
    vals = {
        "freshness": _unit(freshness, name="freshness"),
        "update_need": _unit(update_need, name="update_need"),
        "uncertainty": _unit(uncertainty, name="uncertainty"),
        "disagreement": _unit(disagreement, name="disagreement"),
        "ood": _unit(ood, name="ood"),
        "failure_risk": _unit(failure_risk, name="failure_risk"),
    }
    iv = _finite(information_value_score, name="information_value_score")
    deadline = max(0.0, _finite(deadline_seconds, name="deadline_seconds"))
    if kill_switch or not source_health or not router_health:
        action = "FALLBACK" if fallback_ready else "ABSTAIN"
        reason = "critical_health_failure_or_kill_switch"
    elif deadline <= 30 and fallback_ready and (
        vals["ood"] >= 0.85 or vals["failure_risk"] >= 0.85
    ):
        action, reason = "FALLBACK", "last_second_safety"
    elif vals["ood"] >= 0.90 or vals["failure_risk"] >= 0.90:
        action, reason = "ABSTAIN", "extreme_risk"
    elif iv > 0.10:
        action, reason = "ACQUIRE", "positive_value_of_information"
    elif vals["uncertainty"] >= 0.75 and vals["disagreement"] >= 0.65:
        action, reason = "SCENARIO", "high_uncertainty_and_disagreement"
    elif vals["uncertainty"] >= 0.80 or vals["ood"] >= 0.70:
        action, reason = "DEEP_RECOMPUTE", "high_uncertainty_or_ood"
    elif vals["update_need"] >= 0.70 or vals["freshness"] <= 0.30:
        action, reason = "RECOMPUTE", "stale_or_update_required"
    elif vals["update_need"] >= 0.40 or vals["disagreement"] >= 0.45:
        action, reason = "REVISE", "moderate_update_need_or_disagreement"
    else:
        action, reason = "MAINTAIN", "stable_case"
    return {
        **vals,
        "information_value_score": iv,
        "deadline_seconds": deadline,
        "action": action,
        "reason": reason,
        "research_only": True,
    }


def scope_candidate(
    *,
    candidate_id: str,
    target: str,
    production_value: float,
    learning_value: float,
    coverage_value: float,
    novelty: float,
    pit_risk: float,
    data_risk: float,
    false_discovery_risk: float,
    oos_risk: float,
    operational_risk: float,
    cost: float,
    dependency_risk: float,
    stage: str = "DISCOVERED",
    rationale: str = "",
    next_test: str = "",
) -> dict[str, Any]:
    scores = {
        k: _unit(v, name=k)
        for k, v in {
            "production_value": production_value,
            "learning_value": learning_value,
            "coverage_value": coverage_value,
            "novelty": novelty,
            "pit_risk": pit_risk,
            "data_risk": data_risk,
            "false_discovery_risk": false_discovery_risk,
            "oos_risk": oos_risk,
            "operational_risk": operational_risk,
            "dependency_risk": dependency_risk,
        }.items()
    }
    cost_v = max(0.0, _finite(cost, name="cost"))
    utility = (
        math.fsum(
            scores[k]
            for k in ("production_value", "learning_value", "coverage_value", "novelty")
        )
        - math.fsum(
            scores[k]
            for k in (
                "pit_risk",
                "data_risk",
                "false_discovery_risk",
                "oos_risk",
                "operational_risk",
                "dependency_risk",
            )
        )
        - cost_v
    )
    ladder = {
        "DISCOVERED": 0,
        "METADATA_CHECKED": 1,
        "DATA_FEASIBLE": 2,
        "PIT_VALIDATED": 3,
        "SHADOW": 4,
        "OOS_ROBUSTNESS_VALIDATED": 5,
        "LIMITED_PRODUCTION": 6,
        "STABLE_PRODUCTION": 7,
        "SCALE_UP": 8,
    }
    if stage not in ladder:
        raise ValueError(f"unknown scope stage: {stage}")
    return {
        "candidate_id": candidate_id,
        "target": target,
        **scores,
        "cost": cost_v,
        "utility": utility,
        "stage": stage,
        "rationale": rationale,
        "next_test": next_test,
        "research_only": True,
    }



def operational_utility(
    *,
    accuracy: float,
    reliability: float,
    latency: float,
    coverage: float,
    cost_efficiency: float,
    safety: float,
) -> dict[str, Any]:
    """Research-only utility surface; safety is a hard multiplier, not a tradeable bonus."""
    values = {
        "accuracy": _unit(accuracy, name="accuracy"),
        "reliability": _unit(reliability, name="reliability"),
        "latency": _unit(latency, name="latency"),
        "coverage": _unit(coverage, name="coverage"),
        "cost_efficiency": _unit(cost_efficiency, name="cost_efficiency"),
        "safety": _unit(safety, name="safety"),
    }
    quality = math.fsum(
        values[k] for k in ("accuracy", "reliability", "latency", "coverage", "cost_efficiency")
    ) / 5.0
    utility = quality * values["safety"]
    return {
        "components": values,
        "quality": quality,
        "utility": utility,
        "safety_dominant": values["safety"] >= 0.99,
        "research_only": True,
    }


def validation_depth(
    *,
    case_importance: float,
    failure_risk: float,
    ood: float,
    rare_case: float,
    operational_impact: float,
) -> dict[str, Any]:
    """Choose validation depth while never allowing PIT/safety gates to be skipped."""
    values = {
        "case_importance": _unit(case_importance, name="case_importance"),
        "failure_risk": _unit(failure_risk, name="failure_risk"),
        "ood": _unit(ood, name="ood"),
        "rare_case": _unit(rare_case, name="rare_case"),
        "operational_impact": _unit(operational_impact, name="operational_impact"),
    }
    pressure = math.fsum(values.values()) / len(values)
    if max(values.values()) >= 0.85 or pressure >= 0.65:
        depth = "DEEP"
    elif pressure >= 0.35:
        depth = "STANDARD"
    else:
        depth = "LIGHT"
    return {
        "depth": depth,
        "pressure": pressure,
        "pit_required": True,
        "safety_required": True,
        "research_only": True,
    }


def resource_priority(
    *,
    production_criticality: float,
    information_value: float,
    failure_risk: float,
    unknown_frontier: float,
    learning_value: float,
    operational_risk: float,
    cost: float,
    dependency_risk: float,
) -> dict[str, Any]:
    """Prioritize scarce research resources without making high risk invisible."""
    positives = {
        "production_criticality": _unit(production_criticality, name="production_criticality"),
        "information_value": _unit(information_value, name="information_value"),
        "failure_risk": _unit(failure_risk, name="failure_risk"),
        "unknown_frontier": _unit(unknown_frontier, name="unknown_frontier"),
        "learning_value": _unit(learning_value, name="learning_value"),
        "operational_risk": _unit(operational_risk, name="operational_risk"),
    }
    cost_v = max(0.0, _finite(cost, name="cost"))
    dependency_v = _unit(dependency_risk, name="dependency_risk")
    priority = (
        math.fsum(positives[k] for k in (
            "production_criticality", "information_value", "failure_risk",
            "unknown_frontier", "learning_value", "operational_risk"
        ))
        - cost_v
        - dependency_v
    )
    return {
        "components": positives,
        "cost": cost_v,
        "dependency_risk": dependency_v,
        "priority": priority,
        "research_only": True,
    }


def silent_degradation(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    lower_is_better: Sequence[str] = (),
    warn_delta: float = 0.08,
    alert_count: int = 2,
) -> dict[str, Any]:
    """Detect multi-axis degradation before headline accuracy necessarily collapses."""
    metrics = ("data_quality", "latency", "calibration", "disagreement", "predictability", "coverage")
    low_better = set(lower_is_better)
    changes: dict[str, float] = {}
    degraded: list[str] = []
    for key in metrics:
        if key not in baseline or key not in current:
            continue
        b = _unit(baseline[key], name=f"baseline.{key}")
        c = _unit(current[key], name=f"current.{key}")
        delta = c - b
        changes[key] = delta
        worse = delta <= -warn_delta if key not in low_better else delta >= warn_delta
        if worse:
            degraded.append(key)
    status = "ALERT" if len(degraded) >= alert_count else ("WARN" if degraded else "STABLE")
    return {
        "status": status,
        "changes": changes,
        "degraded_metrics": degraded,
        "headline_accuracy_not_required": True,
        "research_only": True,
    }

def build_decision_object(
    *,
    case_id: str,
    prediction_time: Any,
    valid_until: Any,
    target: str,
    horizon: str,
    granularity: str,
    result: Any,
    probability: Any | None,
    distribution: Any | None,
    predictability: float,
    confidence: float,
    uncertainty: float,
    current_state: str,
    future_state: str,
    current_regime: str,
    future_regime: str,
    trajectory: Any,
    scenario: Any,
    branch: Any,
    disagreement: float,
    error_correlation: float,
    future_failure: float,
    time_to_failure_seconds: float,
    ood: float,
    novelty: float,
    tail_risk: float,
    reversal_risk: float,
    update_need: float,
    next_update_time: Any,
    information_value_score: float,
    model: str,
    strategy: str,
    compute: str,
    output: str,
    action: str,
    pit_status: str,
    provenance: Mapping[str, Any],
    available_at: Any | None = None,
) -> dict[str, Any]:
    pt = _dt(prediction_time, name="prediction_time")
    vu = _dt(valid_until, name="valid_until")
    if vu <= pt:
        raise ValueError("valid_until must be after prediction_time")
    pit = pit_gate(
        prediction_time=pt,
        available_at=available_at,
        pit_status=pit_status,
    )
    if pit["status"] != "PASS":
        raise ValueError(f"decision object blocked by PIT: {pit['reason']}")
    normalized_probability = (
        _normalize_probability(probability) if probability is not None else None
    )
    normalized_distribution = (
        _normalize_probability(distribution, name="distribution")
        if distribution is not None
        and isinstance(distribution, (Mapping, Sequence))
        and not isinstance(distribution, (str, bytes, bytearray))
        else distribution
    )
    fields = {
        "schema_version": SCHEMA_VERSION,
        "research_only": True,
        "case_id": case_id,
        "prediction_time": pt.isoformat(),
        "valid_until": vu.isoformat(),
        "target": target,
        "horizon": horizon,
        "granularity": granularity,
        "result": result,
        "probability": normalized_probability,
        "distribution": normalized_distribution,
        "predictability": _unit(predictability, name="predictability"),
        "confidence": _unit(confidence, name="confidence"),
        "uncertainty": _unit(uncertainty, name="uncertainty"),
        "current_state": current_state,
        "future_state": future_state,
        "current_regime": current_regime,
        "future_regime": future_regime,
        "trajectory": trajectory,
        "scenario": scenario,
        "branch": branch,
        "disagreement": _unit(disagreement, name="disagreement"),
        "error_correlation": _unit(error_correlation, name="error_correlation"),
        "future_failure": _unit(future_failure, name="future_failure"),
        "time_to_failure_seconds": max(
            0.0, _finite(time_to_failure_seconds, name="time_to_failure_seconds")
        ),
        "ood": _unit(ood, name="ood"),
        "novelty": _unit(novelty, name="novelty"),
        "tail_risk": _unit(tail_risk, name="tail_risk"),
        "forecast_lifetime": {
            "prediction_time": pt.isoformat(),
            "valid_until": vu.isoformat(),
        },
        "freshness": 1.0,
        "reversal_risk": _unit(reversal_risk, name="reversal_risk"),
        "update_need": _unit(update_need, name="update_need"),
        "next_update_time": _dt(
            next_update_time, name="next_update_time"
        ).isoformat(),
        "information_value": _finite(
            information_value_score, name="information_value_score"
        ),
        "model": model,
        "strategy": strategy,
        "compute": compute,
        "output": output,
        "action": action,
        "pit_status": pit_status,
        "provenance": dict(provenance),
        "pit": pit,
    }
    fields["confidence_predictability_gap"] = (
        fields["confidence"] - fields["predictability"]
    )
    fields["high_confidence_low_predictability"] = (
        fields["confidence"] >= 0.8 and fields["predictability"] <= 0.3
    )
    return fields
