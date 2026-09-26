"""Research-only operational controls for Ultimate v13.

These controls are deterministic policy/ledger utilities. They do not mutate
production state and do not treat synthetic evidence as deployment evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _norm(p: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(p, dtype=float).reshape(-1)
    if len(arr) < 2 or not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    s = float(arr.sum())
    if s <= 0:
        raise ValueError("probability mass must be positive")
    return arr / s


def output_format(*, predictability: float, uncertainty: float, ood: float, tail_risk: float = 0.0) -> str:
    p = float(predictability)
    u = float(uncertainty)
    o = float(ood)
    t = float(tail_risk)
    if o >= 0.90 or t >= 0.92:
        return "ABSTAIN"
    if p >= 0.72 and u <= 0.32 and o <= 0.35:
        return "SINGLE_PROBABILITY"
    if p >= 0.48 and u <= 0.58:
        return "PROBABILITY_INTERVAL"
    if p >= 0.25:
        return "PREDICTION_SET"
    return "SCENARIO"


def scenario_forecast(
    base_probabilities: Sequence[float],
    *,
    scenario_shifts: Mapping[str, float],
) -> dict[str, Any]:
    base = _norm(base_probabilities)
    rows = []
    for name, shift in scenario_shifts.items():
        q = base.copy()
        s = float(shift)
        if len(q) >= 2:
            q[0] += s
            q[-1] -= s
        q = np.clip(q, 1e-6, None)
        q /= q.sum()
        rows.append({"scenario": str(name), "weight": 1.0, "probabilities": q.tolist()})
    if not rows:
        rows = [{"scenario": "baseline", "weight": 1.0, "probabilities": base.tolist()}]
    total = float(len(rows))
    for row in rows:
        row["weight"] = 1.0 / total
    return {
        "scenarios": rows,
        "scenario_count": len(rows),
        "probability_sum_check": [float(sum(r["probabilities"])) for r in rows],
    }


def prediction_freshness(*, prediction_time: Any, now: Any, valid_until: Any) -> dict[str, Any]:
    pt, nt, vu = _utc(prediction_time), _utc(now), _utc(valid_until)
    age = max(0.0, (nt - pt).total_seconds())
    lifetime = max((vu - pt).total_seconds(), 1.0)
    freshness = float(np.clip(1.0 - age / lifetime, 0.0, 1.0))
    return {"age_seconds": age, "valid": bool(nt <= vu), "freshness": freshness}


def revision_value(previous: Sequence[float], revised: Sequence[float], outcome: int) -> dict[str, Any]:
    a, b = _norm(previous), _norm(revised)
    if len(a) != len(b):
        raise ValueError("revision probability shape mismatch")
    outcome = int(outcome)
    if not 0 <= outcome < len(a):
        raise ValueError("outcome index out of range")
    old_loss = -float(np.log(max(a[outcome], 1e-12)))
    new_loss = -float(np.log(max(b[outcome], 1e-12)))
    return {
        "outcome_matured": True,
        "old_logloss": old_loss,
        "new_logloss": new_loss,
        "logloss_improvement": old_loss - new_loss,
        "revision_better": bool(new_loss < old_loss),
        "revision_probability_shift": float(0.5 * np.abs(a - b).sum()),
    }


def error_attribution(
    *,
    data_quality: float,
    label_quality: float,
    drift: float,
    disagreement: float,
    predictability: float,
    calibration_error: float,
    ood: float,
    router_instability: float,
    information_shock: float,
) -> dict[str, Any]:
    signals = {
        "Data": 1.0 - float(data_quality),
        "Label": 1.0 - float(label_quality),
        "Drift": float(drift),
        "Disagreement": float(disagreement),
        "Predictability": 1.0 - float(predictability),
        "Calibration": float(calibration_error),
        "OOD": float(ood),
        "Router": float(router_instability),
        "InformationShock": float(information_shock),
    }
    clipped = {k: float(np.clip(v, 0.0, 1.0)) for k, v in signals.items()}
    primary = max(clipped.items(), key=lambda x: (x[1], x[0]))[0]
    return {
        "primary": primary,
        "scores": clipped,
        "causal_claim": False,
        "note": "ranked diagnostic signals for post-outcome investigation",
    }


def router_stability(history: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if len(history) < 2:
        return {
            "mean_weight_velocity": 0.0,
            "max_weight_velocity": 0.0,
            "concentration": 0.0,
            "collapse": 0.0,
        }
    names = sorted(set().union(*(row.keys() for row in history)))
    mat = np.array([[float(row.get(name, 0.0)) for name in names] for row in history], dtype=float)
    mat = np.clip(mat, 0.0, None)
    sums = mat.sum(axis=1, keepdims=True)
    mat = mat / np.maximum(sums, 1e-12)
    diffs = 0.5 * np.abs(mat[1:] - mat[:-1]).sum(axis=1)
    concentration = float(np.mean(np.max(mat, axis=1)))
    return {
        "mean_weight_velocity": float(np.mean(diffs)),
        "max_weight_velocity": float(np.max(diffs)),
        "concentration": concentration,
        "collapse": float(np.clip((concentration - 0.85) / 0.15, 0.0, 1.0)),
    }


def failure_memory(
    history: pd.DataFrame,
    query: Mapping[str, float],
    *,
    prediction_time: Any,
    feature_cols: Sequence[str],
    k: int = 5,
) -> dict[str, Any]:
    if history.empty:
        return {"status": "BLOCKED", "reason": "empty_history", "neighbors": []}
    required = {"prediction_time", "available_at", "failure"}
    if not required.issubset(history.columns):
        return {"status": "BLOCKED", "reason": "missing_pit_or_failure_columns", "neighbors": []}
    pt = _utc(prediction_time)
    work = history.copy()
    work["_pt"] = pd.to_datetime(work["prediction_time"], errors="coerce", utc=True)
    work["_at"] = pd.to_datetime(work["available_at"], errors="coerce", utc=True)
    work["_failure"] = work["failure"].astype(bool)
    if work["_pt"].isna().any() or work["_at"].isna().any():
        return {"status": "BLOCKED", "reason": "invalid_pit_timestamp", "neighbors": []}
    work = work[
        (work["_pt"] < pt)
        & (work["_at"] <= work["_pt"])
        & (work["_at"] <= pt)
        & work["_failure"]
    ]
    rows = []
    for idx, row in work.iterrows():
        vals = []
        for col in feature_cols:
            if col in row and col in query and pd.notna(row[col]) and pd.notna(query[col]):
                vals.append((float(row[col]) - float(query[col])) ** 2)
        if vals:
            rows.append((float(np.sqrt(np.mean(vals))), idx))
    rows.sort(key=lambda x: (x[0], str(x[1])))
    return {
        "status": "PASS",
        "pit_filtered_rows": int(len(work)),
        "neighbors": [{"index": str(i), "distance": d} for d, i in rows[: max(1, int(k))]],
    }


def strategy_failure_rate(
    history: pd.DataFrame,
    *,
    prediction_time: Any,
    strategy_col: str = "strategy",
    success_col: str = "success",
) -> dict[str, float]:
    required = {"prediction_time", "available_at", strategy_col, success_col}
    if not required.issubset(history.columns):
        raise ValueError("strategy history missing PIT columns or outcomes")
    pt = _utc(prediction_time)
    work = history.copy()
    work["_pt"] = pd.to_datetime(work["prediction_time"], errors="coerce", utc=True)
    work["_at"] = pd.to_datetime(work["available_at"], errors="coerce", utc=True)
    if work["_pt"].isna().any() or work["_at"].isna().any():
        raise ValueError("strategy history contains invalid PIT timestamps")
    work = work[
        (work["_pt"] < pt)
        & (work["_at"] <= work["_pt"])
        & (work["_at"] <= pt)
    ]
    out = {}
    for strategy, group in work.groupby(strategy_col, dropna=False):
        n = len(group)
        failures = int((~group[success_col].astype(bool)).sum())
        out[str(strategy)] = float((failures + 0.5) / (n + 1.0))
    return out


@dataclass(frozen=True)
class ForecastContract:
    prediction_time: str
    valid_until: str
    model_version: str
    strategy: str
    output_format: str
    confidence: float
    predictability: float
    uncertainty: float
    failure_risk: float
    ood_score: float
    pit_status: str
    data_snapshot_id: str


def build_forecast_contract(
    *,
    prediction_time: Any,
    valid_until: Any,
    model_version: str,
    strategy: str,
    output_format_name: str,
    confidence: float,
    predictability: float,
    uncertainty: float,
    failure_risk: float,
    ood_score: float,
    pit_status: str,
    data_snapshot_id: str,
) -> dict[str, Any]:
    contract = ForecastContract(
        prediction_time=_utc(prediction_time).isoformat(),
        valid_until=_utc(valid_until).isoformat(),
        model_version=str(model_version),
        strategy=str(strategy),
        output_format=str(output_format_name),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        predictability=float(np.clip(predictability, 0.0, 1.0)),
        uncertainty=float(np.clip(uncertainty, 0.0, 1.0)),
        failure_risk=float(np.clip(failure_risk, 0.0, 1.0)),
        ood_score=float(np.clip(ood_score, 0.0, 1.0)),
        pit_status=str(pit_status),
        data_snapshot_id=str(data_snapshot_id),
    )
    if _utc(contract.valid_until) <= _utc(contract.prediction_time):
        raise ValueError("valid_until must be after prediction_time")
    if contract.pit_status != "PASS":
        raise ValueError("forecast contract requires PIT PASS")
    return asdict(contract)


def fallback_policy(*, health: Mapping[str, bool], kill_switch: bool = False) -> dict[str, Any]:
    if kill_switch:
        return {"action": "VERIFIED_BASELINE", "reason": "KILL_SWITCH"}
    if not all(bool(v) for v in health.values()):
        return {"action": "VERIFIED_BASELINE", "reason": "HEALTH_CHECK_FAILURE"}
    return {"action": "FULL_SYSTEM", "reason": "HEALTHY"}


def rollback_target(current_version: str, previous_verified_versions: Sequence[str]) -> dict[str, Any]:
    for version in reversed(list(previous_verified_versions)):
        if version and version != current_version:
            return {"rollback_allowed": True, "target_version": str(version)}
    return {"rollback_allowed": False, "target_version": None}


def experiment_record(
    *,
    experiment_id: str,
    hypothesis: str,
    commit: str,
    dataset: str,
    feature_version: str,
    model_version: str,
    parameters: Mapping[str, Any],
    train_period: str,
    validation_period: str,
    oos_period: str,
    metrics: Mapping[str, float],
    decision: str,
) -> dict[str, Any]:
    return {
        "experiment_id": str(experiment_id),
        "hypothesis": str(hypothesis),
        "commit": str(commit),
        "dataset": str(dataset),
        "feature_version": str(feature_version),
        "model_version": str(model_version),
        "parameters": dict(parameters),
        "train_period": str(train_period),
        "validation_period": str(validation_period),
        "oos_period": str(oos_period),
        "metrics": {str(k): float(v) for k, v in metrics.items()},
        "decision": str(decision),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
