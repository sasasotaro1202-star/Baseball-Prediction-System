"""Leakage-safe Baseball candidate adoption gate.

Selection and final confirmation are deliberately separated:
Development OOS -> Candidate Lock -> Locked Holdout -> Adoption.
For NPB, Home/Draw/Away is a first-class target and draw behaviour is checked
separately from aggregate multiclass LogLoss/Brier.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class GatePolicy:
    min_oos_rows: int = 200
    min_relative_improvement: float = 0.01
    max_logloss_regression: float = 0.005
    max_brier_regression: float = 0.005
    max_accuracy_regression: float = 0.005
    max_score_mae_regression: float = 0.0
    max_hilo_logloss_regression: float = 0.005
    max_hilo_brier_regression: float = 0.005
    max_hilo_accuracy_regression: float = 0.005
    max_draw_recall_regression: float = 0.0
    max_draw_probability_mae_regression: float = 0.005
    require_score_check: bool = True
    require_hilo_check: bool = True
    require_npb_three_way_check: bool = True
    require_two_validation_windows: bool = True
    require_calibration_check: bool = True
    require_no_future_target_data: bool = True
    require_reproducible_candidate: bool = True


def _finite_metric(mapping: Mapping[str, float], key: str) -> float | None:
    if key not in mapping:
        return None
    try:
        value = float(mapping[key])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def candidate_lock(*, development_metrics: Mapping[str, float], candidate_id: str) -> dict:
    """Lock a candidate chosen on Development OOS without evaluating the holdout."""
    if not candidate_id:
        raise ValueError("candidate_id is required")
    rows = int(development_metrics.get("rows", 0))
    return {
        "candidate_id": candidate_id,
        "stage": "candidate_locked",
        "development_rows": rows,
        "development_metrics": dict(development_metrics),
        "holdout_evaluated": False,
    }


def _improvement(baseline: Mapping[str, float], candidate: Mapping[str, float], key: str, higher_is_better: bool) -> float:
    b = _finite_metric(baseline, key)
    c = _finite_metric(candidate, key)
    if b is None or c is None:
        raise ValueError(f"missing or non-finite metric: {key}")
    return c - b if higher_is_better else b - c


def evaluate_locked_holdout(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    policy: GatePolicy = GatePolicy(),
    validation_windows: int = 0,
    calibration_ok: bool = False,
    no_future_target_data: bool = False,
    reproducible: bool = False,
    baseline_score: Mapping[str, float] | None = None,
    candidate_score: Mapping[str, float] | None = None,
    baseline_hilo: Mapping[str, float] | None = None,
    candidate_hilo: Mapping[str, float] | None = None,
    league: str | None = None,
) -> dict:
    """Compare a locked candidate against unseen holdout data only.

    Missing or non-finite metrics fail closed; defaults are never substituted
    for evidence. This prevents incomplete evaluations from being promoted.
    """
    rows = int(candidate.get("rows", 0))
    reasons: list[str] = []
    if rows < policy.min_oos_rows:
        reasons.append("insufficient_locked_holdout_rows")
    if policy.require_two_validation_windows and validation_windows < 2:
        reasons.append("insufficient_validation_windows")
    if policy.require_calibration_check and not calibration_ok:
        reasons.append("calibration_check_failed")
    if policy.require_no_future_target_data and not no_future_target_data:
        reasons.append("future_target_data_not_excluded")
    if policy.require_reproducible_candidate and not reproducible:
        reasons.append("candidate_not_reproducible")

    required_primary = ("LogLoss", "Brier", "Accuracy")
    missing_primary = [k for k in required_primary if _finite_metric(baseline, k) is None or _finite_metric(candidate, k) is None]
    if missing_primary:
        reasons.append("primary_metrics_missing_or_nonfinite:" + ",".join(missing_primary))
        return {
            "stage": "locked_holdout_evaluated", "adopt": False, "decision": "REJECT",
            "reasons": reasons, "baseline": dict(baseline), "candidate": dict(candidate),
            "targets": {}, "improvement": {},
        }

    ll_improvement = _improvement(baseline, candidate, "LogLoss", False)
    br_improvement = _improvement(baseline, candidate, "Brier", False)
    acc_improvement = _improvement(baseline, candidate, "Accuracy", True)
    ll_base = float(baseline["LogLoss"])
    ll_cand = float(candidate["LogLoss"])
    relative_ll = ll_improvement / max(abs(ll_base), 1e-12)

    if relative_ll < policy.min_relative_improvement:
        reasons.append("logloss_improvement_below_gate")
    if ll_cand - ll_base > policy.max_logloss_regression:
        reasons.append("logloss_regression")
    if float(candidate["Brier"]) - float(baseline["Brier"]) > policy.max_brier_regression:
        reasons.append("brier_regression")
    if float(baseline["Accuracy"]) - float(candidate["Accuracy"]) > policy.max_accuracy_regression:
        reasons.append("accuracy_regression")

    target_results: dict[str, dict] = {}
    if policy.require_score_check:
        if baseline_score is None or candidate_score is None or _finite_metric(baseline_score, "ScoreMAE") is None or _finite_metric(candidate_score, "ScoreMAE") is None:
            reasons.append("score_target_not_evaluated")
        else:
            base_mae = float(baseline_score["ScoreMAE"])
            cand_mae = float(candidate_score["ScoreMAE"])
            score_improvement = base_mae - cand_mae
            if cand_mae - base_mae > policy.max_score_mae_regression:
                reasons.append("score_mae_regression")
            target_results["score"] = {"baseline": dict(baseline_score), "candidate": dict(candidate_score), "ScoreMAE_improvement": score_improvement}

    if policy.require_hilo_check:
        required_hilo = ("LogLoss", "Brier", "Accuracy")
        if baseline_hilo is None or candidate_hilo is None or any(_finite_metric(baseline_hilo, k) is None or _finite_metric(candidate_hilo, k) is None for k in required_hilo):
            reasons.append("hilo_target_not_evaluated")
        else:
            hll_imp = _improvement(baseline_hilo, candidate_hilo, "LogLoss", False)
            hb_imp = _improvement(baseline_hilo, candidate_hilo, "Brier", False)
            ha_imp = _improvement(baseline_hilo, candidate_hilo, "Accuracy", True)
            if float(candidate_hilo["LogLoss"]) - float(baseline_hilo["LogLoss"]) > policy.max_hilo_logloss_regression:
                reasons.append("hilo_logloss_regression")
            if float(candidate_hilo["Brier"]) - float(baseline_hilo["Brier"]) > policy.max_hilo_brier_regression:
                reasons.append("hilo_brier_regression")
            if float(baseline_hilo["Accuracy"]) - float(candidate_hilo["Accuracy"]) > policy.max_hilo_accuracy_regression:
                reasons.append("hilo_accuracy_regression")
            target_results["low_high"] = {"baseline": dict(baseline_hilo), "candidate": dict(candidate_hilo), "LogLoss_improvement": hll_imp, "Brier_improvement": hb_imp, "Accuracy_improvement": ha_imp}

    if policy.require_npb_three_way_check and league == "NPB":
        required = ("DrawRecall", "DrawProbabilityMAE")
        if any(_finite_metric(baseline, key) is None or _finite_metric(candidate, key) is None for key in required):
            reasons.append("npb_three_way_target_not_evaluated")
        else:
            draw_recall_imp = _improvement(baseline, candidate, "DrawRecall", True)
            draw_mae_imp = _improvement(baseline, candidate, "DrawProbabilityMAE", False)
            if float(baseline["DrawRecall"]) - float(candidate["DrawRecall"]) > policy.max_draw_recall_regression:
                reasons.append("draw_recall_regression")
            if float(candidate["DrawProbabilityMAE"]) - float(baseline["DrawProbabilityMAE"]) > policy.max_draw_probability_mae_regression:
                reasons.append("draw_probability_mae_regression")
            target_results["npb_three_way"] = {
                "baseline": {"DrawRecall": float(baseline["DrawRecall"]), "DrawProbabilityMAE": float(baseline["DrawProbabilityMAE"])},
                "candidate": {"DrawRecall": float(candidate["DrawRecall"]), "DrawProbabilityMAE": float(candidate["DrawProbabilityMAE"])},
                "DrawRecall_improvement": draw_recall_imp,
                "DrawProbabilityMAE_improvement": draw_mae_imp,
            }

    return {
        "stage": "locked_holdout_evaluated", "adopt": not reasons,
        "decision": "ADOPT" if not reasons else "REJECT", "reasons": reasons,
        "baseline": dict(baseline), "candidate": dict(candidate), "targets": target_results,
        "improvement": {"LogLoss": ll_improvement, "Brier": br_improvement, "Accuracy": acc_improvement, "relative_LogLoss": relative_ll},
    }
