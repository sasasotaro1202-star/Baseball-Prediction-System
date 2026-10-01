"""Target-aware prediction strategy router.

Different outputs require different forecasting objects; win classification,
total-runs, and exact-score prediction are never forced through one target.
The router only selects a research strategy. It never grants production
eligibility.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TargetStrategy:
    competition_id: str
    stage: str
    target: str
    probability_contract: str
    model_family: str
    update_policy: str
    calibration_policy: str
    status: str = "RESEARCH"


def strategy_for_target(
    competition_id: str,
    target: str,
    *,
    stage: str = "regular_season",
    competition_type: str = "professional",
) -> TargetStrategy:
    cid = str(competition_id or "UNKNOWN").strip().upper()
    t = str(target or "").strip().lower()
    stg = str(stage or "unknown").strip().lower()

    if t == "win_3way":
        if cid != "NPB":
            raise ValueError("win_3way is reserved for competitions with an explicit three-way contract")
        contract = "HOME_DRAW_AWAY"
        family = "classification_ensemble_with_regime_router"
        update = "chronological_retrain_and_regime_reweight"
    elif t == "win_2way":
        if cid != "MLB":
            raise ValueError("win_2way is currently defined for MLB only")
        contract = "HOME_AWAY"
        family = "classification_ensemble_with_regime_router"
        update = "chronological_retrain_and_regime_reweight"
    elif t == "low_high":
        contract = "LOW_TOTAL_LE_6_HIGH_TOTAL_GE_7"
        family = "run_distribution"
        update = "run_model_refresh_then_recompute"
    elif t == "exact_score":
        contract = "EXACT_SCORE_TOP4_FROM_FULL_DISTRIBUTION"
        family = "correlated_run_distribution"
        update = "run_model_refresh_then_recompute"
    else:
        raise ValueError(f"unknown prediction target: {target}")

    if stg in {"postseason", "climax_series", "division_series", "league_championship_series", "wild_card", "japan_series"}:
        update = "postseason_shrunk_retrain_and_recalibrate"
    elif stg in {"tournament", "qualifier", "championship"} or competition_type in {"tournament", "qualifier", "international", "international_club"}:
        update = "competition_specific_shrunk_retrain_and_recalibrate"

    calibration = (
        "target_specific_temperature_when_oos_supported"
        if t in {"win_3way", "win_2way"}
        else "distribution_calibration_when_oos_supported"
    )

    return TargetStrategy(
        competition_id=cid,
        stage=stg,
        target=t,
        probability_contract=contract,
        model_family=family,
        update_policy=update,
        calibration_policy=calibration,
    )


def as_dict(strategy: TargetStrategy) -> dict[str, str]:
    return asdict(strategy)


def standard_target_strategies(
    competition_id: str,
    *,
    stage: str = "regular_season",
    competition_type: str = "professional",
) -> dict[str, TargetStrategy]:
    """Return the canonical target contracts for a competition.

    This helper binds prediction targets to explicit contracts without granting
    production eligibility. Competition-specific validation remains separate.
    """
    targets = (
        ("win_3way",) if str(competition_id or "").strip().upper() == "NPB"
        else ("win_2way",) if str(competition_id or "").strip().upper() == "MLB"
        else ()
    ) + ("low_high", "exact_score")

    return {
        target: strategy_for_target(
            competition_id,
            target,
            stage=stage,
            competition_type=competition_type,
        )
        for target in targets
    }
