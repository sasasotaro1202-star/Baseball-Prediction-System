#!/usr/bin/env python3
"""Fast user-facing NPB prediction lane.

The canonical production_npb predictor remains unchanged. This wrapper defers
observation-only enrichment because those snapshots are not consumed by the
current probability path. It therefore changes acquisition latency, not model
inputs, PIT rules, or prediction semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import production_npb as predictor
from core.atomic_io import atomic_write_json


def _deferred_player_context(names: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "npb-player-context-v1",
        "status": "DEFERRED_FAST_MODE",
        "players_requested": list(names),
        "players_resolved": 0,
        "players": [],
        "historical_oos_consumption": "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
    }


def _deferred_roster_context(target_date: str) -> dict[str, Any]:
    return {
        "schema_version": "npb-roster-context-v1",
        "target_date": target_date,
        "status": "DEFERRED_FAST_MODE",
        "teams": {},
        "transactions": [],
        "player_count": 0,
        "historical_oos_consumption": "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
    }


def _deferred_team_context(
    team_names: list[str],
    season: int,
    **_: Any,
) -> dict[str, Any]:
    return {
        "schema_version": "npb-team-player-context-v1",
        "status": "DEFERRED_FAST_MODE",
        "teams_requested": list(team_names),
        "teams": {},
        "historical_oos_consumption": "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
        "season": int(season),
    }


def _deferred_pregame_context(target_date: str) -> dict[str, Any]:
    return {
        "schema_version": "npb-pregame-context-v1",
        "target_date": target_date,
        "status": "DEFERRED_FAST_MODE",
        "games": [],
        "historical_oos_consumption": "DISABLED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
    }



def _normalise_research_output(result: dict[str, Any], *, fallback_reason: str | None = None) -> dict[str, Any]:
    """Normalize a research fallback to the user-request/router contract."""
    result = dict(result)
    result.setdefault("scope", "RESEARCH_SHADOW")
    result["production_eligibility"] = False
    result["production_modified"] = False
    result["research_only"] = True
    if fallback_reason:
        result["fast_fallback_used"] = True
        result["fast_fallback_reason"] = fallback_reason
    predictions = result.get("predictions", [])
    if not isinstance(predictions, list):
        raise RuntimeError("research fallback returned an invalid predictions list")
    starter_pit = [
        str(p.get("starter_pit_status", "")).strip().upper()
        for p in predictions
        if isinstance(p, dict)
    ]
    result["pit_status"] = (
        "PASS"
        if predictions and all(v == "PASS" for v in starter_pit)
        else "UNVERIFIABLE"
    )
    return result


def _run_daily_research_fallback(
    *,
    target_date: str,
    data_dir: str,
    minimum_lead_minutes: float,
) -> dict[str, Any]:
    """Reuse the starter-uncertainty research lane for on-demand recovery.

    The daily lane already has a PIT-aware implementation that does not require
    starter publication. Its normal scheduler window is 15-180 minutes; on-demand
    requests widen only that selection window, not the underlying model/data rules.
    """
    from research import npb_daily_forecast as daily

    old_min = daily.EARLY_MIN_LEAD
    old_max = daily.EARLY_MAX_LEAD
    try:
        daily.EARLY_MIN_LEAD = float(minimum_lead_minutes)
        daily.EARLY_MAX_LEAD = 24.0 * 60.0
        result = daily.run_forecast(target_date=target_date, data_dir=data_dir)
    finally:
        daily.EARLY_MIN_LEAD = old_min
        daily.EARLY_MAX_LEAD = old_max

    if not isinstance(result, dict):
        raise RuntimeError("daily research fallback returned a non-object result")
    return _normalise_research_output(result)


def _run_direct_runrate_emergency_fallback(
    *,
    target_date: str,
    data_dir: str,
    minimum_lead_minutes: float,
) -> dict[str, Any]:
    """Last-resort starter-agnostic forecast from PIT-safe historical PBP only.

    This lane is strictly RESEARCH_ONLY. It is intended to keep future-game
    prediction available when richer model fitting or starter publication fails.
    """
    import pandas as pd

    from baseball_backtest import BaseballBacktest, low_high_probs, score_candidates
    from prediction.pregame_scheduler import _schedule_for_date
    from production_npb import direct_pit_safe_lambdas
    from research.correlated_score import npb_final_outcomes

    now = datetime.now(timezone.utc)
    scheduled: list[dict[str, str]] = []
    for row in _schedule_for_date(target_date):
        home = str(row.get("home") or "").strip()
        away = str(row.get("away") or "").strip()
        start = str(row.get("official_start_time") or "").strip()
        if not home or not away or not start:
            continue
        try:
            start_dt = datetime.fromisoformat(
                f"{target_date}T{start}:00+09:00"
            ).astimezone(timezone.utc)
        except ValueError:
            continue
        lead = (start_dt - now).total_seconds() / 60.0
        if lead >= float(minimum_lead_minutes):
            scheduled.append(
                {
                    "home": home,
                    "away": away,
                    "official_start_time": start,
                    "scheduled_start_utc": start_dt.isoformat(),
                }
            )
    scheduled.sort(key=lambda x: (x["scheduled_start_utc"], x["home"], x["away"]))

    if not scheduled:
        return {
            "schema_version": "npb-user-emergency-forecast-v1",
            "target_date": target_date,
            "execution_status": "NO_FUTURE_GAMES",
            "scope": "RESEARCH_SHADOW",
            "production_eligibility": False,
            "production_modified": False,
            "pit_status": "PASS",
            "predictions": [],
            "prediction_generated_at": datetime.now(timezone.utc).isoformat(),
            "model_status": "NOT_RUN",
        }

    bt = BaseballBacktest(Path(data_dir))
    raw = bt.load_npb_pbp()
    hist = bt.aggregate_npb_games(raw)
    hist_dt = pd.to_datetime(hist["datetime"], errors="coerce", utc=True)
    if hist_dt.isna().any():
        raise RuntimeError("emergency fallback refused: malformed historical datetime")
    cutoff = datetime.now(timezone.utc)
    hist = hist.loc[hist_dt < pd.Timestamp(cutoff)].copy()
    hist = hist.sort_values(["datetime", "game_id"]).drop_duplicates("game_id")
    if len(hist) < 100:
        raise RuntimeError(
            f"emergency fallback refused: insufficient PIT-safe history ({len(hist)})"
        )

    outputs: list[dict[str, Any]] = []
    for game in scheduled:
        game_key = (
            f'{target_date}-{game["home"]}-{game["away"]}-{game["official_start_time"]}'
        )
        game_id = (
            f"NPB-{target_date}-"
            f"{hashlib.sha256(game_key.encode('utf-8')).hexdigest()[:12]}"
        )
        feature_row = pd.Series(
            {
                "league": "NPB",
                "game_id": game_id,
                "datetime": pd.Timestamp(cutoff),
                "home": game["home"],
                "away": game["away"],
                "home_starter": "",
                "away_starter": "",
            }
        )
        lh, la, shared = direct_pit_safe_lambdas(hist, feature_row, bt)
        if not all(math.isfinite(v) and v >= 0.0 for v in (lh, la, shared)):
            raise RuntimeError("emergency fallback produced non-finite run rates")

        top4 = score_candidates(lh, la, shared, 4)
        low, high = low_high_probs(lh, la, shared)
        home_final, draw_final, away_final = npb_final_outcomes(lh, la, shared)
        start_dt = datetime.fromisoformat(game["scheduled_start_utc"])
        generated_at = datetime.now(timezone.utc)
        lead_minutes = (start_dt - generated_at).total_seconds() / 60.0
        prediction_id = (
            f"{game_id}:RESEARCH_EMERGENCY:"
            f"{hashlib.sha256((game_id + '|' + cutoff.isoformat()).encode()).hexdigest()[:16]}"
        )
        outputs.append(
            {
                "prediction_id": prediction_id,
                "game_id": game_id,
                "datetime_jst": start_dt.astimezone(ZoneInfo("Asia/Tokyo")).isoformat(),
                "home": game["home"],
                "away": game["away"],
                "home_starter": "",
                "away_starter": "",
                "starter_evidence_status": "not_yet_public_or_unverifiable",
                "starter_source": None,
                "starter_evidence_observed_at_utc": None,
                "starter_pit_status": "UNVERIFIABLE",
                "prediction_cutoff_utc": cutoff.isoformat(),
                "prediction_generated_at": generated_at.isoformat(),
                "prediction_source": "RESEARCH_NPB_EMERGENCY_DIRECT_RUNRATE",
                "prediction_schedule": "on_demand_recovery",
                "prediction_target_lead_minutes": round(lead_minutes, 3),
                "home_win_pct": round(float(home_final) * 100.0, 4),
                "draw_pct": round(float(draw_final) * 100.0, 4),
                "away_win_pct": round(float(away_final) * 100.0, 4),
                "low_pct": round(float(low) * 100.0, 4),
                "high_pct": round(float(high) * 100.0, 4),
                "top4_exact_scores": [
                    {"score": score, "prob_pct": round(float(prob) * 100.0, 4)}
                    for score, prob in top4
                ],
                "lambda_home": float(lh),
                "lambda_away": float(la),
                "shared_lambda": float(shared),
                "model": "PIT-safe direct run-rate emergency fallback",
                "research_model_version": "npb-emergency-direct-runrate-v1",
                "research_only": True,
                "production_eligible": False,
                "pit_status": "UNVERIFIABLE",
                "pit_quality": "UNVERIFIABLE_STARTER_INFORMATION",
                "starter_uncertainty": True,
                "predictability_status": "REDUCED_STARTER_INFORMATION",
                "historical_games_used": int(len(hist)),
            }
        )

    return {
        "schema_version": "npb-user-emergency-forecast-v1",
        "target_date": target_date,
        "execution_status": "RESEARCH_SHADOW_EXECUTED",
        "scope": "RESEARCH_SHADOW",
        "production_eligibility": False,
        "production_modified": False,
        "pit_status": "UNVERIFIABLE",
        "model_status": "EMERGENCY_DIRECT_RUNRATE",
        "predictions": outputs,
        "prediction_generated_at": datetime.now(timezone.utc).isoformat(),
        "governance": {
            "auto_promotion": False,
            "current_production_runtime_changed": False,
            "starter_unknown_is_fail_closed_for_production": True,
            "fallback_reason_must_be_retained": True,
        },
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYY-MM-DD, JST")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--research-shadow", action="store_true")
    parser.add_argument("--minimum-lead-minutes", type=float, default=0.0)
    args = parser.parse_args()

    # Fast-shadow model profile: reduce ensemble iteration counts only for this
    # explicitly non-production lane. The canonical production runner and
    # adoption registry are untouched.
    os.environ.setdefault("BASEBALL_FAST_OOS", "1")
    os.environ.setdefault("BASEBALL_HISTGB_MAX_ITER", "140")
    os.environ.setdefault("BASEBALL_RF_ESTIMATORS", "140")
    os.environ.setdefault("BASEBALL_ET_ESTIMATORS", "140")
    os.environ.setdefault("BASEBALL_LGBM_ESTIMATORS", "140")
    os.environ.setdefault("BASEBALL_XGB_ESTIMATORS", "100")
    os.environ.setdefault("BASEBALL_CATBOOST_ITERATIONS", "100")
    # Keep the user-facing shadow portfolio bounded. These models remain
    # challengers only; no production/adoption state is changed by this lane.
    os.environ.setdefault(
        "BASEBALL_FAST_MODEL_POOL_NPB",
        "HistGB,RandomForest,ExtraTrees,HierarchicalDrawResult",
    )
    os.environ.setdefault(
        "BASEBALL_FAST_SCORE_MODEL_POOL_NPB",
        "Poisson,HistPoisson,ExtraTreesReg",
    )
    os.environ.setdefault("BASEBALL_LOGISTIC_MAX_ITER", "600")
    os.environ.setdefault("BASEBALL_SCORE_REGRESSION_MAX_ITER", "300")
    # These collectors are explicitly observation/evidence-only in the current
    # NPB production feature contract. Replace them only for this user-facing
    # fast lane; the feature/model/PIT computation itself remains canonical.
    predictor.collect_npb_player_context = _deferred_player_context
    predictor.collect_npb_roster_context = _deferred_roster_context
    predictor.resolve_roster_player_ids = lambda context, teams: context
    predictor.collect_npb_team_player_context = _deferred_team_context
    predictor.collect_npb_pregame_context = _deferred_pregame_context

    # The canonical predictor emits diagnostic progress to stdout while fitting
    # models and loading PIT-safe history. The user-facing lane reserves stdout
    # for one machine-readable JSON document, so redirect diagnostics to stderr.
    diagnostics = StringIO()
    primary_error: str | None = None
    try:
        with redirect_stdout(diagnostics):
            result = predictor.predict(
                args.date,
                args.data_dir,
                research_shadow=True,
                minimum_lead_minutes=float(args.minimum_lead_minutes),
                preferred_lead_minutes=30.0,
            )
    except Exception as exc:
        primary_error = f"{type(exc).__name__}: {exc}"
        try:
            result = _run_daily_research_fallback(
                target_date=args.date,
                data_dir=args.data_dir,
                minimum_lead_minutes=float(args.minimum_lead_minutes),
            )
            result = _normalise_research_output(result, fallback_reason=primary_error)
        except Exception as fallback_exc:
            result = {
                "execution_status": "GENERATION_FAILED",
                "pit_status": "UNKNOWN",
                "predictions": [],
                "fallback_stage": "DAILY_RESEARCH",
                "fallback_error": f"{type(fallback_exc).__name__}: {fallback_exc}",
            }

    captured = diagnostics.getvalue()
    if captured:
        sys.stderr.write(captured)
        if not captured.endswith("\n"):
            sys.stderr.write("\n")

    if not isinstance(result, dict):
        raise RuntimeError("NPB fast prediction returned a non-object result")

    # Last recovery lane: a deterministic PIT-safe direct run-rate forecast.
    if not result.get("predictions") and str(result.get("execution_status", "")) in {
        "BLOCKED_STARTERS",
        "NO_DUE_GAMES",
        "GENERATION_FAILED",
        "GENERATION_TIMEOUT",
    }:
        try:
            emergency = _run_direct_runrate_emergency_fallback(
                target_date=args.date,
                data_dir=args.data_dir,
                minimum_lead_minutes=float(args.minimum_lead_minutes),
            )
            result = _normalise_research_output(
                emergency,
                fallback_reason=primary_error or str(result.get("execution_status")),
            )
        except Exception as emergency_exc:
            if primary_error:
                raise RuntimeError(
                    "all NPB user-prediction lanes failed: "
                    f"primary={primary_error}; "
                    f"emergency={type(emergency_exc).__name__}: {emergency_exc}"
                ) from emergency_exc
            raise

    result["fast_mode"] = True
    result["model_runtime_profile"] = "FAST_SHADOW"
    result["model_runtime_profile_config"] = {
        "BASEBALL_FAST_OOS": os.getenv("BASEBALL_FAST_OOS"),
        "BASEBALL_HISTGB_MAX_ITER": os.getenv("BASEBALL_HISTGB_MAX_ITER"),
        "BASEBALL_RF_ESTIMATORS": os.getenv("BASEBALL_RF_ESTIMATORS"),
        "BASEBALL_ET_ESTIMATORS": os.getenv("BASEBALL_ET_ESTIMATORS"),
        "BASEBALL_LGBM_ESTIMATORS": os.getenv("BASEBALL_LGBM_ESTIMATORS"),
        "BASEBALL_XGB_ESTIMATORS": os.getenv("BASEBALL_XGB_ESTIMATORS"),
        "BASEBALL_CATBOOST_ITERATIONS": os.getenv("BASEBALL_CATBOOST_ITERATIONS"),
        "BASEBALL_FAST_MODEL_POOL_NPB": os.getenv("BASEBALL_FAST_MODEL_POOL_NPB"),
        "BASEBALL_FAST_SCORE_MODEL_POOL_NPB": os.getenv("BASEBALL_FAST_SCORE_MODEL_POOL_NPB"),
        "promotion_status": "RESEARCH_ONLY"
    }
    result["observation_enrichment_mode"] = "DEFERRED_FAST_MODE"
    result["fast_lane_contract"] = (
        "Canonical production_npb probability/PIT path with observation-only "
        "enrichment deferred; no probability-input substitution."
    )

    output = predictor.ROOT / "results" / f"npb_shadow_{args.date}.json"
    atomic_write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
