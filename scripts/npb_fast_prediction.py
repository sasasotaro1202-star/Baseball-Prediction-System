#!/usr/bin/env python3
"""Fast user-facing NPB prediction lane.

The canonical production_npb predictor remains unchanged. This wrapper defers
observation-only enrichment because those snapshots are not consumed by the
current probability path. It therefore changes acquisition latency, not model
inputs, PIT rules, or prediction semantics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

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
    # These collectors are explicitly observation/evidence-only in the current
    # NPB production feature contract. Replace them only for this user-facing
    # fast lane; the feature/model/PIT computation itself remains canonical.
    predictor.collect_npb_player_context = _deferred_player_context
    predictor.collect_npb_roster_context = _deferred_roster_context
    predictor.resolve_roster_player_ids = lambda context, teams: context
    predictor.collect_npb_team_player_context = _deferred_team_context
    predictor.collect_npb_pregame_context = _deferred_pregame_context

    result = predictor.predict(
        args.date,
        args.data_dir,
        research_shadow=True,
        minimum_lead_minutes=float(args.minimum_lead_minutes),
        preferred_lead_minutes=30.0,
    )

    if not isinstance(result, dict):
        raise RuntimeError("NPB fast prediction returned a non-object result")

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
