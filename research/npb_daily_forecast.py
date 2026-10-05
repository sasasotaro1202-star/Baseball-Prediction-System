"""Automatic NPB daily research forecast with starter-uncertainty handling.

This lane exists to guarantee that a future NPB game has a reproducible research
forecast even when official starter publication is not yet available. It never
changes the checked-in production runtime and never upgrades an UNVERIFIABLE
starter snapshot into production-quality OOS evidence.

Probe mode is stdlib-only so scheduled runs can skip dependency installation when
no NPB game is 15-180 minutes from first pitch. Full forecast mode lazily imports
the numerical stack and reuses the repository's canonical historical feature/model
contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
JST = ZoneInfo("Asia/Tokyo")

EARLY_MIN_LEAD = 15.0
EARLY_MAX_LEAD = 180.0
SOURCE = "RESEARCH_DAILY_AUTO_15_180M"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _target_date(value: str | None) -> str:
    if value:
        return str(value)
    return _now_utc().astimezone(JST).date().isoformat()


def _stable_prediction_id(game_id: str, cutoff_utc: str) -> str:
    raw = f"{game_id}|{cutoff_utc}|{SOURCE}".encode("utf-8")
    return f"{game_id}:{SOURCE}:{hashlib.sha256(raw).hexdigest()[:16]}"


def _official_games(target_date: str) -> list[dict[str, str]]:
    from prediction.pregame_scheduler import _schedule_for_date

    rows = _schedule_for_date(target_date)
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        home = str(row.get("home") or "").strip()
        away = str(row.get("away") or "").strip()
        start = str(row.get("official_start_time") or "").strip()
        if not home or not away or not start:
            continue
        key = (home, away, start)
        if key in seen:
            continue
        seen.add(key)
        out.append({"home": home, "away": away, "official_start_time": start})
    out.sort(key=lambda x: (x["official_start_time"], x["home"], x["away"]))
    if not out:
        raise RuntimeError(
            f"official NPB daily schedule yielded no deterministic future slate: {target_date}"
        )
    return out


def probe(target_date: str) -> dict[str, Any]:
    now = _now_utc()
    games = _official_games(target_date)
    due = []
    for index, game in enumerate(games, start=1):
        start = datetime.fromisoformat(
            f'{target_date}T{game["official_start_time"]}:00+09:00'
        ).astimezone(timezone.utc)
        lead = (start - now).total_seconds() / 60.0
        if EARLY_MIN_LEAD < lead <= EARLY_MAX_LEAD:
            game_id_raw = f'NPB-{target_date}-{game["home"]}-{game["away"]}-{game["official_start_time"]}'
            game_id = f'NPB-{target_date}-{hashlib.sha256(game_id_raw.encode("utf-8")).hexdigest()[:12]}'
            due.append({
                **game,
                "game_id": game_id,
                "game_index": index,
                "lead_minutes": round(lead, 3),
                "scheduled_start_utc": start.isoformat(),
            })
    return {
        "schema_version": "npb-daily-forecast-probe-v1",
        "target_date": target_date,
        "checked_at_utc": now.isoformat(),
        "window_min_lead_minutes": EARLY_MIN_LEAD,
        "window_max_lead_minutes": EARLY_MAX_LEAD,
        "source": SOURCE,
        "games_seen": len(games),
        "due_games": due,
        "due": bool(due),
        "production_modified": False,
    }


def _simulation_summary(
    lam_home: float,
    lam_away: float,
    *,
    simulations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Research-only game-script Monte Carlo; never changes endpoint probabilities."""
    import numpy as np

    sims = max(1000, int(simulations))
    rng = np.random.default_rng(int(seed))
    per_half_h = max(float(lam_home) / 9.0, 1e-6)
    per_half_a = max(float(lam_away) / 9.0, 1e-6)
    h_innings = rng.poisson(per_half_h, size=(sims, 12))
    a_innings = rng.poisson(per_half_a, size=(sims, 12))
    h9 = h_innings[:, :9].sum(axis=1)
    a9 = a_innings[:, :9].sum(axis=1)
    h6 = h_innings[:, :6].sum(axis=1)
    a6 = a_innings[:, :6].sum(axis=1)
    h3 = h_innings[:, :3].sum(axis=1)
    a3 = a_innings[:, :3].sum(axis=1)

    h_final = h9.copy()
    a_final = a9.copy()
    for i in range(9, 12):
        tied = h_final == a_final
        if not tied.any():
            break
        h_final[tied] += h_innings[tied, i]
        a_final[tied] += a_innings[tied, i]
    tied_12 = h_final == a_final
    return {
        "home_lead_after_3_pct": float(np.mean(h3 > a3) * 100.0),
        "away_lead_after_3_pct": float(np.mean(h3 < a3) * 100.0),
        "tie_after_3_pct": float(np.mean(h3 == a3) * 100.0),
        "home_lead_after_6_pct": float(np.mean(h6 > a6) * 100.0),
        "away_lead_after_6_pct": float(np.mean(h6 < a6) * 100.0),
        "tie_after_6_pct": float(np.mean(h6 == a6) * 100.0),
        "regulation_tie_pct": float(np.mean(h9 == a9) * 100.0),
        "home_final_mc_pct": float(np.mean((h_final > a_final) & ~tied_12) * 100.0),
        "draw_final_mc_pct": float(np.mean(tied_12) * 100.0),
        "away_final_mc_pct": float(np.mean((h_final < a_final) & ~tied_12) * 100.0),
        "simulations": float(sims),
        "simulation_seed": float(seed),
    }


def _starter_evidence(target_date: str) -> dict[tuple[str, str], dict[str, Any]]:
    from production_npb import official_starters

    try:
        rows = official_starters(target_date)
    except Exception as exc:
        return {
            "status": "UNVERIFIABLE",
            "error": f"{type(exc).__name__}: {exc}",
            "rows": {},
        }
    mapped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        home = str(row.get("home") or "").strip()
        away = str(row.get("away") or "").strip()
        if not home or not away:
            continue
        mapped[(home, away)] = dict(row)
    return {"status": "AVAILABLE", "error": None, "rows": mapped}


def _competition_metadata(target_date: str) -> dict[str, Any]:
    from production_npb import _official_daily_start_times

    meta: dict[str, Any] = {}
    try:
        _official_daily_start_times(target_date, metadata_out=meta)
    except Exception as exc:
        meta["status"] = "unknown"
        meta["error"] = f"{type(exc).__name__}: {exc}"
    return meta


def run_forecast(
    *,
    target_date: str,
    data_dir: str | Path = "data",
    simulations: int = 10000,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    from baseball_backtest import (
        BaseballBacktest,
        low_high_probs,
        norm_team,
        score_candidates,
    )
    from data.competition_registry import production_eligible
    from research.competition_taxonomy import classify_npb
    from research.correlated_score import npb_final_outcomes
    from research.feature_set_variants import select_feature_set
    from production_npb import (
        _feature_schema_metadata,
        direct_pit_safe_lambdas,
        robust_target_lambdas,
    )

    now = _now_utc()
    cutoff_utc = now.isoformat()
    games = probe(target_date)["due_games"]
    if not games:
        return {
            "schema_version": "npb-daily-research-forecast-v1",
            "target_date": target_date,
            "execution_status": "NO_DUE_GAMES",
            "scope": "RESEARCH_SHADOW",
            "production_modified": False,
            "production_runtime_callable": bool(production_eligible("NPB")),
            "predictions": [],
            "prediction_generated_at": _now_utc().isoformat(),
        }

    starter_info = _starter_evidence(target_date)
    metadata = _competition_metadata(target_date)
    competition = str(metadata.get("competition") or "npb_unknown")
    competition_stage = str(metadata.get("stage") or "unknown")
    season_type = str(metadata.get("season_type") or "unknown")
    game_class = str(metadata.get("game_class") or "unknown")
    competition_key = str(metadata.get("competition_key") or "NPB:npb_unknown:unknown")
    classification_status = str(metadata.get("status") or "unknown")
    if classification_status == "unknown":
        label = classify_npb("公式戦")
        if label.status == "classified":
            competition = label.competition
            competition_stage = label.stage
            season_type = label.season_type
            game_class = label.game_class
            competition_key = label.competition_key
            classification_status = label.status

    bt = BaseballBacktest(Path(data_dir))
    raw = bt.load_npb_pbp()
    hist = bt.aggregate_npb_games(raw)
    hist_dt = pd.to_datetime(hist["datetime"], errors="coerce", utc=True)
    if hist_dt.isna().any():
        raise RuntimeError("Daily research forecast refused: malformed historical datetime.")
    hist = hist.loc[hist_dt < pd.Timestamp(cutoff_utc)].copy()
    hist = hist.sort_values(["datetime", "game_id"]).drop_duplicates("game_id")
    if len(hist) < 100:
        raise RuntimeError(f"Insufficient PIT-safe NPB history: {len(hist)} games.")

    label_total = (
        pd.to_numeric(hist["home_score"], errors="coerce")
        + pd.to_numeric(hist["away_score"], errors="coerce")
    )
    label_total = label_total.replace([np.inf, -np.inf], np.nan).dropna()
    if len(label_total) < 100 or label_total.nunique() < 5:
        raise RuntimeError("Daily research forecast refused: historical score label quality is insufficient.")

    dataset_hash = hashlib.sha256(
        pd.util.hash_pandas_object(hist, index=True).values.tobytes()
    ).hexdigest()

    X, y, _meta = bt.build_features(hist)
    if len(X) != len(hist) or len(y) != len(hist):
        raise RuntimeError("Daily research feature/label contract mismatch.")

    # Fast chronological fit is used only to produce a current research forecast.
    # The adoption gate remains independent and no production eligibility is changed.
    fitted, validation_scores, _ = bt.fit_ensemble(X, y, "NPB", fast_oos=True)
    if not fitted:
        raise RuntimeError("Daily research ensemble fitting failed.")
    score_fit = bt.fit_score_ensemble(
        X,
        hist["home_score"].astype(float).to_numpy(),
        hist["away_score"].astype(float).to_numpy(),
        "NPB",
    )

    schema_meta = None
    outputs: list[dict[str, Any]] = []
    for game_index, game in enumerate(games, start=1):
        home = game["home"]
        away = game["away"]
        start_dt = datetime.fromisoformat(game["scheduled_start_utc"]).astimezone(timezone.utc)
        pair_evidence = starter_info.get("rows", {}).get((home, away), {})
        starter_known = bool(
            pair_evidence.get("home_starter")
            and pair_evidence.get("away_starter")
            and pair_evidence.get("starter_evidence_status") == "official_announced"
        )
        home_starter = str(pair_evidence.get("home_starter") or "")
        away_starter = str(pair_evidence.get("away_starter") or "")
        if starter_known:
            starter_status = "official_announced"
            starter_source = str(pair_evidence.get("starter_source") or "")
            starter_observed_at = cutoff_utc
            starter_pit_status = "PASS"
        else:
            starter_status = "not_yet_public_or_unverifiable"
            starter_source = None
            starter_observed_at = None
            starter_pit_status = "UNVERIFIABLE"

        feature_row = pd.Series({
            "league": "NPB",
            "game_id": game["game_id"],
            "datetime": pd.Timestamp(start_dt),
            "home": home,
            "away": away,
            "home_starter": home_starter,
            "away_starter": away_starter,
            "confirmed_starters": starter_known,
            "starter_evidence_status": starter_status,
            "competition": competition,
            "competition_stage": competition_stage,
            "season_type": season_type,
            "game_class": game_class,
            "competition_key": competition_key,
            "competition_classification_status": classification_status,
            "home_score": np.nan,
            "away_score": np.nan,
            "venue": "unknown",
            "series_description": "",
        })
        xrow = pd.DataFrame([bt.match_features(feature_row)]).replace(
            [float("inf"), float("-inf")], float("nan")
        )
        if xrow.isna().any().any() or not np.isfinite(xrow.to_numpy(dtype=float)).all():
            raise RuntimeError(
                f"Daily research target feature vector is non-finite for {home} vs {away}."
            )
        xrow = xrow.astype(float)
        feature_variant = os.getenv(
            "BASEBALL_FEATURE_SET_VARIANT", "FULL_VALIDATED_ENSEMBLE"
        )
        xrow, selected_meta = select_feature_set(
            xrow, "NPB", variant=feature_variant
        )
        current_schema = _feature_schema_metadata(xrow.columns)
        current_schema["feature_set_variant"] = selected_meta["feature_set_variant"]
        current_schema["feature_family_counts"] = selected_meta["feature_family_counts"]
        if schema_meta is None:
            schema_meta = current_schema
        elif schema_meta["feature_schema_hash"] != current_schema["feature_schema_hash"]:
            raise RuntimeError("Daily research target games produced non-deterministic feature schemas.")

        p = bt.ensemble_proba(fitted, xrow, "NPB")[0]
        if score_fit is None:
            lh, la, shared = direct_pit_safe_lambdas(hist, feature_row, bt)
            model_label = "current-ensemble-classifier + PIT-safe direct score recovery"
        else:
            lh, la, shared = bt.predict_scores(score_fit, xrow, "NPB")
            degenerate = abs(lh - la) < 1e-12 and abs(lh - 2.35) < 1e-12
            if degenerate:
                lh, la, shared = direct_pit_safe_lambdas(hist, feature_row, bt)
                model_label = "current-ensemble-classifier + PIT-safe direct score recovery"
            else:
                model_label = "current-ensemble + historical score ensemble"

        # Keep total expected runs from the score model, while using the classifier
        # only for a bounded home/away allocation adjustment, matching production.
        total_runs = max(float(lh) + float(la), 1e-6)
        non_draw_share = float(p[0]) / max(float(p[0] + p[2]), 1e-9)
        score_share = float(lh) / total_runs
        share = float(np.clip(0.75 * score_share + 0.25 * non_draw_share, 0.15, 0.85))
        lh, la = total_runs * share, total_runs * (1.0 - share)

        top4 = score_candidates(lh, la, shared, 4)
        low, high = low_high_probs(lh, la, shared)
        home_final, draw_final, away_final = npb_final_outcomes(lh, la, shared)

        simulation = _simulation_summary(
            lh,
            la,
            simulations=simulations,
            seed=42 + game_index,
        )
        final_gap = max(
            abs(simulation["home_final_mc_pct"] / 100.0 - home_final),
            abs(simulation["draw_final_mc_pct"] / 100.0 - draw_final),
            abs(simulation["away_final_mc_pct"] / 100.0 - away_final),
        )
        outputs.append({
            "prediction_id": _stable_prediction_id(game["game_id"], cutoff_utc),
            "game_id": game["game_id"],
            "datetime_jst": start_dt.astimezone(JST).isoformat(),
            "home": home,
            "away": away,
            "home_starter": home_starter,
            "away_starter": away_starter,
            "starter_evidence_status": starter_status,
            "starter_source": starter_source,
            "starter_evidence_observed_at_utc": starter_observed_at,
            "starter_pit_status": starter_pit_status,
            "prediction_cutoff_utc": cutoff_utc,
            "prediction_generated_at": _now_utc().isoformat(),
            "prediction_source": SOURCE,
            "prediction_schedule": "scheduled_15m_early_research",
            "prediction_target_lead_minutes": round(
                (start_dt - pd.Timestamp(cutoff_utc).to_pydatetime()).total_seconds() / 60.0, 3
            ),
            "preferred_prediction_target_lead_minutes": 60.0,
            "preferred_60m_met": False,
            "home_win_pct": round(float(home_final) * 100.0, 4),
            "draw_pct": round(float(draw_final) * 100.0, 4),
            "away_win_pct": round(float(away_final) * 100.0, 4),
            "low_pct": round(float(low) * 100.0, 4),
            "high_pct": round(float(high) * 100.0, 4),
            "top4_exact_scores": [
                {"score": s, "prob_pct": round(float(v) * 100.0, 4)}
                for s, v in top4
            ],
            "lambda_home": float(lh),
            "lambda_away": float(la),
            "shared_lambda": float(shared),
            "model": model_label,
            "research_model_version": "npb-daily-research-v1",
            "research_only": True,
            "production_eligible": False,
            "pit_status": starter_pit_status,
            "pit_quality": (
                "PIT_PASS_NO_STARTER_DEPENDENCY_PROVEN"
                if starter_known
                else "UNVERIFIABLE_STARTER_INFORMATION"
            ),
            "starter_uncertainty": not starter_known,
            "predictability_status": (
                "NORMAL_STARTER_CONFIRMED"
                if starter_known
                else "REDUCED_STARTER_INFORMATION"
            ),
            "competition": competition,
            "competition_stage": competition_stage,
            "season_type": season_type,
            "game_class": game_class,
            "competition_key": competition_key,
            "competition_classification_status": classification_status,
            "historical_games_used": int(len(hist)),
            "historical_dataset_hash": dataset_hash,
            "feature_manifest_version": schema_meta["feature_manifest_version"],
            "feature_set_id": schema_meta["feature_set_id"],
            "feature_count": schema_meta["feature_count"],
            "feature_schema_hash": schema_meta["feature_schema_hash"],
            "feature_context_mode": schema_meta["feature_context_mode"],
            "validation_scores": validation_scores,
            "game_script": simulation,
            "game_script_endpoint_use": "RESEARCH_ONLY_NOT_USED_TO_CHANGE_FINAL_PROBABILITIES",
            "game_script_consistency_gap": round(float(final_gap), 6),
            "game_script_consistency_status": (
                "OK" if final_gap <= 0.05 else "REVIEW"
            ),
        })

    result = {
        "schema_version": "npb-daily-research-forecast-v1",
        "target_date": target_date,
        "execution_status": "RESEARCH_SHADOW_EXECUTED",
        "scope": "RESEARCH_SHADOW",
        "production_modified": False,
        "production_runtime_callable": bool(production_eligible("NPB")),
        "target_strategy_contract": "NPB_HOME_DRAW_AWAY_SCORE_LOW_HIGH_TOP4_V1",
        "source": {
            "schedule": "NPB_OFFICIAL",
            "starter": "NPB_OFFICIAL_ONLY_WHEN_AVAILABLE",
            "historical_pbp": "repository_historical_pbp",
        },
        "official_schedule_probe": {
            "window_min_lead_minutes": EARLY_MIN_LEAD,
            "window_max_lead_minutes": EARLY_MAX_LEAD,
        },
        "competition_metadata": metadata,
        "prediction_generated_at": _now_utc().isoformat(),
        "prediction_cutoff_utc": cutoff_utc,
        "predictions": outputs,
        "governance": {
            "auto_promotion": False,
            "current_production_runtime_changed": False,
            "unverifiable_starter_snapshots_must_not_enter_production_quality_oos": True,
            "official_starter_refresh_expected_at_60m": True,
            "historical_records_are_append_only": True,
        },
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-date")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--simulations", type=int, default=10000)
    args = parser.parse_args(argv)

    target_date = _target_date(args.target_date)
    if args.probe_only:
        payload = probe(target_date)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    result = run_forecast(
        target_date=target_date,
        data_dir=args.data_dir,
        simulations=args.simulations,
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = _now_utc().strftime("%Y%m%dT%H%M%SZ")
    path = Path(args.output) if args.output else RESULTS / f"npb_daily_research_forecast_{target_date}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(path),
        "execution_status": result.get("execution_status"),
        "prediction_rows": len(result.get("predictions", [])),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
