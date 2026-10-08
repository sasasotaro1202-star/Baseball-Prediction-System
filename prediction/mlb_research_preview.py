"""On-demand MLB research-preview prediction.

This lane is research-only and never changes the checked-in MLB production
runtime. It uses the existing chronological BaseballBacktest model on data
known by the live cutoff. Current probable-pitcher names are not consumed as
starter features because their announcement timestamp is not verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from baseball_backtest import BaseballBacktest, low_high_probs, score_candidates
from research.competition_taxonomy import classify_mlb

ROOT = Path(__file__).resolve().parents[1]
JST = ZoneInfo("Asia/Tokyo")
MLB_API = "https://statsapi.mlb.com/api/v1"
OUTPUT_SCHEMA = "baseball-mlb-research-preview-v1"
MODEL_VERSION = "mlb-research-preview-ensemble-v1"
FEATURE_VERSION = "baseball-features-v1"


def _target_date(value: str):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("date must be YYYY-MM-DD") from exc


def _utc_schedule_dates(target):
    start = datetime.combine(target, time.min, tzinfo=JST).astimezone(timezone.utc)
    end = (start + timedelta(days=1)).astimezone(timezone.utc)
    return start.date().isoformat(), end.date().isoformat()


def _schedule(bt: BaseballBacktest, target, cutoff):
    start_date, end_date = _utc_schedule_dates(target)
    payload = bt._get_json(
        f"{MLB_API}/schedule",
        params={
            "sportId": 1,
            "startDate": start_date,
            "endDate": end_date,
            "hydrate": "probablePitcher,venue",
        },
    )
    retrieved = datetime.now(timezone.utc).isoformat()
    rows = []
    for block in (payload.get("dates") or []) if isinstance(payload, dict) else []:
        for game in (block.get("games") or []) if isinstance(block, dict) else []:
            game_dt = pd.to_datetime(game.get("gameDate"), errors="coerce", utc=True)
            if pd.isna(game_dt) or game_dt.tz_convert(JST).date() != target:
                continue
            status = str((game.get("status") or {}).get("abstractGameState") or "")
            if status in {"Final", "Live"}:
                continue
            teams = game.get("teams") or {}
            home = teams.get("home") or {}
            away = teams.get("away") or {}
            home_name = str((home.get("team") or {}).get("name") or "").strip()
            away_name = str((away.get("team") or {}).get("name") or "").strip()
            if not home_name or not away_name:
                continue
            gt = str(game.get("gameType") or "")
            sd = str(game.get("seriesDescription") or "")
            cls = classify_mlb(gt, sd)
            if str(cls.status).lower() != "classified":
                continue
            rows.append({
                "game_id": str(game.get("gamePk") or "").strip(),
                "datetime": game_dt,
                "home": home_name,
                "away": away_name,
                "venue": str((game.get("venue") or {}).get("name") or ""),
                "game_type": gt,
                "series_description": sd,
                "season": str(game.get("season") or target.year),
                "competition": cls.competition,
                "competition_stage": cls.stage,
                "competition_key": cls.competition_key,
                "competition_classification_status": cls.status,
                "status": status,
                "home_starter": "",
                "away_starter": "",
                "confirmed_starters": False,
                "starter_evidence_status": "unknown",
                "prediction_cutoff": cutoff.isoformat(),
            })
    rows.sort(key=lambda x: (pd.Timestamp(x["datetime"]), x["game_id"]))
    return rows, retrieved


def _history_hash(history: pd.DataFrame) -> str:
    ordered = history.sort_values(["datetime", "game_id"]).reset_index(drop=True)
    return hashlib.sha256(
        pd.util.hash_pandas_object(ordered, index=True).values.tobytes()
    ).hexdigest()


def _future_features(bt, row, columns):
    out = pd.DataFrame([bt.match_features(pd.Series(row))]).reindex(columns=columns)
    if out.isna().any().any():
        missing = out.columns[out.isna().any()].tolist()
        raise RuntimeError("undefined MLB preview features: " + ", ".join(missing))
    return out.astype(float)


def run_prediction(*, date: str, data_dir: str = "data", output: str | None = None):
    target = _target_date(date)
    cutoff = datetime.now(timezone.utc)
    if target < cutoff.astimezone(JST).date():
        result = {
            "schema_version": OUTPUT_SCHEMA,
            "target_date": target.isoformat(),
            "execution_status": "BLOCKED_STALE_TARGET_DATE",
            "scope": "RESEARCH_SHADOW",
            "production_eligibility": False,
            "pit_status": "UNVERIFIABLE",
            "predictions": [],
        }
        _write(result, output, target)
        return result

    os.environ.setdefault("BASEBALL_FAST_OOS", "1")
    os.environ.setdefault(
        "BASEBALL_FAST_MODEL_POOL_MLB",
        "Logistic,HistGB,RandomForest,ExtraTrees",
    )
    os.environ.setdefault("BASEBALL_TIME_BUDGET_SEC", "2400")
    os.environ.setdefault("BASEBALL_HARD_CAP_SEC", "3000")

    bt = BaseballBacktest(Path(data_dir))
    history = bt.load_mlb(2020, max(2026, target.year))
    history["datetime"] = pd.to_datetime(history["datetime"], errors="coerce", utc=True)
    history = history[history["datetime"] <= cutoff].copy()
    if history.empty:
        raise RuntimeError("MLB historical corpus is empty at live cutoff")

    X, y, _ = bt.build_features(history)
    games, schedule_retrieved = _schedule(bt, target, cutoff)
    if not games:
        result = {
            "schema_version": OUTPUT_SCHEMA,
            "target_date": target.isoformat(),
            "execution_status": "NO_FUTURE_GAMES",
            "scope": "RESEARCH_SHADOW",
            "production_eligibility": False,
            "pit_status": "UNVERIFIABLE",
            "prediction_cutoff": cutoff.isoformat(),
            "schedule_available_at": schedule_retrieved,
            "history_rows": int(len(history)),
            "predictions": [],
        }
        _write(result, output, target)
        return result

    fitted, _, _ = bt.fit_ensemble(X, y, "MLB", fast_oos=True)
    if not fitted:
        raise RuntimeError("MLB research preview ensemble fit failed")
    score_fit = bt.fit_score_ensemble(
        X,
        history["home_score"].astype(float).to_numpy(),
        history["away_score"].astype(float).to_numpy(),
        "MLB",
    )
    if score_fit is None:
        raise RuntimeError("MLB research preview score fit failed")

    preds = []
    snapshot = f"mlb-history-{_history_hash(history)[:16]}"
    for row in games:
        xf = _future_features(bt, row, list(X.columns))
        p = np.asarray(bt.ensemble_proba(fitted, xf, "MLB")[0], dtype=float)
        if p.size != 2 or not np.isfinite(p).all():
            raise RuntimeError("invalid MLB probability vector")
        p = np.clip(p, 1e-9, 1.0)
        p /= p.sum()
        lam_h, lam_a, shared = bt.predict_scores(score_fit, xf, "MLB")
        start = pd.Timestamp(row["datetime"])
        preds.append({
            "game_id": row["game_id"],
            "home": row["home"],
            "away": row["away"],
            "datetime": start.isoformat(),
            "venue": row["venue"],
            "competition": row["competition"],
            "competition_stage": row["competition_stage"],
            "competition_key": row["competition_key"],
            "competition_classification_status": row["competition_classification_status"],
            "season": row["season"],
            "probabilities": {"home": float(p[0]), "away": float(p[1])},
            "score_candidates": [
                {"score": s, "probability": float(q)}
                for s, q in score_candidates(lam_h, lam_a, shared, n=4)
            ],
            "low_probability": float(low_high_probs(lam_h, lam_a, shared)[0]),
            "high_probability": float(low_high_probs(lam_h, lam_a, shared)[1]),
            "home_run_lambda": float(lam_h),
            "away_run_lambda": float(lam_a),
            "shared_lambda": float(shared),
            "starter": {
                "home": None,
                "away": None,
                "evidence_status": "UNVERIFIABLE",
            },
            "uncertainty": {
                "starter_pit": "UNVERIFIABLE",
                "model_disagreement": "NOT_COMPUTED",
                "information_quality": "PARTIAL",
            },
            "confidence": None,
            "predictability": None,
            "prediction_cutoff": cutoff.isoformat(),
            "available_at": schedule_retrieved,
            "prediction_generated_at": cutoff.isoformat(),
            "prediction_target_lead_minutes": round(
                (start.to_pydatetime() - cutoff).total_seconds() / 60.0, 3
            ),
            "prediction_source": "MLB_STATS_API_RESEARCH_PREVIEW",
            "prediction_schedule": "on_demand",
            "data_snapshot_id": snapshot,
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
        })

    result = {
        "schema_version": OUTPUT_SCHEMA,
        "target_date": target.isoformat(),
        "execution_status": "RESEARCH_SHADOW_EXECUTED",
        "scope": "RESEARCH_SHADOW",
        "production_eligibility": False,
        "pit_status": "UNVERIFIABLE",
        "prediction_cutoff": cutoff.isoformat(),
        "schedule_available_at": schedule_retrieved,
        "source": f"{MLB_API}/schedule",
        "history_rows": int(len(history)),
        "feature_rows": int(len(X)),
        "feature_schema_hash": str(
            getattr(bt, "_feature_set_metadata", {}).get("feature_schema_hash", "")
        ),
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "predictions": preds,
        "uncertainty": {
            "starter_pit": "UNVERIFIABLE",
            "model_disagreement": "NOT_COMPUTED",
            "information_quality": "PARTIAL",
        },
        "research_only": True,
        "formal_adoption_status": "NOT_PRODUCTION",
    }
    _write(result, output, target)
    return result


def _write(result, output, target):
    path = Path(output) if output else ROOT / "results" / f"mlb_research_preview_{target.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run_prediction(date=args.date, data_dir=args.data_dir, output=args.output)
    return 0 if result.get("execution_status") in {"RESEARCH_SHADOW_EXECUTED", "NO_FUTURE_GAMES"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
