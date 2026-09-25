"""Accumulate production prediction experience and post-game performance.

This module keeps prediction snapshots immutable-ish at the event/cutoff level,
then reconciles them with official completed NPB results. It produces a durable
experience ledger that can be used by future forward research only after the
result became available.

No model is modified by this module.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.npb_official_results import _fetch_month

ROOT = Path(__file__).resolve().parents[1]
EXPERIENCE = ROOT / "data" / "experience"
PRED_DIR = EXPERIENCE / "predictions"
RESULT_DIR = EXPERIENCE / "official_results"
SUMMARY_PATH = EXPERIENCE / "experience_summary.json"
LEDGER_PATH = EXPERIENCE / "experience_ledger.csv"
LEDGER_JSONL = EXPERIENCE / "experience_ledger.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(v: Any) -> float:
    x = float(v)
    if not math.isfinite(x):
        raise ValueError("non-finite value")
    return x


def _prediction_id(row: dict[str, Any]) -> str:
    raw = f"{row['game_id']}|{row['prediction_cutoff_utc']}|{row.get('git_commit','unknown')}"
    import hashlib
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _iter_prediction_files() -> list[Path]:
    return sorted(PRED_DIR.glob("*.jsonl"))


def archive_production_output(input_json: str | Path, *, run_id: str | None = None) -> dict[str, int]:
    """Archive every production snapshot, preserving every prediction cutoff."""
    src = Path(input_json)
    obj = json.loads(src.read_text(encoding="utf-8"))
    if obj.get("execution_status") != "EXECUTED":
        return {"archived": 0, "skipped": len(obj.get("predictions", [])), "updated": 0}
    target_date = str(obj.get("target_date"))
    predictions = obj.get("predictions", [])
    if not target_date or not isinstance(predictions, list):
        raise ValueError("invalid production output contract")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{target_date}.jsonl"

    existing: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("prediction_id") or "")
            if key:
                existing[key] = row

    archived = updated = 0
    for pred in predictions:
        if not isinstance(pred, dict):
            raise ValueError("prediction row must be an object")
        if not pred.get("game_id") or not pred.get("prediction_cutoff_utc"):
            raise ValueError("prediction row missing game_id or cutoff")
        record = dict(pred)
        record["prediction_id"] = _prediction_id(record)
        record["source_run_id"] = str(run_id) if run_id is not None else None
        record["archived_at_utc"] = _utc_now()
        key = record["prediction_id"]
        if key not in existing:
            existing[key] = record
            archived += 1
        else:
            updated += 1

    rows = sorted(
        existing.values(),
        key=lambda x: (str(x.get("datetime_jst", "")), str(x.get("prediction_cutoff_utc", "")), str(x.get("game_id", "")))
    )
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )
    tmp.replace(path)
    return {"archived": archived, "skipped": 0, "updated": updated}


def _load_predictions() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in _iter_prediction_files():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "prediction_id" not in df:
        df["prediction_id"] = df.apply(lambda r: _prediction_id(r.to_dict()), axis=1)
    # Canonical production experience = latest pregame snapshot for each event.
    df["prediction_cutoff_utc"] = pd.to_datetime(df["prediction_cutoff_utc"], utc=True, errors="coerce")
    df["datetime_jst"] = pd.to_datetime(df["datetime_jst"], utc=True, errors="coerce")
    df = df.dropna(subset=["prediction_cutoff_utc", "datetime_jst", "game_id"])
    df = df.sort_values(["game_id", "prediction_cutoff_utc", "prediction_id"])
    df = df.drop_duplicates("game_id", keep="last").reset_index(drop=True)
    return df


def _result_cache_path(year: int, month: int) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR / f"{year:04d}-{month:02d}.csv"


def _load_cached_results(dates: list[pd.Timestamp]) -> pd.DataFrame:
    if not dates:
        return pd.DataFrame()
    needed = sorted({(int(d.year), int(d.month)) for d in dates})
    chunks: list[pd.DataFrame] = []
    for year, month in needed:
        path = _result_cache_path(year, month)
        if not path.exists() or path.stat().st_size == 0:
            rows = _fetch_month(year, month)
            pd.DataFrame(rows, columns=[
                "date", "home", "away", "home_score", "away_score", "source_url"
            ]).to_csv(path, index=False)
        try:
            got = pd.read_csv(path)
        except Exception:
            path.unlink(missing_ok=True)
            rows = _fetch_month(year, month)
            pd.DataFrame(rows, columns=[
                "date", "home", "away", "home_score", "away_score", "source_url"
            ]).to_csv(path, index=False)
            got = pd.read_csv(path)
        chunks.append(got)
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def _expand_weights(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("classification_regime_model_weights", "score_regime_model_weights"):
        raw = row.get(key)
        if not raw:
            continue
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                continue
        if isinstance(raw, dict):
            prefix = "classification_weight_" if key.startswith("classification") else "score_weight_"
            for name, value in raw.items():
                try:
                    out[f"{prefix}{name}"] = float(value)
                except (TypeError, ValueError):
                    continue
    return out


def _parse_top4(row: Any) -> list[tuple[str, float]]:
    if isinstance(row, list):
        return [(str(x.get("score")), float(x.get("prob_pct", 0.0))) for x in row if isinstance(x, dict)]
    if isinstance(row, str):
        try:
            return _parse_top4(json.loads(row))
        except Exception:
            return []
    return []


def reconcile() -> dict[str, Any]:
    pred = _load_predictions()
    if pred.empty:
        payload = {
            "generated_at_utc": _utc_now(),
            "status": "NO_PREDICTIONS",
            "matched_rows": 0,
            "new_experiences": 0,
        }
        EXPERIENCE.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    dates = [d for d in pred["datetime_jst"] if pd.notna(d)]
    results = _load_cached_results(dates)
    if results.empty:
        return {
            "generated_at_utc": _utc_now(),
            "status": "NO_COMPLETED_RESULTS",
            "matched_rows": 0,
            "new_experiences": 0,
            "prediction_rows": int(len(pred)),
        }

    pred = pred.copy()
    pred["date_key"] = pred["datetime_jst"].dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m-%d")
    results["date_key"] = results["date"].dt.strftime("%Y-%m-%d")
    for col in ("home", "away"):
        pred[f"{col}_key"] = pred[col].astype(str)
        results[f"{col}_key"] = results[col].astype(str)

    merged = pred.merge(
        results[["date_key", "home_key", "away_key", "home_score", "away_score", "source_url"]],
        on=["date_key", "home_key", "away_key"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_result"),
    )
    matched = merged.dropna(subset=["home_score", "away_score"]).copy()
    if matched.empty:
        return {
            "generated_at_utc": _utc_now(),
            "status": "NO_COMPLETED_RESULTS",
            "matched_rows": 0,
            "new_experiences": 0,
            "prediction_rows": int(len(pred)),
        }

    matched["actual_home_score"] = matched["home_score"].astype(int)
    matched["actual_away_score"] = matched["away_score"].astype(int)
    matched["actual_total_runs"] = matched["actual_home_score"] + matched["actual_away_score"]
    matched["actual_outcome"] = np.where(
        matched["actual_home_score"] > matched["actual_away_score"], "HOME_WIN",
        np.where(matched["actual_home_score"] == matched["actual_away_score"], "DRAW", "AWAY_WIN")
    )
    prob_cols = ["home_win_pct", "draw_pct", "away_win_pct"]
    for c in prob_cols:
        matched[c] = pd.to_numeric(matched[c], errors="coerce") / 100.0
    if not np.isfinite(matched[prob_cols].to_numpy(float)).all():
        raise RuntimeError("experience probabilities contain non-finite values")
    matched["predicted_outcome"] = np.array([
        ["HOME_WIN", "DRAW", "AWAY_WIN"][int(np.argmax(r))]
        for r in matched[prob_cols].to_numpy(float)
    ])
    matched["outcome_correct"] = (matched["predicted_outcome"] == matched["actual_outcome"]).astype(int)

    y_idx = matched["actual_outcome"].map({"HOME_WIN": 0, "DRAW": 1, "AWAY_WIN": 2}).to_numpy(int)
    pp = matched[prob_cols].to_numpy(float)
    matched["logloss"] = -np.log(np.clip(pp[np.arange(len(pp)), y_idx], 1e-12, 1.0))
    matched["brier"] = np.sum((pp - np.eye(3)[y_idx]) ** 2, axis=1)
    matched["home_probability_error"] = pp[:, 0] - (y_idx == 0)
    matched["draw_probability_error"] = pp[:, 1] - (y_idx == 1)
    matched["away_probability_error"] = pp[:, 2] - (y_idx == 2)

    matched["low_probability"] = pd.to_numeric(matched["low_pct"], errors="coerce") / 100.0
    matched["high_probability"] = pd.to_numeric(matched["high_pct"], errors="coerce") / 100.0
    matched["low_high_actual"] = (matched["actual_total_runs"] >= 7).astype(int)
    matched["low_high_predicted"] = (matched["high_probability"] >= 0.5).astype(int)
    matched["low_high_correct"] = (matched["low_high_predicted"] == matched["low_high_actual"]).astype(int)

    def score_eval(row: pd.Series) -> tuple[float, int, int]:
        picks = _parse_top4(row.get("top4_exact_scores"))
        actual = f"{int(row['actual_home_score'])}-{int(row['actual_away_score'])}"
        top1 = picks[0][0] if picks else None
        hit = int(actual in {x[0] for x in picks})
        top1_hit = int(actual == top1) if top1 else 0
        # MAE of the 4-candidate mean is intentionally not claimed as exact-score accuracy.
        pred_lambda_h = float(row.get("lambda_home", 0.0))
        pred_lambda_a = float(row.get("lambda_away", 0.0))
        mae = (abs(pred_lambda_h - row["actual_home_score"]) + abs(pred_lambda_a - row["actual_away_score"])) / 2.0
        return float(mae), hit, top1_hit

    score_vals = matched.apply(score_eval, axis=1, result_type="expand")
    score_vals.columns = ["score_mae", "top4_hit", "top1_exact_hit"]
    matched = pd.concat([matched.reset_index(drop=True), score_vals.reset_index(drop=True)], axis=1)

    expanded_weight_rows = matched.apply(lambda r: _expand_weights(r.to_dict()), axis=1).tolist()
    for row in expanded_weight_rows:
        for k, v in row.items():
            # Add lazily; later cast below.
            pass
    weight_keys = sorted({k for row in expanded_weight_rows for k in row})
    for k in weight_keys:
        matched[k] = [row.get(k, np.nan) for row in expanded_weight_rows]
    matched["dominant_classification_expert"] = matched.apply(
        lambda r: max(
            ((k, r[k]) for k in weight_keys if k.startswith("classification_weight_") and pd.notna(r[k])),
            key=lambda x: x[1],
            default=(None, np.nan),
        )[0],
        axis=1,
    )

    matched["experience_available_at_utc"] = _utc_now()

    keep = [
        "prediction_id", "game_id", "date_key", "datetime_jst", "prediction_cutoff_utc",
        "prediction_generated_at", "home", "away", "home_starter", "away_starter",
        "regime", "score_regime", "model", "situation_tags",
        "home_win_pct", "draw_pct", "away_win_pct", "predicted_outcome",
        "actual_outcome", "outcome_correct", "logloss", "brier",
        "home_probability_error", "draw_probability_error", "away_probability_error",
        "low_pct", "high_pct", "low_high_actual", "low_high_correct",
        "score_mae", "top1_exact_hit", "top4_hit",
        "lambda_home", "lambda_away", "shared_lambda",
        "dominant_classification_expert", "experience_available_at_utc", "source_url",
    ] + weight_keys
    keep = [c for c in keep if c in matched.columns]
    experience = matched[keep].copy()

    # Upsert by prediction_id so reruns never duplicate an experience case.
    existing = pd.DataFrame()
    if LEDGER_PATH.exists() and LEDGER_PATH.stat().st_size > 0:
        existing = pd.read_csv(LEDGER_PATH)
    if not existing.empty:
        experience = pd.concat([existing, experience], ignore_index=True)
    if not experience.empty:
        experience = experience.drop_duplicates("prediction_id", keep="last").sort_values(
            ["prediction_cutoff_utc", "game_id"]
        ).reset_index(drop=True)
    EXPERIENCE.mkdir(parents=True, exist_ok=True)
    experience.to_csv(LEDGER_PATH, index=False)
    LEDGER_JSONL.write_text(
        "".join(json.dumps(r, ensure_ascii=False, default=str, sort_keys=True) + "\n"
                for r in experience.to_dict("records")),
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "generated_at_utc": _utc_now(),
        "status": "UPDATED",
        "prediction_rows": int(len(pred)),
        "matched_rows_total": int(len(experience)),
        "newly_matched_rows": int(len(matched)),
        "outcome_accuracy": float(experience["outcome_correct"].mean()),
        "logloss": float(experience["logloss"].mean()),
        "brier": float(experience["brier"].mean()),
        "draw_rows": int((experience["actual_outcome"] == "DRAW").sum()),
        "draw_recall": float((
            (experience["predicted_outcome"] == "DRAW")
            & (experience["actual_outcome"] == "DRAW")
        ).sum() / max(1, (experience["actual_outcome"] == "DRAW").sum())),
        "low_high_accuracy": float(experience["low_high_correct"].mean()),
        "score_mae": float(experience["score_mae"].mean()),
        "top1_exact_hit_rate": float(experience["top1_exact_hit"].mean()),
        "top4_exact_hit_rate": float(experience["top4_hit"].mean()),
        "by_regime": {},
        "by_dominant_expert": {},
        "rolling": {},
    }

    for key, group in experience.groupby("regime", dropna=False):
        summary["by_regime"][str(key)] = {
            "rows": int(len(group)),
            "accuracy": float(group["outcome_correct"].mean()),
            "logloss": float(group["logloss"].mean()),
            "brier": float(group["brier"].mean()),
        }
    for key, group in experience.groupby("dominant_classification_expert", dropna=False):
        summary["by_dominant_expert"][str(key)] = {
            "rows": int(len(group)),
            "accuracy": float(group["outcome_correct"].mean()),
            "logloss": float(group["logloss"].mean()),
        }
    sorted_exp = experience.sort_values(["prediction_cutoff_utc", "game_id"])
    for n in (30, 100, 300):
        g = sorted_exp.tail(n)
        if len(g):
            summary["rolling"][str(n)] = {
                "rows": int(len(g)),
                "accuracy": float(g["outcome_correct"].mean()),
                "logloss": float(g["logloss"].mean()),
                "brier": float(g["brier"].mean()),
                "low_high_accuracy": float(g["low_high_correct"].mean()),
                "top4_exact_hit_rate": float(g["top4_hit"].mean()),
            }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=str)
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--reconcile", action="store_true")
    args = parser.parse_args()
    if args.archive:
        print(json.dumps(archive_production_output(args.archive, run_id=args.run_id), ensure_ascii=False, indent=2))
    if args.reconcile:
        print(json.dumps(reconcile(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
