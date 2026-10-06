"""PIT-safe research-shadow experience for NPB pregame forecasts.

This lane is intentionally separate from the production Experience ledger. It is
used to collect future-game evidence while production promotion remains blocked.
No shadow result can change current production runtime eligibility.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.experience_ledger import (
    _binary_ece,
    _horizon_bucket,
    _load_cached_results,
    _multiclass_ece,
    _parse_top4,
    _validate_prediction_probability_contract,
    _validate_prediction_time_contract,
    _prediction_id,
)

ROOT = Path(__file__).resolve().parents[1]
SHADOW_ROOT = ROOT / "data" / "experience" / "research_shadow"
PRED_DIR = SHADOW_ROOT / "predictions"
LEDGER_PATH = SHADOW_ROOT / "shadow_experience_ledger.csv"
LEDGER_JSONL = SHADOW_ROOT / "shadow_experience_ledger.jsonl"
SUMMARY_PATH = SHADOW_ROOT / "shadow_experience_summary.json"
CURRENT_METHOD_SUMMARY_PATH = SHADOW_ROOT / "current_method_performance.json"


def _read_prediction_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not PRED_DIR.exists():
        return rows
    for path in sorted(PRED_DIR.glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"shadow prediction row is not an object: {path}:{line_no}")
            rows.append(row)
    return rows


def _method_signature(obj: dict[str, Any]) -> str:
    """Identify the forecast algorithm without splitting on infrastructure-only commits."""
    predictions = obj.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("research-shadow output contains no predictions for method identity")
    model_labels = sorted({str(row.get("model") or "").strip() for row in predictions})
    if not model_labels or any(not label for label in model_labels):
        raise ValueError("research-shadow output is missing per-prediction model identity")
    parts = (
        str(obj.get("schema_version") or "unknown"),
        str(obj.get("feature_set_id") or "unknown"),
        str(obj.get("feature_schema_hash") or "unknown"),
        str(obj.get("feature_set_variant") or "unknown"),
        str(obj.get("feature_context_mode") or "unknown"),
        "models:" + ",".join(model_labels),
    )
    if any(value == "unknown" for value in parts):
        raise ValueError("research-shadow output is missing method identity metadata")
    return "|".join(parts)


def _write_summary(payload: dict[str, Any]) -> None:
    SHADOW_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    current = payload.get("current_method_performance")
    current_payload = {
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "scope": "RESEARCH_SHADOW_CURRENT_METHOD",
        "production_modified": False,
        "method_signature": payload.get("current_method_signature"),
        "metrics": current,
        "source": "data/experience/research_shadow/shadow_experience_summary.json",
    }
    CURRENT_METHOD_SUMMARY_PATH.write_text(
        json.dumps(current_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def archive_shadow_output(input_json: str | Path, *, run_id: str | None = None) -> dict[str, int]:
    src = Path(input_json)
    obj = json.loads(src.read_text(encoding="utf-8"))
    if obj.get("execution_status") != "RESEARCH_SHADOW_EXECUTED":
        return {"archived": 0, "skipped": len(obj.get("predictions", [])), "updated": 0}

    predictions = obj.get("predictions", [])
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("research-shadow output contains no predictions")
    method_signature = _method_signature(obj)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    target_date = str(obj.get("target_date") or "")
    if not target_date:
        raise ValueError("research-shadow output missing target_date")
    path = PRED_DIR / f"{target_date}.jsonl"

    existing: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[str(row.get("prediction_id") or "")] = row

    archived = updated = 0
    for pred in predictions:
        if not isinstance(pred, dict):
            raise ValueError("shadow prediction row must be an object")
        record = dict(pred)
        _validate_prediction_time_contract(record)
        _validate_prediction_probability_contract(record)
        record["prediction_id"] = str(record.get("prediction_id") or _prediction_id(record))
        record["source_run_id"] = str(run_id) if run_id is not None else None
        record["prediction_scope"] = "RESEARCH_SHADOW"
        record["production_eligible"] = False
        record["method_signature"] = method_signature
        record["method_git_commit"] = str(obj.get("git_commit"))
        record["feature_set_id"] = str(obj.get("feature_set_id"))
        record["feature_schema_hash"] = str(obj.get("feature_schema_hash"))
        record["feature_manifest_version"] = str(obj.get("feature_manifest_version"))
        record["feature_set_variant"] = str(obj.get("feature_set_variant"))
        record["feature_context_mode"] = str(obj.get("feature_context_mode"))
        key = record["prediction_id"]
        if key in existing:
            updated += 1
        else:
            archived += 1
        existing[key] = record

    rows = sorted(
        existing.values(),
        key=lambda r: (
            str(r.get("datetime_jst", "")),
            str(r.get("prediction_cutoff_utc", "")),
            str(r.get("prediction_id", "")),
        ),
    )
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )
    return {"archived": archived, "skipped": 0, "updated": updated}


def reconcile_shadow() -> dict[str, Any]:
    rows = _read_prediction_rows()
    generated_at = datetime.now(timezone.utc).isoformat()
    if not rows:
        payload = {
            "generated_at_utc": generated_at,
            "status": "NO_SHADOW_PREDICTIONS",
            "scope": "RESEARCH_SHADOW",
            "production_modified": False,
            "prediction_rows": 0,
            "matched_snapshots": 0,
            "canonical_cases": 0,
        }
        _write_summary(payload)
        return payload

    df = pd.DataFrame(rows)
    for row in rows:
        _validate_prediction_time_contract(row)
        _validate_prediction_probability_contract(row)
    df["datetime_jst"] = pd.to_datetime(df["datetime_jst"], utc=True, errors="coerce")
    df["prediction_generated_at"] = pd.to_datetime(df["prediction_generated_at"], utc=True, errors="coerce")
    df["prediction_cutoff_utc"] = pd.to_datetime(df["prediction_cutoff_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime_jst", "prediction_generated_at", "prediction_cutoff_utc", "game_id"]).copy()

    results = _load_cached_results(list(df["datetime_jst"]))
    if results.empty:
        payload = {
            "generated_at_utc": generated_at,
            "status": "NO_COMPLETED_RESULTS",
            "scope": "RESEARCH_SHADOW",
            "production_modified": False,
            "prediction_rows": int(len(df)),
            "matched_snapshots": 0,
            "canonical_cases": 0,
        }
        _write_summary(payload)
        return payload

    df["date_key"] = df["datetime_jst"].dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m-%d")
    results["date_key"] = pd.to_datetime(results["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ("home", "away"):
        df[f"{col}_key"] = df[col].astype(str)
        results[f"{col}_key"] = results[col].astype(str)

    merged = df.merge(
        results[["date_key", "home_key", "away_key", "home_score", "away_score", "source_url"]],
        on=["date_key", "home_key", "away_key"],
        how="left",
        validate="many_to_one",
    )
    matched = merged.dropna(subset=["home_score", "away_score"]).copy()
    if matched.empty:
        payload = {
            "generated_at_utc": generated_at,
            "status": "NO_COMPLETED_RESULTS",
            "scope": "RESEARCH_SHADOW",
            "production_modified": False,
            "prediction_rows": int(len(df)),
            "matched_snapshots": 0,
            "canonical_cases": 0,
        }
        _write_summary(payload)
        return payload

    matched["actual_home_score"] = matched["home_score"].astype(int)
    matched["actual_away_score"] = matched["away_score"].astype(int)
    matched["actual_total_runs"] = matched["actual_home_score"] + matched["actual_away_score"]
    matched["actual_outcome"] = np.where(
        matched["actual_home_score"] > matched["actual_away_score"], "HOME_WIN",
        np.where(matched["actual_home_score"] == matched["actual_away_score"], "DRAW", "AWAY_WIN"),
    )

    def to_unit(values: Any) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if np.nanmax(arr) > 1.0:
            arr = arr / 100.0
        return arr

    p = to_unit(matched[["home_win_pct", "draw_pct", "away_win_pct"]].to_numpy(float))
    y = matched["actual_outcome"].map({"HOME_WIN": 0, "DRAW": 1, "AWAY_WIN": 2}).to_numpy(int)
    pred = p.argmax(axis=1)
    matched["outcome_correct"] = (pred == y).astype(int)
    matched["logloss"] = -np.log(np.clip(p[np.arange(len(p)), y], 1e-12, 1.0))
    matched["brier"] = np.sum((p - np.eye(3)[y]) ** 2, axis=1)

    lh = to_unit(matched[["low_pct", "high_pct"]].to_numpy(float))
    matched["low_high_actual"] = (matched["actual_total_runs"] >= 7).astype(int)
    matched["low_high_predicted"] = (lh[:, 1] >= 0.5).astype(int)
    matched["low_high_correct"] = (
        matched["low_high_predicted"] == matched["low_high_actual"]
    ).astype(int)

    def score_eval(row: pd.Series) -> tuple[float, int, int]:
        picks = _parse_top4(row.get("top4_exact_scores"))
        actual = f"{int(row['actual_home_score'])}-{int(row['actual_away_score'])}"
        top1 = picks[0][0] if picks else None
        top4 = int(actual in {x[0] for x in picks})
        top1_hit = int(actual == top1) if top1 else 0
        mae = (
            abs(float(row.get("lambda_home", 0.0)) - row["actual_home_score"])
            + abs(float(row.get("lambda_away", 0.0)) - row["actual_away_score"])
        ) / 2.0
        return float(mae), top4, top1_hit

    score_values = matched.apply(score_eval, axis=1, result_type="expand")
    score_values.columns = ["score_mae", "top4_hit", "top1_exact_hit"]
    matched = pd.concat([matched.reset_index(drop=True), score_values.reset_index(drop=True)], axis=1)
    matched["actual_lead_minutes"] = (
        matched["datetime_jst"] - matched["prediction_generated_at"]
    ).dt.total_seconds() / 60.0
    matched["prediction_horizon"] = matched["actual_lead_minutes"].map(_horizon_bucket)

    # Canonical case = latest valid pregame snapshot per game.
    matched = matched.sort_values(["game_id", "prediction_cutoff_utc", "prediction_id"])
    canonical = matched.drop_duplicates("game_id", keep="last").copy()

    def metrics(frame: pd.DataFrame) -> dict[str, Any]:
        probs = p[frame.index] if set(frame.index).issubset(set(matched.index)) else None
        if probs is None:
            pp = to_unit(frame[["home_win_pct", "draw_pct", "away_win_pct"]].to_numpy(float))
        else:
            pp = probs
        yy = frame["actual_outcome"].map({"HOME_WIN": 0, "DRAW": 1, "AWAY_WIN": 2}).to_numpy(int)
        lowprob = to_unit(frame["high_pct"].to_numpy(float))
        return {
            "rows": int(len(frame)),
            "accuracy": float(frame["outcome_correct"].mean()),
            "logloss": float(frame["logloss"].mean()),
            "brier": float(frame["brier"].mean()),
            "ece": _multiclass_ece(pp, yy),
            "low_high_accuracy": float(frame["low_high_correct"].mean()),
            "top1_exact_hit_rate": float(frame["top1_exact_hit"].mean()),
            "top4_exact_hit_rate": float(frame["top4_hit"].mean()),
            "score_mae": float(frame["score_mae"].mean()),
            "mean_actual_lead_minutes": float(frame["actual_lead_minutes"].mean()),
            "revision_rate": float(frame.get("revision_status", pd.Series(index=frame.index, dtype=object)).eq("REVISED").mean()),
        }

    all_metrics = metrics(matched)
    canonical_metrics = metrics(canonical)

    by_method: dict[str, Any] = {}
    for method_signature, method_frame in matched.groupby("method_signature", dropna=False):
        method_canonical = (
            method_frame.sort_values(["game_id", "prediction_cutoff_utc", "prediction_id"])
            .drop_duplicates("game_id", keep="last")
            .copy()
        )
        by_method[str(method_signature)] = {
            "snapshot_metrics": metrics(method_frame),
            "canonical_metrics": metrics(method_canonical),
            "prediction_rows": int(len(method_frame)),
            "canonical_cases": int(len(method_canonical)),
            "first_prediction_generated_at": str(method_frame["prediction_generated_at"].min()),
            "last_prediction_generated_at": str(method_frame["prediction_generated_at"].max()),
        }

    latest_method_row = matched.sort_values(
        ["prediction_generated_at", "prediction_id"], kind="mergesort"
    ).iloc[-1]
    current_method_signature = str(latest_method_row["method_signature"])
    current_method_frame = matched.loc[matched["method_signature"] == current_method_signature].copy()
    current_method_canonical = (
        current_method_frame.sort_values(["game_id", "prediction_cutoff_utc", "prediction_id"])
        .drop_duplicates("game_id", keep="last")
        .copy()
    )
    current_method_performance = {
        "method_signature": current_method_signature,
        "snapshot_metrics": metrics(current_method_frame),
        "canonical_metrics": metrics(current_method_canonical),
        "prediction_rows": int(len(current_method_frame)),
        "canonical_cases": int(len(current_method_canonical)),
        "first_prediction_generated_at": str(current_method_frame["prediction_generated_at"].min()),
        "last_prediction_generated_at": str(current_method_frame["prediction_generated_at"].max()),
    }

    def metrics_by_horizon(frame: pd.DataFrame) -> dict[str, Any]:
        """Partition metrics by realized lead time while preserving case unit."""
        grouped: dict[str, Any] = {}
        for key in ("LT_30M", "30_TO_60M", "1_TO_3H", "3_TO_6H", "GE_6H", "UNKNOWN"):
            group = frame.loc[frame["prediction_horizon"].astype(str) == key]
            if group.empty:
                continue
            grouped[key] = metrics(group)
        return grouped

    by_horizon = metrics_by_horizon(matched)
    canonical_by_horizon = metrics_by_horizon(canonical)

    by_source: dict[str, Any] = {}
    if "prediction_source" in matched.columns:
        for source, frame in matched.groupby("prediction_source", dropna=False):
            by_source[str(source)] = metrics(frame)

    keep = [
        "prediction_id", "game_id", "datetime_jst", "prediction_cutoff_utc", "prediction_generated_at",
        "prediction_source", "prediction_schedule", "prediction_target_lead_minutes",
        "home", "away", "home_starter", "away_starter",
        "home_win_pct", "draw_pct", "away_win_pct", "actual_outcome", "outcome_correct",
        "logloss", "brier", "low_pct", "high_pct", "low_high_actual", "low_high_correct",
        "top1_exact_hit", "top4_hit", "score_mae", "lambda_home", "lambda_away",
        "actual_lead_minutes", "prediction_horizon", "source_url",
    ]
    ledger = matched[[c for c in keep if c in matched.columns]].copy()
    ledger["method_signature"] = matched["method_signature"].values
    ledger.to_csv(LEDGER_PATH, index=False)
    LEDGER_JSONL.write_text(
        "".join(json.dumps(r, ensure_ascii=False, default=str, sort_keys=True) + "\n"
                for r in ledger.to_dict("records")),
        encoding="utf-8",
    )

    payload = {
        "generated_at_utc": generated_at,
        "status": "UPDATED",
        "scope": "RESEARCH_SHADOW",
        "production_modified": False,
        "prediction_rows": int(len(df)),
        "matched_snapshots": int(len(matched)),
        "canonical_cases": int(len(canonical)),
        "all_snapshot_metrics": all_metrics,
        "canonical_metrics": canonical_metrics,
        "by_prediction_source": by_source,
        "by_method": by_method,
        "current_method_signature": current_method_signature,
        "current_method_performance": current_method_performance,
        "by_horizon": by_horizon,
        "canonical_by_horizon": canonical_by_horizon,
    }
    _write_summary(payload)
    return payload


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive")
    parser.add_argument("--run-id")
    parser.add_argument("--reconcile", action="store_true")
    args = parser.parse_args()
    if args.archive:
        print(json.dumps(archive_shadow_output(args.archive, run_id=args.run_id), ensure_ascii=False, indent=2))
    if args.reconcile:
        print(json.dumps(reconcile_shadow(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
