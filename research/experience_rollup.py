"""Full production-prediction experience rollup.

The canonical experience ledger keeps the latest pregame prediction per game.
This module additionally evaluates *every* preserved pregame prediction
snapshot after official results become available. This creates a richer
learning corpus without double-counting repeated predictions in the main
summary.

The result is deliberately read-only with respect to the model: it records
what happened, when the prediction was made, which weights were used, and how
the final forecast performed. Future research may consume these rows only when
their experience_available_at_utc is earlier than the research cutoff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
SNAPSHOT_PATH = EXPERIENCE / "snapshot_experience_ledger.csv"
SNAPSHOT_JSONL = EXPERIENCE / "snapshot_experience_ledger.jsonl"
CASE_SUMMARY_PATH = EXPERIENCE / "experience_case_summary.json"
TRAINING_INDEX_PATH = EXPERIENCE / "experience_training_index.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prediction_id(row: dict[str, Any]) -> str:
    raw = f"{row.get('game_id')}|{row.get('prediction_cutoff_utc')}|{row.get('git_commit','unknown')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _parse_json_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _top4(row: Any) -> list[str]:
    raw = _parse_json_value(row)
    if not isinstance(raw, list):
        return []
    return [
        str(x.get("score"))
        for x in raw
        if isinstance(x, dict) and x.get("score") is not None
    ]


def _weight_dict(row: pd.Series) -> dict[str, float]:
    raw = _parse_json_value(row.get("classification_regime_model_weights"))
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            x = float(v)
            if math.isfinite(x) and x >= 0:
                out[str(k)] = x
        except (TypeError, ValueError):
            pass
    return out


def _load_predictions() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(PRED_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "prediction_id" not in df:
        df["prediction_id"] = df.apply(lambda r: _prediction_id(r.to_dict()), axis=1)

    df["prediction_cutoff_utc"] = pd.to_datetime(
        df.get("prediction_cutoff_utc"), utc=True, errors="coerce"
    )
    df["datetime_jst"] = pd.to_datetime(
        df.get("datetime_jst"), utc=True, errors="coerce"
    )
    df = df.dropna(subset=["prediction_cutoff_utc", "datetime_jst", "game_id"])

    # Experience must represent information available before first pitch.
    pregame = df["prediction_cutoff_utc"] < df["datetime_jst"]
    df = df.loc[pregame].copy()
    if df.empty:
        return df

    df = df.drop_duplicates("prediction_id", keep="last")
    return df.sort_values(["prediction_cutoff_utc", "game_id", "prediction_id"]).reset_index(drop=True)


def _cache_results(dates: list[pd.Timestamp]) -> pd.DataFrame:
    if not dates:
        return pd.DataFrame()
    needed = sorted({(int(d.year), int(d.month)) for d in dates})
    chunks: list[pd.DataFrame] = []
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for year, month in needed:
        path = RESULT_DIR / f"{year:04d}-{month:02d}.csv"
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


def _evaluate(merged: pd.DataFrame) -> pd.DataFrame:
    x = merged.copy()
    x["actual_home_score"] = pd.to_numeric(x["home_score"], errors="coerce")
    x["actual_away_score"] = pd.to_numeric(x["away_score"], errors="coerce")
    x = x.dropna(subset=["actual_home_score", "actual_away_score"]).copy()
    if x.empty:
        return x

    x["actual_home_score"] = x["actual_home_score"].astype(int)
    x["actual_away_score"] = x["actual_away_score"].astype(int)
    x["actual_total_runs"] = x["actual_home_score"] + x["actual_away_score"]
    x["actual_outcome"] = np.where(
        x["actual_home_score"] > x["actual_away_score"], "HOME_WIN",
        np.where(x["actual_home_score"] == x["actual_away_score"], "DRAW", "AWAY_WIN"),
    )

    for col in ("home_win_pct", "draw_pct", "away_win_pct", "low_pct", "high_pct"):
        x[col] = pd.to_numeric(x[col], errors="coerce") / 100.0
    probs = x[["home_win_pct", "draw_pct", "away_win_pct"]].to_numpy(float)
    if not np.isfinite(probs).all():
        raise RuntimeError("experience snapshot contains non-finite probabilities")

    y = x["actual_outcome"].map({
        "HOME_WIN": 0, "DRAW": 1, "AWAY_WIN": 2
    }).to_numpy(int)
    pred_idx = np.argmax(probs, axis=1)
    x["predicted_outcome"] = np.asarray(
        ["HOME_WIN", "DRAW", "AWAY_WIN"], dtype=object
    )[pred_idx]
    x["outcome_correct"] = (pred_idx == y).astype(int)
    x["logloss"] = -np.log(np.clip(probs[np.arange(len(x)), y], 1e-12, 1.0))
    x["brier"] = np.sum((probs - np.eye(3)[y]) ** 2, axis=1)

    x["high_probability"] = pd.to_numeric(x["high_pct"], errors="coerce")
    x["low_high_actual"] = (x["actual_total_runs"] >= 7).astype(int)
    x["low_high_predicted"] = (x["high_probability"] >= 50.0).astype(int)
    x["low_high_correct"] = (
        x["low_high_predicted"] == x["low_high_actual"]
    ).astype(int)

    x["top4_hit"] = [
        int(f"{h}-{a}" in set(_top4(r.get("top4_exact_scores"))))
        for h, a, r in zip(
            x["actual_home_score"], x["actual_away_score"], x.to_dict("records")
        )
    ]
    x["top1_exact_hit"] = [
        int(
            bool(_top4(r.get("top4_exact_scores")))
            and f"{h}-{a}" == _top4(r.get("top4_exact_scores"))[0]
        )
        for h, a, r in zip(
            x["actual_home_score"], x["actual_away_score"], x.to_dict("records")
        )
    ]

    lh = pd.to_numeric(x.get("lambda_home"), errors="coerce").fillna(0.0)
    la = pd.to_numeric(x.get("lambda_away"), errors="coerce").fillna(0.0)
    x["score_mae"] = (
        np.abs(lh.to_numpy(float) - x["actual_home_score"].to_numpy(float))
        + np.abs(la.to_numpy(float) - x["actual_away_score"].to_numpy(float))
    ) / 2.0

    x["prediction_horizon_minutes"] = (
        x["datetime_jst"].dt.tz_convert("UTC")
        - x["prediction_cutoff_utc"]
    ).dt.total_seconds() / 60.0
    x["experience_available_at_utc"] = _now()

    return x


def _write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n"
            for row in df.to_dict("records")
        ),
        encoding="utf-8",
    )


def rollup() -> dict[str, Any]:
    pred = _load_predictions()
    EXPERIENCE.mkdir(parents=True, exist_ok=True)
    if pred.empty:
        payload = {
            "generated_at_utc": _now(),
            "status": "NO_PREGAME_PREDICTIONS",
            "snapshot_rows": 0,
            "matched_rows": 0,
        }
        CASE_SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        TRAINING_INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    results = _cache_results(pred["datetime_jst"].tolist())
    if results.empty:
        payload = {
            "generated_at_utc": _now(),
            "status": "NO_COMPLETED_RESULTS",
            "snapshot_rows": int(len(pred)),
            "matched_rows": 0,
        }
        CASE_SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    pred["date_key"] = pred["datetime_jst"].dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m-%d")
    # Official result caches are CSV-backed and may be loaded as plain strings
    # (e.g. YYYY-MM-DD). Normalize them explicitly before using .dt so the
    # rollup remains compatible across pandas versions and test fixtures.
    results["date"] = pd.to_datetime(results["date"], errors="coerce", utc=True)
    results = results.dropna(subset=["date"]).copy()
    results["date_key"] = results["date"].dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m-%d")
    pred["home_key"] = pred["home"].astype(str)
    pred["away_key"] = pred["away"].astype(str)
    results["home_key"] = results["home"].astype(str)
    results["away_key"] = results["away"].astype(str)

    merged = pred.merge(
        results[[
            "date_key", "home_key", "away_key",
            "home_score", "away_score", "source_url",
        ]],
        on=["date_key", "home_key", "away_key"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_result"),
    )
    scored = _evaluate(merged)
    if scored.empty:
        payload = {
            "generated_at_utc": _now(),
            "status": "NO_COMPLETED_RESULTS",
            "snapshot_rows": int(len(pred)),
            "matched_rows": 0,
        }
        CASE_SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    weight_names = sorted({
        name
        for _, row in scored.iterrows()
        for name in _weight_dict(row)
    })
    for name in weight_names:
        scored[f"classification_weight_{name}"] = scored.apply(
            lambda r: _weight_dict(r).get(name, np.nan), axis=1
        )
    scored["dominant_classification_expert"] = scored.apply(
        lambda r: max(
            (
                (name, r.get(f"classification_weight_{name}", np.nan))
                for name in weight_names
                if pd.notna(r.get(f"classification_weight_{name}", np.nan))
            ),
            key=lambda item: item[1],
            default=(None, np.nan),
        )[0],
        axis=1,
    )

    keep = [
        "prediction_id", "source_run_id", "game_id", "date_key",
        "datetime_jst", "prediction_cutoff_utc", "prediction_generated_at",
        "prediction_horizon_minutes", "home", "away", "home_starter", "away_starter",
        "regime", "score_regime", "model", "situation_tags",
        "home_win_pct", "draw_pct", "away_win_pct", "predicted_outcome",
        "actual_outcome", "outcome_correct", "logloss", "brier",
        "low_pct", "high_pct", "low_high_actual", "low_high_predicted",
        "low_high_correct", "score_mae", "top1_exact_hit", "top4_hit",
        "lambda_home", "lambda_away", "shared_lambda",
        "dominant_classification_expert", "experience_available_at_utc",
        "source_url",
    ] + [f"classification_weight_{n}" for n in weight_names]
    keep = [c for c in keep if c in scored.columns]
    scored = scored[keep].copy()
    scored = scored.drop_duplicates("prediction_id", keep="last").sort_values(
        ["prediction_cutoff_utc", "game_id", "prediction_id"]
    )

    # Preserve every snapshot for research. This is intentionally separate from
    # the canonical one-row-per-game experience ledger.
    scored.to_csv(SNAPSHOT_PATH, index=False)
    _write_jsonl(scored, SNAPSHOT_JSONL)

    # Canonical case summary = last valid pregame forecast per game.
    canonical = scored.sort_values(
        ["game_id", "prediction_cutoff_utc", "prediction_id"]
    ).drop_duplicates("game_id", keep="last").reset_index(drop=True)

    by_tag: dict[str, dict[str, float]] = {}
    for _, row in canonical.iterrows():
        tags = _parse_json_value(row.get("situation_tags"))
        if not isinstance(tags, list):
            tags = []
        for tag in tags:
            key = str(tag)
            by_tag.setdefault(key, {"rows": 0, "accuracy": 0.0, "logloss": 0.0})
            by_tag[key]["rows"] += 1
            by_tag[key]["accuracy"] += float(row["outcome_correct"])
            by_tag[key]["logloss"] += float(row["logloss"])
    for vals in by_tag.values():
        vals["accuracy"] /= max(1, vals["rows"])
        vals["logloss"] /= max(1, vals["rows"])

    def compact(df: pd.DataFrame) -> dict[str, float]:
        return {
            "rows": int(len(df)),
            "accuracy": float(df["outcome_correct"].mean()),
            "logloss": float(df["logloss"].mean()),
            "brier": float(df["brier"].mean()),
            "draw_recall": float(
                (
                    (df["predicted_outcome"] == "DRAW")
                    & (df["actual_outcome"] == "DRAW")
                ).sum()
                / max(1, (df["actual_outcome"] == "DRAW").sum())
            ),
            "low_high_accuracy": float(df["low_high_correct"].mean()),
            "top1_exact_hit_rate": float(df["top1_exact_hit"].mean()),
            "top4_exact_hit_rate": float(df["top4_hit"].mean()),
            "score_mae": float(df["score_mae"].mean()),
        }

    result = {
        "generated_at_utc": _now(),
        "status": "UPDATED",
        "prediction_snapshot_rows_total": int(len(pred)),
        "matched_snapshot_rows": int(len(scored)),
        "unique_games_with_results": int(canonical["game_id"].nunique()),
        "experience_cases": compact(canonical),
        "all_snapshot_experience": compact(scored),
        "by_regime": {},
        "by_dominant_expert": {},
        "by_situation_tag": by_tag,
        "rolling": {},
        "training_contract": {
            "usable_after_result_available_only": True,
            "pregame_cutoff_required": True,
            "future_target_data_after_cutoff_for_training": False,
            "deduplicate_for_main_case_metrics_by_game": True,
            "preserve_all_snapshots_for_research": True,
        },
    }

    for key, group in canonical.groupby("regime", dropna=False):
        result["by_regime"][str(key)] = compact(group)
    for key, group in canonical.groupby("dominant_classification_expert", dropna=False):
        result["by_dominant_expert"][str(key)] = compact(group)

    ordered = canonical.sort_values(["prediction_cutoff_utc", "game_id"])
    for n in (30, 100, 300):
        g = ordered.tail(n)
        if len(g):
            result["rolling"][str(n)] = compact(g)

    CASE_SUMMARY_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    TRAINING_INDEX_PATH.write_text(
        json.dumps({
            "generated_at_utc": result["generated_at_utc"],
            "status": result["status"],
            "canonical_experience_cases": int(len(canonical)),
            "all_prediction_snapshots": int(len(scored)),
            "artifacts": {
                "snapshot_csv": str(SNAPSHOT_PATH),
                "snapshot_jsonl": str(SNAPSHOT_JSONL),
                "canonical_csv": str(EXPERIENCE / "experience_ledger.csv"),
                "summary_json": str(EXPERIENCE / "experience_summary.json"),
            },
            "strict_reuse_rule": "Only rows whose prediction_cutoff_utc and experience_available_at_utc are both earlier than the future research cutoff may be used as experience-derived inputs.",
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(rollup(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
