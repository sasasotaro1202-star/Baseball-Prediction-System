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
from research import experience_ledger as ledger
from research.experience_dimensions import add_dimensions
from research.experience_evidence import build_experience_evidence
from research.experience_ledger import (
    _horizon_bucket,
    _prediction_target_metrics,
    _validate_prediction_time_contract,
    _is_legacy_scheduled_cutoff,
    _should_refresh_result_cache,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIENCE = ROOT / "data" / "experience"
PRED_DIR = EXPERIENCE / "predictions"
RESULT_DIR = EXPERIENCE / "official_results"
SNAPSHOT_PATH = EXPERIENCE / "snapshot_experience_ledger.csv"
SNAPSHOT_JSONL = EXPERIENCE / "snapshot_experience_ledger.jsonl"
CASE_SUMMARY_PATH = EXPERIENCE / "experience_case_summary.json"
TRAINING_INDEX_PATH = EXPERIENCE / "experience_training_index.json"
PERFORMANCE_BREAKDOWN_PATH = EXPERIENCE / "performance_breakdown.json"


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


def _normalize_probability_rows(
    values: pd.DataFrame,
    *,
    label: str,
    row_ids: pd.Series | None = None,
) -> np.ndarray:
    """Validate percentage probabilities and normalize serialization-rounding drift.

    Production snapshots store percentages rounded to four decimals. The exact
    row sum can therefore differ from 100 by up to the rounding error. Only a
    tiny, explicit tolerance is accepted; materially inconsistent rows remain
    fail-closed.
    """
    arr = values.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise RuntimeError(f"{label} probability matrix must have 2 or 3 columns")
    if not np.isfinite(arr).all() or (arr < 0.0).any() or (arr > 100.0).any():
        raise RuntimeError(
            f"invalid {label} probabilities "
            "(must be finite, in [0,100], and row-normalized)"
        )
    sums = arr.sum(axis=1)
    # Three four-decimal percentage fields have <=0.00015 percentage-point
    # aggregate rounding drift. Keep a small safety margin for binary floats.
    rounding_tolerance_pct = 0.0003
    bad = np.abs(sums - 100.0) > rounding_tolerance_pct
    if bad.any():
        idx = int(np.flatnonzero(bad)[0])
        ident = row_ids.iloc[idx] if row_ids is not None and len(row_ids) > idx else idx
        raise RuntimeError(
            f"invalid {label} probabilities: "
            f"row={ident!r} sum_pct={sums[idx]:.8f}"
        )
    arr = arr / sums[:, None]
    return arr


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

    # Validate the raw snapshot contract before any filtering or research reuse.
    # Pre-v18 scheduled-cutoff snapshots are preserved but quarantined from the
    # research corpus because their stored cutoff was a planned deadline, not
    # the actual observation time.
    records = df.to_dict("records")
    legacy_rows = {
        idx for idx, row in enumerate(records) if _is_legacy_scheduled_cutoff(row)
    }
    for idx, row in enumerate(records):
        if idx in legacy_rows:
            continue
        _validate_prediction_time_contract(row)
    if legacy_rows:
        print(json.dumps({
            "event": "EXPERIENCE_LEGACY_TIMING_QUARANTINE",
            "rows": len(legacy_rows),
            "reason": "pre-v18 scheduled cutoff is not an actual observed prediction cutoff",
        }, ensure_ascii=False))

    df["prediction_cutoff_utc"] = pd.to_datetime(
        df["prediction_cutoff_utc"], utc=True, errors="coerce"
    )
    df["datetime_jst"] = pd.to_datetime(
        df["datetime_jst"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["prediction_cutoff_utc", "datetime_jst", "game_id"])

    # Experience must represent information available before first pitch.
    pregame = (
        (~df.index.isin(legacy_rows))
        & (df["prediction_cutoff_utc"] < df["datetime_jst"])
    )
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
    columns = ["date", "home", "away", "home_score", "away_score", "source_url"]
    for year, month in needed:
        path = RESULT_DIR / f"{year:04d}-{month:02d}.csv"
        got: pd.DataFrame | None = None

        refresh = _should_refresh_result_cache(year, month)
        if not refresh and path.exists() and path.stat().st_size > 0:
            try:
                got = pd.read_csv(path)
            except Exception:
                path.unlink(missing_ok=True)
                got = None

        # The active month is mutable: new games can finish after a
        # previously non-empty cache was written. Refresh it every run.
        # Closed months remain cached for efficient, reproducible reuse.
        if refresh or got is None or got.empty:
            rows = _fetch_month(year, month)
            refreshed = pd.DataFrame(rows, columns=columns)
            refreshed.to_csv(path, index=False)
            got = refreshed

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

    probs = _normalize_probability_rows(
        x[["home_win_pct", "draw_pct", "away_win_pct"]],
        label="win",
        row_ids=x.get("game_id"),
    )

    x[["home_win_pct", "draw_pct", "away_win_pct"]] = probs

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

    # high_pct was normalized from percentage points to [0, 1] above.
    # Keep the classification threshold on the same probability scale.
    low_high = _normalize_probability_rows(
        x[["low_pct", "high_pct"]],
        label="Low/High",
        row_ids=x.get("game_id"),
    )[:, :2]
    x[["low_pct", "high_pct"]] = low_high
    x["low_probability"] = low_high[:, 0]
    x["high_probability"] = low_high[:, 1]
    x["low_high_actual"] = (x["actual_total_runs"] >= 7).astype(int)
    x["low_high_predicted"] = (x["high_probability"] >= 0.5).astype(int)
    x["low_high_correct"] = (
        x["low_high_predicted"] == x["low_high_actual"]
    ).astype(int)

    top4_values = []
    top1_values = []
    for h, a, r in zip(
        x["actual_home_score"], x["actual_away_score"], x.to_dict("records")
    ):
        picks = _top4(r.get("top4_exact_scores"))
        actual = f"{h}-{a}"
        # Missing score-ranked predictions are unevaluable, not incorrect.
        top4_values.append(float(actual in set(picks)) if len(picks) == 4 else np.nan)
        top1_values.append(float(actual == picks[0]) if picks else np.nan)
    x["top4_hit"] = top4_values
    x["top1_exact_hit"] = top1_values

    lh = pd.to_numeric(x.get("lambda_home"), errors="coerce")
    la = pd.to_numeric(x.get("lambda_away"), errors="coerce")
    valid_score = lh.notna() & la.notna()
    x["score_mae"] = np.nan
    x.loc[valid_score, "score_mae"] = (
        np.abs(lh.loc[valid_score].to_numpy(float) - x.loc[valid_score, "actual_home_score"].to_numpy(float))
        + np.abs(la.loc[valid_score].to_numpy(float) - x.loc[valid_score, "actual_away_score"].to_numpy(float))
    ) / 2.0

    x["prediction_horizon_minutes"] = (
        x["datetime_jst"].dt.tz_convert("UTC")
        - x["prediction_cutoff_utc"]
    ).dt.total_seconds() / 60.0
    # Older/reduced evaluation fixtures may not carry the observed
    # generation timestamp. Do not fall back to the planned cutoff here:
    # that would silently relabel "actual lead" with scheduled-cutoff timing.
    # Missing observed-generation provenance remains explicitly UNKNOWN.
    if "prediction_generated_at" in x.columns:
        x["prediction_actual_lead_minutes"] = (
            x["datetime_jst"].dt.tz_convert("UTC")
            - pd.to_datetime(
                x["prediction_generated_at"], utc=True, errors="coerce"
            )
        ).dt.total_seconds() / 60.0
        x["prediction_horizon"] = x["prediction_actual_lead_minutes"].map(
            _horizon_bucket
        )
    else:
        x["prediction_actual_lead_minutes"] = np.nan
        x["prediction_horizon"] = "UNKNOWN"
    x["experience_available_at_utc"] = _now()

    return x


def _performance_bundle(frame: pd.DataFrame, compact_metrics: Any) -> dict[str, Any]:
    bundle = {
        "rows": int(len(frame)),
        "overall": compact_metrics(frame),
        "targets": _prediction_target_metrics(frame),
    }
    if "prediction_horizon" in frame.columns:
        by_horizon: dict[str, Any] = {}
        for key in ("LT_30M", "30_TO_60M", "1_TO_3H", "3_TO_6H", "GE_6H", "UNKNOWN"):
            group = frame.loc[frame["prediction_horizon"].astype(str) == key]
            if group.empty:
                continue
            by_horizon[key] = {
                "rows": int(len(group)),
                "overall": compact_metrics(group),
                "targets": _prediction_target_metrics(group),
                "mean_actual_lead_minutes": float(pd.to_numeric(
                    group["prediction_actual_lead_minutes"], errors="coerce"
                ).dropna().mean()) if "prediction_actual_lead_minutes" in group and group["prediction_actual_lead_minutes"].notna().any() else None,
            }
        bundle["by_horizon"] = by_horizon
    return bundle


def _performance_hierarchy(frame: pd.DataFrame, compact_metrics: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if frame.empty:
        return result
    for league_key, league_frame in frame.groupby("league", dropna=False):
        league = str(league_key)
        node = _performance_bundle(league_frame, compact_metrics)
        node["by_competition"] = {}
        for competition_key, competition_frame in league_frame.groupby("competition_key", dropna=False):
            competition = str(competition_key)
            comp_node = _performance_bundle(competition_frame, compact_metrics)
            comp_node["by_phase"] = {}
            for phase_key, phase_frame in competition_frame.groupby("competition_stage", dropna=False):
                comp_node["by_phase"][str(phase_key)] = _performance_bundle(
                    phase_frame, compact_metrics
                )
            node["by_competition"][competition] = comp_node
        result[league] = node
    return result


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
        # Keep the artifact contract complete even when no official result is
        # available yet. The scheduled reconciliation workflow validates this
        # index before any later training reuse.
        TRAINING_INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
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
        TRAINING_INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    scored = add_dimensions(scored)

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
        "prediction_horizon_minutes", "prediction_actual_lead_minutes", "prediction_horizon", "league", "competition_id", "competition_key",
        "competition", "competition_stage", "season_type", "game_class",
        "competition_classification_status", "home", "away", "home_starter", "away_starter",
        "revision_status", "revision_previous_prediction_id", "revision_l1_pct_points",
        "revision_max_abs_pct_points", "revision_outcome_changed",
        "regime", "score_regime", "model", "model_version", "feature_version", "calibration_version", "git_commit", "situation_tags",
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

    def _mean_or_none(series: pd.Series) -> float | None:
        values = pd.to_numeric(series, errors="coerce")
        if int(values.notna().sum()) == 0:
            return None
        return float(values.mean())

    def compact(df: pd.DataFrame) -> dict[str, float | None]:
        return {
            "rows": int(len(df)),
            "accuracy": _mean_or_none(df["outcome_correct"]),
            "logloss": _mean_or_none(df["logloss"]),
            "brier": _mean_or_none(df["brier"]),
            "draw_recall": float(
                (
                    (df["predicted_outcome"] == "DRAW")
                    & (df["actual_outcome"] == "DRAW")
                ).sum()
                / max(1, (df["actual_outcome"] == "DRAW").sum())
            ),
            "low_high_accuracy": _mean_or_none(df["low_high_correct"]),
            "top1_exact_hit_rate": _mean_or_none(df["top1_exact_hit"]),
            "top4_exact_hit_rate": _mean_or_none(df["top4_hit"]),
            "score_mae": _mean_or_none(df["score_mae"]),
            "score_evaluable_rows": int(pd.to_numeric(df["score_mae"], errors="coerce").notna().sum()),
            "top1_evaluable_rows": int(pd.to_numeric(df["top1_exact_hit"], errors="coerce").notna().sum()),
            "top4_evaluable_rows": int(pd.to_numeric(df["top4_hit"], errors="coerce").notna().sum()),
        }

    result = {
        "generated_at_utc": _now(),
        "status": "UPDATED",
        "evidence": build_experience_evidence(scored),
        "prediction_snapshot_rows_total": int(len(pred)),
        "matched_snapshot_rows": int(len(scored)),
        "unique_games_with_results": int(canonical["game_id"].nunique()),
        "experience_cases": compact(canonical),
        "all_snapshot_experience": compact(scored),
        "by_regime": {},
        "by_dominant_expert": {},
        "by_prediction_target": _prediction_target_metrics(canonical),
        "by_prediction_target_all_snapshots": _prediction_target_metrics(scored),
        "performance_breakdown": _performance_hierarchy(canonical, compact),
        "performance_breakdown_all_snapshots": _performance_hierarchy(scored, compact),
        "by_situation_tag": by_tag,
        "timing_30m": ledger._timing_30m_metrics(scored),
        "revision_intelligence": ledger._revision_metrics(scored),
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
    PERFORMANCE_BREAKDOWN_PATH.write_text(
        json.dumps({
            "schema_version": 4,
            "dimension_order": ["league", "competition", "phase", "target", "horizon"],
            "generated_at_utc": result["generated_at_utc"],
            "status": result["status"],
            "evidence": result["evidence"],
            "canonical_cases": int(len(canonical)),
            "all_prediction_snapshots": int(len(scored)),
            "league_competition_phase_target": result["performance_breakdown"],
            "all_snapshots_league_competition_phase_target": result["performance_breakdown_all_snapshots"],
            "target_metrics": result["by_prediction_target"],
            "target_metrics_all_snapshots": result["by_prediction_target_all_snapshots"],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    TRAINING_INDEX_PATH.write_text(
        json.dumps({
            "generated_at_utc": result["generated_at_utc"],
            "status": result["status"],
            "evidence": result["evidence"],
            "canonical_experience_cases": int(len(canonical)),
            "all_prediction_snapshots": int(len(scored)),
            "timing_30m": result["timing_30m"],
            "revision_intelligence": result["revision_intelligence"],
            "performance_breakdown": result["performance_breakdown"],
            "performance_breakdown_all_snapshots": result["performance_breakdown_all_snapshots"],
            "artifacts": {
                "snapshot_csv": str(SNAPSHOT_PATH),
                "snapshot_jsonl": str(SNAPSHOT_JSONL),
                "canonical_csv": str(EXPERIENCE / "experience_ledger.csv"),
                "summary_json": str(EXPERIENCE / "experience_summary.json"),
                "performance_breakdown_json": str(PERFORMANCE_BREAKDOWN_PATH),
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
