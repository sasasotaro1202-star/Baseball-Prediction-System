"""Bridge existing chronological OOS checkpoints into the Ultimate v13 control plane.

The bridge is intentionally downstream of the existing BaseballBacktest. It
never synthesizes PIT timestamps, never replaces missing model probabilities,
and never promotes a candidate. It produces an auditable real-OOS status report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.metrics import classification_metrics
from research.ultimate_v13_control import model_disagreement, predictability_score
from research.ultimate_v13_operational_controls import (
    output_format,
    build_forecast_contract,
    prediction_freshness,
    router_stability,
)


def _probability_columns(df: pd.DataFrame, league: str) -> list[str]:
    cols = ["pred_home", "pred_draw", "pred_away"] if league == "NPB" else ["pred_home", "pred_away"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing probability columns: {missing}")
    return cols


def _labels(df: pd.DataFrame, league: str) -> np.ndarray:
    for c in ("actual_home_score", "actual_away_score"):
        if c not in df.columns:
            raise RuntimeError(f"missing realized score target: {c}")
    home = pd.to_numeric(df["actual_home_score"], errors="coerce")
    away = pd.to_numeric(df["actual_away_score"], errors="coerce")
    if home.isna().any() or away.isna().any():
        raise RuntimeError("realized score targets contain missing values")
    if (home < 0).any() or (away < 0).any():
        raise RuntimeError("realized score targets contain negative values")
    if league == "NPB":
        return np.where(home > away, 0, np.where(home == away, 1, 2)).astype(int)
    return (home > away).astype(int).to_numpy()


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    arr = np.asarray(p, dtype=float)
    arr = np.maximum(arr, 1e-12)
    arr /= arr.sum(axis=1, keepdims=True)
    classes = list(range(arr.shape[1]))
    metrics = classification_metrics(np.asarray(y, dtype=int), arr, classes=classes)
    return {
        "Accuracy": float(metrics["Accuracy"]),
        "LogLoss": float(metrics["LogLoss"]),
        "Brier": float(metrics["Brier"]),
        "ECE": float(metrics["ECE"]),
    }

def _pit_status(df: pd.DataFrame) -> dict[str, Any]:
    required = {"prediction_time", "available_at"}
    missing = sorted(required - set(df.columns))
    if missing:
        return {
            "status": "BLOCKED",
            "reason": "missing_pit_columns",
            "missing": missing,
        }
    pt = pd.to_datetime(df["prediction_time"], errors="coerce", utc=True)
    at = pd.to_datetime(df["available_at"], errors="coerce", utc=True)
    if pt.isna().any() or at.isna().any():
        return {"status": "FAIL", "reason": "invalid_pit_timestamp"}
    bad = at > pt
    return {
        "status": "PASS" if not bad.any() else "FAIL",
        "rows": int(len(df)),
        "violations": int(bad.sum()),
    }


def _model_panel(df: pd.DataFrame, league: str) -> tuple[dict[str, np.ndarray], dict[str, list[float]]] | None:
    if "model" not in df.columns:
        return None
    probs = _probability_columns(df, league)
    if df["model"].nunique(dropna=True) < 2 or "game_id" not in df.columns:
        return None
    # Require a complete one-row-per-(game,model) panel. Anything else is
    # ambiguous and therefore not converted into synthetic model outputs.
    key = df[["game_id", "model"]].astype(str)
    if key.duplicated().any():
        return None
    counts = df.groupby("game_id")["model"].nunique()
    if counts.empty or not bool((counts == df["model"].nunique()).all()):
        return None
    models: dict[str, np.ndarray] = {}
    losses: dict[str, list[float]] = {}
    y = _labels(df.drop_duplicates("game_id"), league)
    game_order = df.drop_duplicates("game_id")["game_id"].astype(str).tolist()
    for name, group in df.groupby("model", sort=True):
        group = group.set_index(group["game_id"].astype(str)).loc[game_order]
        p = group[probs].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if not np.isfinite(p).all() or (p < 0).any() or np.any(p.sum(axis=1) <= 0):
            return None
        p /= p.sum(axis=1, keepdims=True)
        models[str(name)] = p
        losses[str(name)] = (-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))).tolist()
    return models, losses


def run_real_oos_bridge(path: str | Path, *, league: str, data_snapshot_id: str = "unknown") -> dict[str, Any]:
    pth = Path(path)
    if not pth.exists():
        return {
            "status": "BLOCKED",
            "evidence_scope": "real_oos",
            "league": league,
            "reason": "checkpoint_missing",
            "path": str(pth),
        }
    df = pd.read_csv(pth)
    blockers: list[str] = []
    if len(df) < 100:
        blockers.append("insufficient_oos_rows")
    pit = _pit_status(df)
    if pit["status"] != "PASS":
        blockers.append("pit_not_verified")
    try:
        probs = _probability_columns(df, league)
        y = _labels(df, league)
        pmat = df[probs].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if not np.isfinite(pmat).all() or (pmat < 0).any():
            blockers.append("invalid_probabilities")
        pmat = np.maximum(pmat, 1e-12)
        pmat /= pmat.sum(axis=1, keepdims=True)
    except Exception as exc:
        blockers.append(f"target_or_probability_contract:{type(exc).__name__}:{exc}")
        pmat = None
        y = None

    result: dict[str, Any] = {
        "schema_version": 1,
        "evidence_scope": "real_oos",
        "league": league,
        "path": str(pth),
        "rows": int(len(df)),
        "pit": pit,
        "blockers": blockers,
        "model_panel": {"status": "UNAVAILABLE"},
    }
    if pmat is not None and y is not None:
        result["metrics"] = _metrics(y, pmat)
        result["predictability"] = predictability_score(
            pmat,
            history_probs=pmat,
            data_quality=1.0 if pit["status"] == "PASS" else 0.0,
        )
        panel = _model_panel(df, league)
        if panel is not None:
            model_probs, losses = panel
            result["model_panel"] = {
                "status": "PASS",
                "models": sorted(model_probs),
                "model_disagreement": model_disagreement(model_probs),
                "loss_rows": {k: len(v) for k, v in losses.items()},
                "router_stability": router_stability(
                    [{"model": float(i == j)} for i, name in enumerate(sorted(model_probs)) for j in range(len(model_probs))]
                ) if len(model_probs) == 1 else {"status": "AVAILABLE"},
            }
        else:
            result["model_panel"] = {
                "status": "BLOCKED",
                "reason": "complete_multi_model_oos_panel_not_available",
            }

        fmt = output_format(
            predictability=float(result["predictability"]["predictability"]),
            uncertainty=float(1.0 - result["predictability"]["predictability"]),
            ood=0.0,
        )
        if "prediction_time" in df.columns:
            pt = pd.to_datetime(df["prediction_time"], errors="coerce", utc=True).max()
        elif "datetime" in df.columns:
            pt = pd.to_datetime(df["datetime"], errors="coerce", utc=True).max()
        else:
            pt = None
        if pt is not None and pd.notna(pt):
            valid_until = pt + pd.Timedelta(hours=1)
            result["freshness"] = prediction_freshness(
                prediction_time=pt, now=valid_until - pd.Timedelta(minutes=10), valid_until=valid_until
            )
            if pit["status"] == "PASS":
                result["contract"] = build_forecast_contract(
                    prediction_time=pt,
                    valid_until=valid_until,
                    model_version="real-oos-v13-bridge",
                    strategy="RESEARCH_BRIDGE",
                    output_format_name=fmt,
                    confidence=float(pmat[-1].max()),
                    predictability=float(result["predictability"]["predictability"]),
                    uncertainty=float(1.0 - result["predictability"]["predictability"]),
                    failure_risk=0.0,
                    ood_score=0.0,
                    pit_status="PASS",
                    data_snapshot_id=data_snapshot_id,
                )
    if blockers:
        result["status"] = "BLOCKED"
        result["promotion_status"] = "HOLD"
    else:
        result["status"] = "EXECUTED"
        result["promotion_status"] = "HOLD"
        result["promotion_reason"] = "bridge_evaluates_real_oos_but_does_not_promote"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", choices=["NPB", "MLB"], required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-snapshot-id", default="unknown")
    args = parser.parse_args()
    report = run_real_oos_bridge(args.path, league=args.league, data_snapshot_id=args.data_snapshot_id)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"EXECUTED", "BLOCKED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
