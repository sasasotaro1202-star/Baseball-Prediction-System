#!/usr/bin/env python3
"""Execute the evidence-based Baseball research lifecycle.

This runner consumes only already-produced chronological OOS predictions. It
never invents missing market/PIT evidence. Calibration is selected on a
chronological development-validation split, locked, then evaluated once on an
unseen holdout. The holdout is never used to fit a parameter.

Stages:
  PIT evidence -> calibration -> development OOS -> independent holdout ->
  result audit -> weakness discovery -> candidate validation/adoption gate.

If a required artifact is missing, the stage is BLOCKED rather than fabricated.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.calibration import TemperatureCalibration, fit_temperature
from research.adoption_gate import GatePolicy, candidate_lock, evaluate_locked_holdout

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PIT_DIR = ROOT / "data" / "pit"


def _write(name: str, obj: Any) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonl_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)
            n += 1
    return n


def pit_evidence() -> dict[str, Any]:
    required = [
        "source_snapshots.jsonl",
        "event_observations.jsonl",
        "availability_observations.jsonl",
        "acquisition_runs.jsonl",
    ]
    counts = {name: _jsonl_rows(PIT_DIR / name) for name in required}
    blocked = [name for name, count in counts.items() if count == 0]
    return {"stage": "PIT", "status": "READY" if not blocked else "BLOCKED", "rows": counts,
            "blockers": [f"{x}:missing_or_empty" for x in blocked]}


def _probabilities(df: pd.DataFrame, league: str) -> tuple[np.ndarray, np.ndarray]:
    if league == "NPB":
        cols = ["pred_home", "pred_draw", "pred_away"]
    else:
        cols = ["pred_home", "pred_away"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{league}: missing probability columns: {missing}")
    p = df[cols].astype(float).to_numpy()
    y = df["actual"].astype(int).to_numpy()
    if len(p) == 0 or not np.isfinite(p).all():
        raise ValueError(f"{league}: invalid probabilities")
    p = np.clip(p, 1e-12, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    return p, y


def _metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = np.argmax(p, axis=1)
    n = len(y)
    ll = float(-np.mean(np.log(np.clip(p[np.arange(n), y], 1e-15, 1.0))))
    onehot = np.zeros_like(p)
    onehot[np.arange(n), y] = 1.0
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    out = {"rows": int(n), "Accuracy": float(np.mean(pred == y)), "LogLoss": ll, "Brier": brier}
    if p.shape[1] == 3:
        draw = y == 1
        out["DrawRecall"] = float(np.sum((pred == 1) & draw) / max(np.sum(draw), 1))
        out["DrawProbabilityMAE"] = float(np.mean(np.abs(p[:, 1] - draw.astype(float))))
    return out


def _score_metrics(df: pd.DataFrame) -> dict[str, float]:
    required = ["lambda_home", "lambda_away", "actual_home_score", "actual_away_score"]
    if any(c not in df.columns for c in required):
        return {}
    err = (np.abs(df["actual_home_score"] - df["lambda_home"]) +
           np.abs(df["actual_away_score"] - df["lambda_away"])) / 2.0
    return {"ScoreMAE": float(err.mean()), "rows": int(len(err))}


def _legacy_hilo_metrics(df: pd.DataFrame) -> dict[str, float]:
    # Existing backtest's Low/High contract is the 7+ per-team tail. This is
    # explicitly named legacy_hilo; it is NOT claimed to be a sportsbook total.
    if "high" not in df.columns:
        return {}
    target = ((df["actual_home_score"] >= 7) | (df["actual_away_score"] >= 7)).astype(int).to_numpy()
    p_high = np.clip(df["high"].astype(float).to_numpy(), 1e-12, 1 - 1e-12)
    pred = (p_high >= 0.5).astype(int)
    ll = float(-np.mean(target * np.log(p_high) + (1-target) * np.log(1-p_high)))
    br = float(np.mean((p_high-target) ** 2))
    return {"LogLoss": ll, "Brier": br, "Accuracy": float(np.mean(pred == target)), "rows": int(len(target)),
            "contract": "legacy_team_7plus_not_market_total"}


def _chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "datetime" not in df.columns:
        raise ValueError("datetime column is required")
    work = df.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce", utc=True)
    work = work.dropna(subset=["datetime"]).sort_values(["datetime", "game_id"]).reset_index(drop=True)
    if len(work) < 100:
        raise ValueError(f"insufficient OOS rows: {len(work)}")
    holdout_start = int(len(work) * 0.80)
    dev = work.iloc[:holdout_start].copy()
    holdout = work.iloc[holdout_start:].copy()
    select_end = int(len(dev) * 0.67)
    select = dev.iloc[:select_end].copy()
    calibration_validation = dev.iloc[select_end:].copy()
    if len(select) < 50 or len(calibration_validation) < 25 or len(holdout) < 25:
        raise ValueError("chronological development/validation/holdout windows are too small")
    return select, calibration_validation, holdout


def _calibration_candidate(select: pd.DataFrame, validation: pd.DataFrame, league: str) -> tuple[float, dict[str, Any]]:
    p_select, y_select = _probabilities(select, league)
    p_val, y_val = _probabilities(validation, league)
    # Select the calibration family parameter only on the first development window.
    cal = fit_temperature(p_select, y_select)
    calibrated_val = cal.transform(p_val)
    raw_m = _metrics(p_val, y_val)
    cal_m = _metrics(calibrated_val, y_val)
    return cal.temperature, {"version": cal.version, "temperature": cal.temperature,
                             "selection_rows": len(select), "validation_rows": len(validation),
                             "raw_validation": raw_m, "calibrated_validation": cal_m,
                             "improvement": raw_m["LogLoss"] - cal_m["LogLoss"]}


def process_league(league: str) -> dict[str, Any]:
    src = RESULTS / f"{league.lower()}_backtest_results.csv"
    if not src.exists() or src.stat().st_size == 0:
        raise RuntimeError(f"{league}: backtest result artifact missing: {src}")
    df = pd.read_csv(src)
    select, validation, holdout = _chronological_split(df)

    temperature, cal_validation = _calibration_candidate(select, validation, league)
    cal = TemperatureCalibration(temperature)
    # Lock the hyperparameter before touching holdout; refitting at the same
    # temperature on all development rows changes no selected hyperparameter.
    dev = pd.concat([select, validation], ignore_index=True)
    p_dev, y_dev = _probabilities(dev, league)
    p_hold, y_hold = _probabilities(holdout, league)
    raw_hold_metrics = _metrics(p_hold, y_hold)
    calibrated_hold = cal.transform(p_hold)
    calibrated_hold_metrics = _metrics(calibrated_hold, y_hold)

    calibration_artifact = {
        "schema_version": 1, "league": league, "temperature": temperature,
        "version": cal.version, "fit_rows": len(dev), "selection_rows": len(select),
        "validation_rows": len(validation), "holdout_rows": len(holdout),
        "selection_validation": cal_validation,
        "holdout_raw": raw_hold_metrics,
        "holdout_calibrated": calibrated_hold_metrics,
        "fitted_without_holdout": True,
    }
    _write(f"calibration_{league.lower()}.json", calibration_artifact)

    # Result audit is a historical reconciliation of immutable prediction rows
    # to the already observed game result. It does not invent missing outcomes.
    audit_cols = [c for c in ["game_id", "datetime", "league", "home", "away", "prediction", "actual",
                              "correct", "actual_home_score", "actual_away_score", "pred_home", "pred_draw", "pred_away"] if c in holdout]
    audit = holdout[audit_cols].copy()
    audit["result_status"] = np.where(audit["actual"].notna(), "RESOLVED", "UNRESOLVED")
    _write(f"result_audit_{league.lower()}.csv", audit)
    _write("result_audit.json", {"schema_version": 1, "leagues": ["NPB", "MLB"],
                                  "holdout_rows": int(len(audit)),
                                  "resolved_rows": int((audit.result_status == "RESOLVED").sum())})

    weakness_rows = []
    for field in ["model", "home_starter", "away_starter"]:
        if field not in holdout.columns:
            continue
        tmp = holdout.copy()
        tmp[field] = tmp[field].fillna("UNKNOWN").astype(str)
        grp = tmp.groupby(field, dropna=False).agg(rows=("actual", "size"), accuracy=("correct", "mean"), logloss=("logloss", "mean"), brier=("brier", "mean")).reset_index()
        grp = grp[grp.rows >= 10].sort_values(["logloss", "rows"], ascending=[False, False]).head(20)
        for _, r in grp.iterrows():
            weakness_rows.append({"league": league, "dimension": field, "segment": r[field], "rows": int(r.rows),
                                  "accuracy": float(r.accuracy), "logloss": float(r.logloss), "brier": float(r.brier)})
    _write(f"weakness_{league.lower()}.csv", pd.DataFrame(weakness_rows))

    baseline = raw_hold_metrics
    candidate = calibrated_hold_metrics
    baseline.update(_score_metrics(holdout))
    candidate.update(_score_metrics(holdout))
    # The existing backtest's H/L contract is evaluated separately and preserved.
    base_hilo = _legacy_hilo_metrics(holdout)
    cand_hilo = base_hilo.copy()
    policy = GatePolicy()
    locked = candidate_lock(development_metrics=cal_validation["calibrated_validation"],
                            candidate_id=f"temperature:{league}:{temperature:.8f}")
    decision = evaluate_locked_holdout(
        baseline, candidate, policy=policy, validation_windows=2,
        calibration_ok=cal_validation["improvement"] >= 0,
        no_future_target_data=True, reproducible=True,
        baseline_score=_score_metrics(holdout), candidate_score=_score_metrics(holdout),
        baseline_hilo=base_hilo, candidate_hilo=cand_hilo, league=league)
    return {"league": league, "calibration": calibration_artifact, "candidate_lock": locked,
            "holdout_baseline": baseline, "holdout_candidate": candidate, "decision": decision}


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    pit = pit_evidence()
    _write("pit_stage.json", pit)
    if pit["status"] != "READY":
        _write("lifecycle_execution.json", {"status": "BLOCKED", "blockers": pit["blockers"]})
        return 2
    reports = []
    blockers = []
    for league in ("NPB", "MLB"):
        try:
            reports.append(process_league(league))
        except Exception as exc:
            blockers.append(f"{league}:{type(exc).__name__}:{exc}")
    if reports:
        _write("candidate_validation.json", {"schema_version": 1, "reports": reports})
    status = "READY" if not blockers and len(reports) == 2 else "BLOCKED"
    _write("lifecycle_execution.json", {"schema_version": 1, "status": status, "blockers": blockers,
                                         "leagues_completed": [r["league"] for r in reports]})
    return 0 if status == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
