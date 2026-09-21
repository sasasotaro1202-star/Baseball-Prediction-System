#!/usr/bin/env python3
"""Leakage-safe closed-loop evaluation for production baseball models."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.calibration import fit_temperature
from core.atomic_io import atomic_write_json, file_sha256
from research.adoption_gate import candidate_lock, evaluate_locked_holdout

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
PRIMARY_FILES = {
    "NPB": RESULTS / "checkpoints" / "npb_walkforward.csv",
    "MLB": RESULTS / "checkpoints" / "mlb_walkforward.csv",
}


def write_json(name: str, obj: Any) -> None:
    # Never expose a partially-written manifest/report as a valid artifact.
    atomic_write_json(RESULTS / name, obj)


def clip_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    if not np.isfinite(p).all():
        raise RuntimeError("probability matrix contains non-finite values")
    p = np.maximum(p, 1e-9)
    sums = p.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise RuntimeError("probability row has non-positive sum")
    return p / sums


def poisson_result_probs(lh: float, la: float, league: str) -> np.ndarray:
    lh = max(float(lh), 1e-6)
    la = max(float(la), 1e-6)
    ks = np.arange(16)
    ph = np.exp(-lh) * np.array([lh ** int(k) / math.factorial(int(k)) for k in ks])
    pa = np.exp(-la) * np.array([la ** int(k) / math.factorial(int(k)) for k in ks])
    matrix = np.outer(ph, pa)
    matrix /= max(float(matrix.sum()), 1e-12)
    home = float(sum(matrix[i, j] for i in range(16) for j in range(16) if i > j))
    draw = float(sum(matrix[i, j] for i in range(16) for j in range(16) if i == j))
    away = float(sum(matrix[i, j] for i in range(16) for j in range(16) if i < j))
    values = [home, draw, away] if league == "NPB" else [home, away]
    return clip_probs(np.asarray(values))[0]


def build_probabilities(df: pd.DataFrame, league: str) -> tuple[np.ndarray, str]:
    if league == "NPB":
        cols = ["pred_home", "pred_draw", "pred_away"]
        raw = (
            df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            if all(c in df.columns for c in cols)
            else np.full((len(df), 3), np.nan)
        )
        usable = (
            np.isfinite(raw).all(axis=1)
            & (raw >= 0).all(axis=1)
            & (raw.sum(axis=1) > 0.999)
            & (raw.sum(axis=1) < 1.001)
            & (raw.max(axis=1) < 0.999999)
        )
        repaired = []
        for i, row in df.iterrows():
            if usable[i]:
                repaired.append(raw[i])
                continue
            if not np.isfinite(float(row["lambda_home"])) or not np.isfinite(float(row["lambda_away"])):
                raise RuntimeError("NPB invalid probabilities and missing Poisson lambdas")
            repaired.append(poisson_result_probs(row["lambda_home"], row["lambda_away"], "NPB"))
        return clip_probs(np.asarray(repaired)), (
            "raw_classifier_with_poisson_repair" if (~usable).any() else "raw_classifier"
        )

    cols = ["pred_home", "pred_away"]
    if not set(cols).issubset(df.columns):
        raise RuntimeError("MLB probability columns missing")
    raw = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    usable = (
        np.isfinite(raw).all(axis=1)
        & (raw >= 0).all(axis=1)
        & (raw.sum(axis=1) > 0.999)
        & (raw.sum(axis=1) < 1.001)
        & (raw.max(axis=1) < 0.999999)
    )
    if usable.all():
        return clip_probs(raw), "raw_classifier"
    repaired = []
    for i, row in df.iterrows():
        if usable[i]:
            repaired.append(raw[i])
            continue
        if not np.isfinite(float(row["lambda_home"])) or not np.isfinite(float(row["lambda_away"])):
            raise RuntimeError("MLB invalid probabilities and missing Poisson lambdas")
        repaired.append(poisson_result_probs(row["lambda_home"], row["lambda_away"], "MLB"))
    return clip_probs(np.asarray(repaired)), "raw_classifier_with_poisson_repair"


def actual_labels(df: pd.DataFrame, league: str) -> np.ndarray:
    hs = pd.to_numeric(df["actual_home_score"], errors="coerce")
    aw = pd.to_numeric(df["actual_away_score"], errors="coerce")
    if hs.isna().any() or aw.isna().any():
        raise RuntimeError(f"{league} contains missing realized score targets")
    if (~np.isfinite(hs.to_numpy()) | ~np.isfinite(aw.to_numpy())).any():
        raise RuntimeError(f"{league} contains non-finite realized score targets")
    if (hs < 0).any() or (aw < 0).any() or (hs % 1 != 0).any() or (aw % 1 != 0).any():
        raise RuntimeError(f"{league} contains invalid non-integer/non-negative realized scores")
    if league == "NPB":
        y = np.where(hs > aw, 0, np.where(hs == aw, 1, 2)).astype(int)
    else:
        y = (hs > aw).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise RuntimeError(f"{league} realized target is degenerate")
    return y


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = clip_probs(p)
    return float(-np.mean(np.log(np.maximum(p[np.arange(len(y)), y], 1e-15))))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    p = clip_probs(p)
    one = np.zeros_like(p)
    one[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - one) ** 2, axis=1)))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.argmax(p, axis=1) == y))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"LogLoss": logloss(y, p), "Brier": brier(y, p), "Accuracy": accuracy(y, p), "rows": int(len(y))}


def fit_temperature_grid(logits: np.ndarray, y: np.ndarray) -> float:
    """Use the shared calibration implementation with the legacy grid."""
    z = np.asarray(logits, dtype=float)
    if z.ndim != 2 or len(z) != len(y) or len(z) == 0:
        raise ValueError("logits and y are incompatible or empty")
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return float(
        fit_temperature(
            p,
            y,
            grid=np.linspace(0.5, 3.0, 101),
        ).temperature
    )


def apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    z = np.log(clip_probs(p))
    z -= z.max(axis=1, keepdims=True)
    return clip_probs(np.exp(z / max(float(temperature), 1e-6)))


def score_metrics(df: pd.DataFrame, y: np.ndarray, p: np.ndarray, league: str) -> dict[str, float]:
    out = metrics(y, p)
    if league == "NPB":
        out["DrawRecall"] = float(np.sum((y == 1) & (np.argmax(p, axis=1) == 1)) / max(np.sum(y == 1), 1))
        out["DrawProbabilityMAE"] = float(np.mean(np.abs(p[:, 1] - (y == 1).astype(float))) )
    if {"lambda_home", "lambda_away"}.issubset(df.columns):
        lh = pd.to_numeric(df["lambda_home"], errors="coerce").to_numpy(float)
        la = pd.to_numeric(df["lambda_away"], errors="coerce").to_numpy(float)
        hs = pd.to_numeric(df["actual_home_score"], errors="coerce").to_numpy(float)
        aw = pd.to_numeric(df["actual_away_score"], errors="coerce").to_numpy(float)
        valid = np.isfinite(lh) & np.isfinite(la) & np.isfinite(hs) & np.isfinite(aw)
        if valid.any():
            out["ScoreMAE"] = float(np.mean((np.abs(lh[valid] - hs[valid]) + np.abs(la[valid] - aw[valid])) / 2.0))
    return out


def hilo_probs(df: pd.DataFrame) -> np.ndarray:
    if {"low", "high"}.issubset(df.columns):
        p = df[["low", "high"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        valid = np.isfinite(p).all(axis=1) & (p >= 0).all(axis=1) & (p.sum(axis=1) > 0.999) & (p.sum(axis=1) < 1.001)
        if valid.all():
            return clip_probs(p)
    values = []
    for _, row in df.iterrows():
        lh = max(float(row["lambda_home"]), 1e-9)
        la = max(float(row["lambda_away"]), 1e-9)
        # LOW/HIGH is defined on total runs, not on two independent per-team
        # thresholds. Compute P(H+A <= 6) from the full joint distribution so
        # the evaluation contract matches the production score distribution.
        ph = np.asarray([math.exp(-lh) * lh**k / math.factorial(k) for k in range(16)], dtype=float)
        pa = np.asarray([math.exp(-la) * la**k / math.factorial(k) for k in range(16)], dtype=float)
        matrix = np.outer(ph, pa)
        matrix /= max(float(matrix.sum()), 1e-12)
        low = float(np.clip(sum(matrix[i, j] for i in range(16) for j in range(16) if i + j <= 6), 0.0, 1.0))
        values.append([low, 1.0 - low])
    return clip_probs(np.asarray(values))


def hilo_metrics(df: pd.DataFrame, p: np.ndarray) -> dict[str, float]:
    hs = pd.to_numeric(df["actual_home_score"], errors="coerce").to_numpy(float)
    aw = pd.to_numeric(df["actual_away_score"], errors="coerce").to_numpy(float)
    y = ((hs >= 7) | (aw >= 7)).astype(int)
    return metrics(y, p)


def weakness_report(df: pd.DataFrame, y: np.ndarray, p: np.ndarray, league: str) -> dict[str, Any]:
    work = df.copy()
    work["_correct"] = (np.argmax(p, axis=1) == y).astype(int)
    work["_ll"] = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-15))
    work["_month"] = pd.to_datetime(work["datetime"], errors="coerce", utc=True).dt.strftime("%Y-%m")
    groups = {}
    for key in ["model", "_month", "confirmed_starters"]:
        if key not in work:
            continue
        grouped = work.groupby(key, dropna=False).agg(rows=("_correct", "size"), accuracy=("_correct", "mean"), logloss=("_ll", "mean")).reset_index()
        groups[key] = grouped[grouped["rows"] >= 30].sort_values("logloss", ascending=False).head(10).to_dict(orient="records")
    return {"league": league, "rows": len(work), "groups": groups}


def split_four_windows(df: pd.DataFrame):
    n = len(df)
    if n < 500:
        raise RuntimeError(f"insufficient chronological OOS rows: {n}")
    selection_end = int(n * 0.45)
    validation1_end = int(n * 0.65)
    validation2_end = int(n * 0.80)
    if selection_end < 150 or validation1_end - selection_end < 75:
        raise RuntimeError("selection/validation1 window too small")
    if validation2_end - validation1_end < 75 or n - validation2_end < 100:
        raise RuntimeError("validation2/holdout window too small")
    return (
        df.iloc[:selection_end].copy(),
        df.iloc[selection_end:validation1_end].copy(),
        df.iloc[validation1_end:validation2_end].copy(),
        df.iloc[validation2_end:].copy(),
    )


def process_league(league: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"{league} walkforward file missing: {path}")
    df = pd.read_csv(path)
    if "datetime" not in df.columns or "game_id" not in df.columns:
        raise RuntimeError(f"{league} walkforward identity columns missing")
    if df["game_id"].isna().any() or df["game_id"].astype(str).str.strip().eq("").any():
        raise RuntimeError(f"{league} walkforward contains missing game_id")
    if df["game_id"].duplicated().any():
        raise RuntimeError(f"{league} walkforward contains duplicate game_id")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    if df["datetime"].isna().any():
        raise RuntimeError(f"{league} walkforward contains invalid datetime")
    df = df.sort_values(["datetime", "game_id"], kind="mergesort").reset_index(drop=True)
    y = actual_labels(df, league)
    p, probability_source = build_probabilities(df, league)

    selection, val1, val2, holdout = split_four_windows(df)
    n1, n2, n3 = len(selection), len(selection) + len(val1), len(selection) + len(val1) + len(val2)
    p_selection, p_val1, p_val2, p_holdout = p[:n1], p[n1:n2], p[n2:n3], p[n3:]
    y_selection, y_val1, y_val2, y_holdout = y[:n1], y[n1:n2], y[n2:n3], y[n3:]

    # Fail closed when any calibration/evaluation window is class-degenerate.
    # A globally non-degenerate dataset is not enough: temperature scaling and
    # LogLoss/Brier comparisons become unreliable when a chronological window
    # contains only one realized outcome class. This also prevents a hidden
    # calendar/regime shift from being silently treated as valid evidence.
    windows = {
        "selection": y_selection,
        "validation_1": y_val1,
        "validation_2": y_val2,
        "independent_holdout": y_holdout,
    }
    for window_name, labels in windows.items():
        if len(labels) < 2 or len(np.unique(labels)) < 2:
            raise RuntimeError(
                f"{league} {window_name} realized target is class-degenerate; "
                "refusing calibration/selection/holdout evaluation."
            )
        if not np.isfinite(labels).all():
            raise RuntimeError(f"{league} {window_name} realized target contains non-finite labels")

    # Enforce chronological ordering explicitly after the deterministic sort.
    # This is redundant under normal operation but makes the PIT/OOS contract
    # fail closed if an upstream artifact is ever malformed.
    if not df["datetime"].is_monotonic_increasing:
        raise RuntimeError(f"{league} walkforward chronology is not monotonic")
    temperature = fit_temperature_grid(np.log(np.maximum(p_selection, 1e-12)), y_selection)
    base_v1 = score_metrics(val1, y_val1, p_val1, league)
    cand_v1 = score_metrics(val1, y_val1, apply_temperature(p_val1, temperature), league)
    base_v2 = score_metrics(val2, y_val2, p_val2, league)
    cand_v2 = score_metrics(val2, y_val2, apply_temperature(p_val2, temperature), league)
    calibration_ok = cand_v1["LogLoss"] <= base_v1["LogLoss"] and cand_v2["LogLoss"] <= base_v2["LogLoss"]

    base_holdout = score_metrics(holdout, y_holdout, p_holdout, league)
    cand_holdout = score_metrics(holdout, y_holdout, apply_temperature(p_holdout, temperature), league)
    base_hilo = hilo_metrics(holdout, hilo_probs(holdout))
    cand_hilo = hilo_metrics(holdout, apply_temperature(hilo_probs(holdout), temperature))

    lock = candidate_lock(
        development_metrics={"rows": int(len(selection) + len(val1) + len(val2)), "validation_window_1_LogLoss": cand_v1["LogLoss"], "validation_window_2_LogLoss": cand_v2["LogLoss"], "temperature": temperature},
        candidate_id="temperature_calibration_v2",
    )
    gate = evaluate_locked_holdout(
        base_holdout,
        cand_holdout,
        validation_windows=2,
        calibration_ok=calibration_ok,
        no_future_target_data=True,
        reproducible=True,
        baseline_score={"ScoreMAE": base_holdout.get("ScoreMAE", float("nan"))},
        candidate_score={"ScoreMAE": cand_holdout.get("ScoreMAE", float("nan"))},
        baseline_hilo=base_hilo,
        candidate_hilo=cand_hilo,
        league=league,
    )
    return {
        "league": league,
        "rows": int(len(df)),
        "probability_source": probability_source,
        "split": {"selection": len(selection), "validation_1": len(val1), "validation_2": len(val2), "independent_holdout": len(holdout)},
        "calibration": {"method": "chronological_temperature_grid", "temperature": temperature, "fit_window": "selection_only"},
        "development_oos": {"validation_1_baseline": base_v1, "validation_1_candidate": cand_v1, "validation_2_baseline": base_v2, "validation_2_candidate": cand_v2, "calibration_ok": calibration_ok},
        "holdout": {"baseline": base_holdout, "candidate": cand_holdout, "hilo_baseline": base_hilo, "hilo_candidate": cand_hilo, "used_for_candidate_selection": False},
        "candidate_lock": lock,
        "candidate_gate": gate,
        "weakness": weakness_report(selection, y_selection, p_selection, league),
        "result_audit": {"rows": int(len(df)), "actual_class_counts": pd.Series(y).value_counts().sort_index().to_dict(), "score_target_coverage": float(np.isfinite(pd.to_numeric(df["actual_home_score"], errors="coerce")).mean())},
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("GITHUB_SHA", "UNKNOWN")


def main() -> int:
    reports = {}
    blockers = []
    for league, path in PRIMARY_FILES.items():
        try:
            reports[league] = process_league(league, path)
        except Exception as exc:
            blockers.append(f"{league}:{type(exc).__name__}:{exc}")
    if blockers:
        write_json("lifecycle_execution.json", {"status": "BLOCKED", "blockers": blockers, "reports": reports})
        raise SystemExit("; ".join(blockers))

    write_json("calibration.json", {"version": 3, "method": "chronological temperature calibration", "leagues": {k: v["calibration"] for k, v in reports.items()}, "holdout_untouched_during_fit_and_selection": True})
    write_json("development_oos.json", {k: v["development_oos"] for k, v in reports.items()})
    write_json("independent_holdout.json", {k: v["holdout"] for k, v in reports.items()})
    write_json("result_audit.json", {k: v["result_audit"] for k, v in reports.items()})
    write_json("weakness_report.json", {k: v["weakness"] for k, v in reports.items()})
    write_json("candidate_validation.json", {k: v["candidate_gate"] for k, v in reports.items()})

    decisions = {k: v["candidate_gate"]["decision"] for k, v in reports.items()}
    lifecycle = {
        "status": "READY",
        "blockers": [],
        "candidate_decisions": decisions,
        "production_principles": [
            "chronological split",
            "four-window development/validation/holdout separation",
            "calibration fitted only on selection",
            "candidate selected only on validation windows",
            "independent final holdout",
            "deterministic candidate",
            "fail-closed target/probability integrity",
            "no promotion without locked-holdout gate",
        ],
        "source_fingerprints": {k: file_sha256(v) for k, v in PRIMARY_FILES.items()},
        "git_commit": _git_commit(),
    }
    write_json("lifecycle_execution.json", lifecycle)
    print(json.dumps(lifecycle, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
