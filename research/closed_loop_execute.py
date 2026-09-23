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
from research.adoption_gate import GatePolicy, evaluate_locked_holdout
from research.candidates import CandidateSpec, candidate_fingerprint, lock_candidate as persist_candidate_lock
from evaluation.uncertainty import paired_block_bootstrap, to_dict as uncertainty_to_dict

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
    """Normalize a 1D probability vector or 2D probability matrix without changing rank."""
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        if not np.isfinite(p).all():
            raise RuntimeError("probability vector contains non-finite values")
        p = np.maximum(p, 1e-9)
        total = float(p.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("probability vector has non-positive sum")
        return p / total
    if p.ndim != 2 or p.shape[0] == 0:
        raise RuntimeError("probability matrix must be a non-empty 2D array")
    if not np.isfinite(p).all():
        raise RuntimeError("probability matrix contains non-finite values")
    p = np.maximum(p, 1e-9)
    sums = p.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(sums)) or np.any(sums <= 0):
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
    return clip_probs(np.asarray(values, dtype=float))


def build_probabilities(df: pd.DataFrame, league: str) -> tuple[np.ndarray, str]:
    """Accept only finite, normalized classifier probabilities.

    Invalid probability rows are evidence that the upstream OOS artifact is
    malformed or incomplete. Do not silently substitute another model because
    that would change the evaluated system and obscure the failure.
    """
    cols = ["pred_home", "pred_draw", "pred_away"] if league == "NPB" else ["pred_home", "pred_away"]
    if not set(cols).issubset(df.columns):
        raise RuntimeError(f"{league} probability columns missing: {cols}")

    raw = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    finite = np.isfinite(raw).all(axis=1)
    nonnegative = (raw >= 0).all(axis=1)
    sums = raw.sum(axis=1)
    normalized = (sums > 0.999) & (sums < 1.001)
    valid = finite & nonnegative & normalized
    if not bool(valid.all()):
        bad = int((~valid).sum())
        raise RuntimeError(
            f"{league} OOS probability artifact contains {bad} invalid rows; "
            "refusing Poisson substitution during evaluation"
        )
    return clip_probs(raw), "raw_classifier_strict"

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
    """Use the shared calibration implementation and its current default grid."""
    z = np.asarray(logits, dtype=float)
    if z.ndim != 2 or len(z) != len(y) or len(z) == 0:
        raise ValueError("logits and y are incompatible or empty")
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return float(
        fit_temperature(p, y).temperature
    )


def apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and strictly positive")
    arr = np.asarray(p, dtype=float)
    if arr.ndim != 2 or len(arr) == 0:
        raise ValueError("probability matrix must be a non-empty 2D array")
    if not np.isfinite(arr).all() or (arr < 0.0).any():
        raise ValueError("probability matrix contains invalid values")
    row_sums = arr.sum(axis=1)
    if not np.isfinite(row_sums).all() or (row_sums <= 0.0).any():
        raise ValueError("probability matrix contains non-positive row sums")
    arr = arr / row_sums[:, None]
    z = np.log(clip_probs(arr))
    z -= z.max(axis=1, keepdims=True)
    return clip_probs(np.exp(z / temperature))


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
    required = {"low", "high"}
    if not required.issubset(df.columns):
        raise RuntimeError(
            "Low/High probability columns are missing from the canonical OOS artifact"
        )
    p = df[["low", "high"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid = (
        np.isfinite(p).all(axis=1)
        & (p >= 0).all(axis=1)
        & (p.sum(axis=1) > 0.999)
        & (p.sum(axis=1) < 1.001)
    )
    if not bool(valid.all()):
        raise RuntimeError(
            f"Low/High probability artifact contains {int((~valid).sum())} invalid rows; "
            "refusing synthetic probability reconstruction during evaluation"
        )
    return clip_probs(p)

def development_candidate_id(
    df: pd.DataFrame,
    *,
    league: str,
    temperature: float,
    probability_source: str,
) -> str:
    """Deterministic candidate identity derived only from development OOS."""
    development_cols = [
        c for c in (
            "game_id", "datetime", "pred_home", "pred_draw", "pred_away",
            "actual", "model", "input_fingerprint",
        ) if c in df.columns
    ]
    payload = {
        "league": league,
        "temperature": float(temperature),
        "probability_source": probability_source,
        "development_rows": df[development_cols].to_dict(orient="records"),
    }
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return "tempcal-v2-" + hashlib.sha256(raw).hexdigest()[:20]


def hilo_metrics(df: pd.DataFrame, p: np.ndarray) -> dict[str, float]:
    hs = pd.to_numeric(df["actual_home_score"], errors="coerce").to_numpy(float)
    aw = pd.to_numeric(df["actual_away_score"], errors="coerce").to_numpy(float)
    if not np.isfinite(hs).all() or not np.isfinite(aw).all():
        raise RuntimeError("Low/High evaluation contains non-finite realized scores")
    y = ((hs + aw) >= 7).astype(int)
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
    cand_holdout_p = apply_temperature(p_holdout, temperature)
    cand_holdout = score_metrics(holdout, y_holdout, cand_holdout_p, league)
    base_hilo = hilo_metrics(holdout, hilo_probs(holdout))
    try:
        uncertainty = uncertainty_to_dict(
            paired_block_bootstrap(
                y=y_holdout,
                baseline_proba=p_holdout,
                candidate_proba=cand_holdout_p,
                block_size=30,
                replications=400,
                seed=42,
            )
        )
    except Exception as exc:
        uncertainty = {
            "status": "UNAVAILABLE",
            "reason": f"{type(exc).__name__}: {exc}",
            "rows": int(len(y_holdout)),
        }
    # Win-probability temperature is calibrated for the classifier only.
    # Do not apply it to the independent score/Low-High distribution: doing so
    # would reuse a classifier calibration parameter for a different target.
    cand_hilo = hilo_metrics(holdout, hilo_probs(holdout))

    development = pd.concat([selection, val1, val2], ignore_index=True)
    candidate_id = development_candidate_id(
        development,
        league=league,
        temperature=temperature,
        probability_source=probability_source,
    )
    development_payload = {
        "rows": int(len(development)),
        "validation_window_1_LogLoss": cand_v1["LogLoss"],
        "validation_window_2_LogLoss": cand_v2["LogLoss"],
        "temperature": temperature,
    }
    development_hash = hashlib.sha256(
        development.to_json(
            orient="split", date_format="iso", double_precision=15
        ).encode("utf-8")
    ).hexdigest()
    lock_spec = CandidateSpec(
        candidate_id=candidate_id,
        league=league,
        objective="win",
        model_version="temperature_scaled_raw",
        feature_version="closed-loop-output-v2",
        development_metrics=development_payload,
        selection_reason=(
            "Development OOS only; temperature selected on the selection window "
            "and checked on two chronological validation windows."
        ),
        git_commit=_git_commit(),
        dataset_hash=development_hash,
    )
    lock = persist_candidate_lock(lock_spec)
    starter_pit_evidence_ok = (
        "confirmed_starters" in development.columns
        and bool(development["confirmed_starters"].all())
        and "starter_evidence_status" in development.columns
        and bool((development["starter_evidence_status"] == "pit_safe").all())
    )
    holdout_pit_starter_evidence_ok = (
        "confirmed_starters" in holdout.columns
        and bool(holdout["confirmed_starters"].all())
        and "starter_evidence_status" in holdout.columns
        and bool((holdout["starter_evidence_status"] == "pit_safe").all())
    )
    lock_path = RESULTS / f"{league.lower()}_candidate_lock.json"
    lock_disk = json.loads(lock_path.read_text(encoding="utf-8"))
    reproducible = bool(
        lock_disk.get("candidate_fingerprint") == candidate_fingerprint(
            CandidateSpec(**lock_disk["candidate"])
        )
        and lock_disk.get("holdout_evaluated") is False
    )
    gate = evaluate_locked_holdout(
        base_holdout,
        cand_holdout,
        policy=GatePolicy(
            require_uncertainty_check=True,
            # MLB starter identities require authoritative pre-cutoff
            # announcement evidence. NPB may be evaluated without starter
            # features when those features are explicitly absent.
            require_pit_starter_evidence=(league == "MLB"),
        ),
        validation_windows=2,
        calibration_ok=calibration_ok,
        no_future_target_data=True,
        reproducible=reproducible,
        pit_starter_evidence_ok=starter_pit_evidence_ok,
        holdout_pit_starter_evidence_ok=holdout_pit_starter_evidence_ok,
        baseline_score={"ScoreMAE": base_holdout.get("ScoreMAE", float("nan"))},
        candidate_score={"ScoreMAE": cand_holdout.get("ScoreMAE", float("nan"))},
        baseline_hilo=base_hilo,
        candidate_hilo=cand_hilo,
        league=league,
        holdout_uncertainty=uncertainty,
    )
    return {
        "league": league,
        "rows": int(len(df)),
        "probability_source": probability_source,
        "split": {"selection": len(selection), "validation_1": len(val1), "validation_2": len(val2), "independent_holdout": len(holdout)},
        "calibration": {"method": "chronological_temperature_grid", "temperature": temperature, "fit_window": "selection_only"},
        "development_oos": {"validation_1_baseline": base_v1, "validation_1_candidate": cand_v1, "validation_2_baseline": base_v2, "validation_2_candidate": cand_v2, "calibration_ok": calibration_ok},
        "holdout": {
            "baseline": base_holdout,
            "candidate": cand_holdout,
            "hilo_baseline": base_hilo,
            "hilo_candidate": cand_hilo,
            "uncertainty": uncertainty,
            "used_for_candidate_selection": False,
        },
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
        # This workflow evaluates and locks candidates but never performs an
        # automatic production promotion. Production approval requires the
        # independent pregame PIT gate and an explicit deployment step.
        "production_approved": False,
        "promotion_status": "BLOCKED",
        "promotion_reason": "evaluation_only_no_auto_promotion",
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
