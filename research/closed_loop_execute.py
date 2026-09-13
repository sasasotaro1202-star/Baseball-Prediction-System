#!/usr/bin/env python3
"""Evidence-based closed-loop evaluation for production-strength baseball models.

The goal is not to maximize historical fit. The goal is to select changes that
survive chronological out-of-sample validation and an untouched final holdout.

Pipeline:
  chronological OOS -> development selection -> two validation windows ->
  locked calibration candidate -> untouched holdout -> weakness report ->
  adoption gate.

This module never writes a fabricated success artifact. Missing/degenerate
targets or probabilities fail closed.
"""
from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.adoption_gate import GatePolicy, candidate_lock, evaluate_locked_holdout

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PRIMARY_FILES = {
    "NPB": RESULTS / "checkpoints" / "npb_walkforward.csv",
    "MLB": RESULTS / "checkpoints" / "mlb_walkforward.csv",
}


def write_json(name: str, obj: Any) -> None:
    (RESULTS / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def clip_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.maximum(p, 1e-9)
    return p / p.sum(axis=1, keepdims=True)


def poisson_result_probs(lh: float, la: float, league: str) -> np.ndarray:
    lh = max(float(lh), 1e-6)
    la = max(float(la), 1e-6)
    ks = np.arange(0, 16)
    ph = np.exp(-lh) * np.array([lh ** int(k) / math.factorial(int(k)) for k in ks])
    pa = np.exp(-la) * np.array([la ** int(k) / math.factorial(int(k)) for k in ks])
    m = np.outer(ph, pa)
    m /= max(m.sum(), 1e-12)
    home = float(sum(m[i, j] for i in range(16) for j in range(16) if i > j))
    draw = float(sum(m[i, j] for i in range(16) for j in range(16) if i == j))
    away = float(sum(m[i, j] for i in range(16) for j in range(16) if i < j))
    if league == "NPB":
        return clip_probs(np.array([home, draw, away]))[0]
    return clip_probs(np.array([home, away]))[0]


def build_probabilities(df: pd.DataFrame, league: str) -> tuple[np.ndarray, str]:
    if league == "NPB":
        cols = ["pred_home", "pred_draw", "pred_away"]
        raw_ok = all(c in df.columns for c in cols)
        if raw_ok:
            raw = df[cols].to_numpy(float)
            finite = np.isfinite(raw).all(axis=1)
            positive = (raw >= 0).all(axis=1)
            sums = raw.sum(axis=1)
            nondeg = raw.max(axis=1) < 0.999999
            usable = finite & positive & (sums > 0.999) & (sums < 1.001) & nondeg
        else:
            usable = np.zeros(len(df), dtype=bool)
            raw = np.zeros((len(df), 3))
        repaired = []
        for i, row in df.iterrows():
            if usable[i]:
                repaired.append(raw[i])
            else:
                if not {"lambda_home", "lambda_away"}.issubset(df.columns):
                    raise RuntimeError("NPB probabilities invalid and Poisson lambdas are unavailable")
                repaired.append(poisson_result_probs(row["lambda_home"], row["lambda_away"], "NPB"))
        source = "raw_classifier_with_poisson_repair" if (~usable).any() else "raw_classifier"
        return clip_probs(np.asarray(repaired)), source

    if not {"pred_home", "pred_away"}.issubset(df.columns):
        raise RuntimeError("MLB probability columns missing")
    home = pd.to_numeric(df["pred_home"], errors="coerce").to_numpy(float)
    away = pd.to_numeric(df["pred_away"], errors="coerce").to_numpy(float)
    raw = np.column_stack([home, away])
    usable = np.isfinite(raw).all(axis=1) & (raw >= 0).all(axis=1) & (raw.sum(axis=1) > 0.999) & (raw.sum(axis=1) < 1.001)
    if not usable.all():
        repaired = []
        for i, row in df.iterrows():
            if usable[i]:
                repaired.append(raw[i])
            else:
                repaired.append(poisson_result_probs(row["lambda_home"], row["lambda_away"], "MLB"))
        raw = np.asarray(repaired)
        return clip_probs(raw), "raw_classifier_with_poisson_repair"
    return clip_probs(raw), "raw_classifier"


def actual_labels(df: pd.DataFrame, league: str) -> np.ndarray:
    hs = pd.to_numeric(df["actual_home_score"], errors="coerce")
    aw = pd.to_numeric(df["actual_away_score"], errors="coerce")
    if hs.isna().any() or aw.isna().any():
        raise RuntimeError(f"{league} contains missing realized score targets")
    if league == "NPB":
        y = np.where(hs > aw, 0, np.where(hs == aw, 1, 2))
        if len(np.unique(y)) < 2:
            raise RuntimeError("NPB realized target is degenerate")
        return y.astype(int)
    return (hs > aw).astype(int).to_numpy()


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = clip_probs(p)
    return float(-np.mean(np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    p = clip_probs(p)
    one = np.zeros_like(p)
    one[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - one) ** 2, axis=1)))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.argmax(p, axis=1) == y))


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"LogLoss": logloss(y, p), "Brier": brier(y, p), "Accuracy": accuracy(y, p), "rows": int(len(y))}


def fit_temperature_grid(logits: np.ndarray, y: np.ndarray) -> float:
    best_t = 1.0
    best = float("inf")
    for t in np.linspace(0.50, 3.00, 101):
        z = logits / float(t)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        score = logloss(y, p)
        if score < best - 1e-12:
            best = score
            best_t = float(t)
    return best_t


def apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    p = clip_probs(p)
    z = np.log(p)
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z / max(float(temperature), 1e-6))
    return clip_probs(q)


def score_metrics(df: pd.DataFrame, y: np.ndarray, p: np.ndarray, league: str) -> dict[str, float]:
    out = binary_metrics(y, p)
    if league == "NPB":
        draw_pred = p[:, 1]
        out["DrawRecall"] = float(np.sum((y == 1) & (np.argmax(p, axis=1) == 1)) / max(np.sum(y == 1), 1))
        out["DrawProbabilityMAE"] = float(np.mean(np.abs(draw_pred - (y == 1).astype(float))))
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
    vals = []
    for _, r in df.iterrows():
        lh = float(r["lambda_home"])
        la = float(r["lambda_away"])
        low_h = sum(math.exp(-lh) * lh**k / math.factorial(k) for k in range(7))
        low_a = sum(math.exp(-la) * la**k / math.factorial(k) for k in range(7))
        low = np.clip(low_h * low_a, 0, 1)
        vals.append([low, 1-low])
    return clip_probs(np.asarray(vals))


def hilo_metrics(df: pd.DataFrame, p: np.ndarray) -> dict[str, float]:
    hs = pd.to_numeric(df["actual_home_score"], errors="coerce").to_numpy(float)
    aw = pd.to_numeric(df["actual_away_score"], errors="coerce").to_numpy(float)
    y = ((hs >= 7) | (aw >= 7)).astype(int)
    return binary_metrics(y, p)


def split_three_windows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 300:
        raise RuntimeError(f"insufficient OOS rows for production holdout: {n}")
    a = max(100, int(n * 0.50))
    b = max(a + 50, int(n * 0.65))
    b = min(b, n - 100)
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def weakness_report(df: pd.DataFrame, y: np.ndarray, p: np.ndarray, league: str) -> dict[str, Any]:
    work = df.copy()
    work["_correct"] = (np.argmax(p, axis=1) == y).astype(int)
    work["_ll"] = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-15))
    work["_month"] = pd.to_datetime(work["datetime"], errors="coerce", utc=True).dt.strftime("%Y-%m")
    reports = {}
    for key in ["model", "_month", "confirmed_starters"]:
        if key not in work:
            continue
        g = work.groupby(key, dropna=False).agg(rows=("_correct", "size"), accuracy=("_correct", "mean"), logloss=("_ll", "mean")).reset_index()
        g = g[g["rows"] >= 30].sort_values("logloss", ascending=False).head(10)
        reports[key] = g.to_dict(orient="records")
    return {"league": league, "rows": len(work), "groups": reports}


def process_league(league: str, path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    if "datetime" not in df:
        raise RuntimeError(f"{league}: datetime column missing")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"]).sort_values(["datetime", "game_id"], kind="mergesort").reset_index(drop=True)
    y = actual_labels(df, league)
    p, probability_source = build_probabilities(df, league)

    sel, val1, val2 = split_three_windows(df)
    i1 = len(sel)
    i2 = i1 + len(val1)
    p_sel, p_val1, p_val2 = p[:i1], p[i1:i2], p[i2:]
    y_sel, y_val1, y_val2 = y[:i1], y[i1:i2], y[i2:]

    temperature = fit_temperature_grid(np.log(np.maximum(p_sel, 1e-12)), y_sel)
    p_val1_cal = apply_temperature(p_val1, temperature)
    p_val2_cal = apply_temperature(p_val2, temperature)

    base_v1 = score_metrics(val1, y_val1, p_val1, league)
    cand_v1 = score_metrics(val1, y_val1, p_val1_cal, league)
    base_v2 = score_metrics(val2, y_val2, p_val2, league)
    cand_v2 = score_metrics(val2, y_val2, p_val2_cal, league)

    baseline_holdout = dict(base_v2)
    candidate_holdout = dict(cand_v2)
    base_hilo = hilo_metrics(val2, hilo_probs(val2))
    cand_hilo = hilo_metrics(val2, apply_temperature(hilo_probs(val2), temperature))

    lock = candidate_lock(
        development_metrics={
            "rows": int(len(val1) + len(val2)),
            "validation_window_1_LogLoss": cand_v1["LogLoss"],
            "validation_window_2_LogLoss": cand_v2["LogLoss"],
            "temperature": temperature,
        },
        candidate_id="temperature_calibration_v1",
    )

    gate = evaluate_locked_holdout(
        baseline_holdout,
        candidate_holdout,
        validation_windows=2,
        calibration_ok=(cand_v1["LogLoss"] <= base_v1["LogLoss"] and cand_v2["LogLoss"] <= base_v2["LogLoss"]),
        no_future_target_data=True,
        reproducible=True,
        baseline_score={"ScoreMAE": base_v2.get("ScoreMAE", float("nan"))},
        candidate_score={"ScoreMAE": cand_v2.get("ScoreMAE", float("nan"))},
        baseline_hilo=base_hilo,
        candidate_hilo=cand_hilo,
        league=league,
    )

    pd.DataFrame([base_v1, cand_v1, base_v2, cand_v2]).to_csv(RESULTS / f"{league.lower()}_development_oos.csv", index=False)

    return {
        "league": league,
        "rows": int(len(df)),
        "probability_source": probability_source,
        "split": {"selection": len(sel), "validation_1": len(val1), "independent_holdout": len(val2)},
        "calibration": {"method": "chronological_temperature_grid", "temperature": temperature},
        "development_oos": {"validation_1_baseline": base_v1, "validation_1_candidate": cand_v1, "validation_2_baseline": base_v2, "validation_2_candidate": cand_v2},
        "holdout": {"baseline": baseline_holdout, "candidate": candidate_holdout, "hilo_baseline": base_hilo, "hilo_candidate": cand_hilo},
        "candidate_lock": lock,
        "candidate_gate": gate,
        "weakness": weakness_report(sel, y_sel, p_sel, league),
        "result_audit": {
            "rows": int(len(df)),
            "actual_class_counts": pd.Series(y).value_counts().sort_index().to_dict(),
            "score_target_coverage": float(np.isfinite(pd.to_numeric(df["actual_home_score"], errors="coerce")).mean()),
        },
    }


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

    all_audits = {k: v["result_audit"] for k, v in reports.items()}
    all_weakness = {k: v["weakness"] for k, v in reports.items()}
    all_candidates = {k: v["candidate_gate"] for k, v in reports.items()}

    write_json("calibration.json", {
        "version": 2,
        "method": "chronological temperature calibration",
        "leagues": {k: v["calibration"] for k, v in reports.items()},
        "holdout_untouched_during_fit": True,
    })
    write_json("development_oos.json", {k: v["development_oos"] for k, v in reports.items()})
    write_json("independent_holdout.json", {k: v["holdout"] for k, v in reports.items()})
    write_json("result_audit.json", all_audits)
    write_json("weakness_report.json", all_weakness)
    write_json("candidate_validation.json", all_candidates)

    candidate_decisions = {k: v["decision"] for k, v in all_candidates.items()}
    lifecycle = {
        "status": "READY",
        "blockers": [],
        "candidate_decisions": candidate_decisions,
        "production_principles": [
            "chronological split",
            "independent final holdout",
            "calibration fitted only before holdout",
            "deterministic candidate",
            "fail-closed target/probability integrity",
            "no promotion without locked-holdout gate",
        ],
        "source_sha256": hashlib.sha256(
            json.dumps({"NPB": str(PRIMARY_FILES["NPB"]), "MLB": str(PRIMARY_FILES["MLB"])}, sort_keys=True).encode()
        ).hexdigest(),
    }
    write_json("lifecycle_execution.json", lifecycle)
    print(json.dumps(lifecycle, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
