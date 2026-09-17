#!/usr/bin/env python3
"""Offline, PIT-gated evaluation harness for X-derived candidate features.

This module never edits the production model. It evaluates a precomputed
candidate probability column against the production baseline using only rows
whose X feature availability is demonstrably <= prediction_cutoff.

Expected CSV columns:
  y_true, baseline_prob, candidate_prob, prediction_cutoff,
  feature_available_at
Optional: season, league, team, data_quality

A row with missing/invalid timestamps is excluded rather than guessed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _probs(values: pd.Series) -> np.ndarray:
    p = pd.to_numeric(values, errors="coerce").to_numpy(float)
    if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError("probabilities must be finite and strictly between 0 and 1")
    return p


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    ll = -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    acc = float(np.mean((p >= 0.5).astype(int) == y))
    return {"rows": int(len(y)), "LogLoss": ll, "Brier": brier, "Accuracy": acc}


def evaluate(path: Path) -> dict:
    df = pd.read_csv(path)
    required = {"y_true", "baseline_prob", "candidate_prob", "prediction_cutoff", "feature_available_at"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    cutoff = pd.to_datetime(df["prediction_cutoff"], errors="coerce", utc=True)
    available = pd.to_datetime(df["feature_available_at"], errors="coerce", utc=True)
    pit_safe = cutoff.notna() & available.notna() & (available <= cutoff)
    dropped = int((~pit_safe).sum())
    work = df.loc[pit_safe].copy()
    if len(work) < 200:
        raise ValueError(f"insufficient PIT-safe evaluation rows: {len(work)}")

    y = pd.to_numeric(work["y_true"], errors="coerce").to_numpy(float)
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only binary 0/1 labels")
    y = y.astype(int)
    baseline = _probs(work["baseline_prob"])
    candidate = _probs(work["candidate_prob"])

    result = {
        "schema_version": 1,
        "rows_input": int(len(df)),
        "rows_pit_safe": int(len(work)),
        "rows_dropped_pit": dropped,
        "baseline": binary_metrics(y, baseline),
        "candidate": binary_metrics(y, candidate),
        "improvement": {
            "LogLoss": float(binary_metrics(y, baseline)["LogLoss"] - binary_metrics(y, candidate)["LogLoss"]),
            "Brier": float(binary_metrics(y, baseline)["Brier"] - binary_metrics(y, candidate)["Brier"]),
            "Accuracy": float(binary_metrics(y, candidate)["Accuracy"] - binary_metrics(y, baseline)["Accuracy"]),
        },
        "status": "EVALUATED",
        "promotion": "NOT_APPLICABLE",
        "note": "This is an offline research comparison only; no production feature/model is modified.",
    }
    if "season" in work.columns:
        result["by_season"] = {}
        for season, g in work.groupby("season", dropna=False):
            yy = pd.to_numeric(g["y_true"], errors="coerce").to_numpy(float)
            bp = _probs(g["baseline_prob"])
            cp = _probs(g["candidate_prob"])
            if len(g) >= 30 and np.isfinite(yy).all() and np.isin(yy, [0, 1]).all():
                result["by_season"][str(season)] = {
                    "baseline": binary_metrics(yy.astype(int), bp),
                    "candidate": binary_metrics(yy.astype(int), cp),
                }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--output", default="results/x_offline_eval.json")
    args = parser.parse_args()
    result = evaluate(Path(args.csv))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
