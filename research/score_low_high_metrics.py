#!/usr/bin/env python3
"""Leakage-safe evaluation helpers for baseball score and Low/High predictions.

The functions are deliberately model-agnostic: they consume an already-generated
OOS prediction frame and never fit on the evaluation targets. They are intended to
make score and Low/High quality first-class research metrics alongside win metrics.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _require(frame: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def score_metrics(
    frame: pd.DataFrame,
    *,
    predicted_home_score: str = "pred_home_score",
    predicted_away_score: str = "pred_away_score",
    actual_home_score: str = "actual_home_score",
    actual_away_score: str = "actual_away_score",
    top_score_candidates: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate continuous score forecasts and optional top-N exact-score hits."""
    _require(frame, [predicted_home_score, predicted_away_score, actual_home_score, actual_away_score])
    x = frame[[predicted_home_score, predicted_away_score, actual_home_score, actual_away_score]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if x.empty:
        raise ValueError("no valid score rows")
    ph, pa = x[predicted_home_score].to_numpy(float), x[predicted_away_score].to_numpy(float)
    ah, aa = x[actual_home_score].to_numpy(float), x[actual_away_score].to_numpy(float)
    abs_home = np.abs(ph - ah)
    abs_away = np.abs(pa - aa)
    total_pred, total_actual = ph + pa, ah + aa
    out = {
        "rows": float(len(x)),
        "mae_home_score": float(abs_home.mean()),
        "mae_away_score": float(abs_away.mean()),
        "mae_total_score": float(np.abs(total_pred - total_actual).mean()),
        "rmse_total_score": float(np.sqrt(np.mean((total_pred - total_actual) ** 2))),
        "exact_score_rate": float(((np.rint(ph) == ah) & (np.rint(pa) == aa)).mean()),
        "within_one_each_rate": float(((abs_home <= 1) & (abs_away <= 1)).mean()),
    }
    if top_score_candidates and top_score_candidates in frame.columns:
        hits = 0
        usable = 0
        for _, row in frame.iterrows():
            raw = row[top_score_candidates]
            if not isinstance(raw, (list, tuple)):
                continue
            usable += 1
            actual = f"{int(row[actual_home_score])}-{int(row[actual_away_score])}"
            candidates = {str(v[0]) if isinstance(v, (list, tuple)) and v else str(v) for v in raw}
            hits += actual in candidates
        if usable:
            out["top_n_score_hit_rate"] = float(hits / usable)
    return out


def binary_probability_metrics(
    frame: pd.DataFrame,
    *,
    probability: str,
    actual: str,
    positive_label: int = 1,
) -> Dict[str, float]:
    """Evaluate a binary probability such as High probability without refitting."""
    _require(frame, [probability, actual])
    x = frame[[probability, actual]].copy()
    x[probability] = pd.to_numeric(x[probability], errors="coerce")
    x[actual] = pd.to_numeric(x[actual], errors="coerce")
    x = x.dropna()
    if x.empty:
        raise ValueError("no valid binary probability rows")
    p = np.clip(x[probability].to_numpy(float), 1e-9, 1 - 1e-9)
    y = (x[actual].to_numpy(float) == positive_label).astype(int)
    pred = (p >= 0.5).astype(int)
    return {
        "rows": float(len(x)),
        "accuracy": float((pred == y).mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "positive_rate": float(y.mean()),
    }


def low_high_from_scores(
    frame: pd.DataFrame,
    *,
    actual_home_score: str = "actual_home_score",
    actual_away_score: str = "actual_away_score",
    line: float = 7.5,
) -> pd.DataFrame:
    """Derive realized Low/High labels from realized total runs.

    This helper does not use predicted scores, preventing accidental target-derived
    probability construction. Games exactly on the line are treated as neither
    Low nor High and excluded by callers when appropriate.
    """
    _require(frame, [actual_home_score, actual_away_score])
    out = frame.copy()
    total = pd.to_numeric(out[actual_home_score], errors="coerce") + pd.to_numeric(out[actual_away_score], errors="coerce")
    out["realized_total_runs"] = total
    out["realized_low"] = (total < line).astype("Int64")
    out["realized_high"] = (total > line).astype("Int64")
    out.loc[total == line, ["realized_low", "realized_high"]] = pd.NA
    return out
