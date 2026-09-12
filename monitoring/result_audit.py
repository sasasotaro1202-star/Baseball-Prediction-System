"""Post-game prediction audit: result reconciliation, errors and weaknesses."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.metrics import classification_metrics, score_metrics


def _validate_probability_rows(merged: pd.DataFrame, columns: list[str]) -> np.ndarray:
    probs = merged[columns].to_numpy(float)
    if not np.isfinite(probs).all():
        raise ValueError("audit probabilities must be finite")
    if (probs < 0).any() or (probs > 1).any():
        raise ValueError("audit probabilities must be in [0,1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("audit probabilities must sum to 1 per event")
    return probs


def audit_predictions(predictions: pd.DataFrame, results: pd.DataFrame, *, league: str) -> dict[str, Any]:
    """Join predictions to final results by event_id and calculate post-game metrics.

    Duplicate event IDs are rejected to prevent a many-to-many merge from
    silently overweighting games. Unmatched predictions are reported, not scored
    as misses.
    """
    if league not in {"NPB", "MLB"}:
        raise ValueError("league must be NPB or MLB")
    if "event_id" not in predictions or "event_id" not in results:
        raise ValueError("both predictions and results require event_id")
    p = predictions.copy()
    r = results.copy()
    if p["event_id"].duplicated().any():
        raise ValueError("prediction results contain duplicate event_id values")
    if r["event_id"].duplicated().any():
        raise ValueError("final results contain duplicate event_id values")

    merged = p.merge(r, on="event_id", how="inner", suffixes=("_prediction", "_result"), validate="one_to_one")
    if merged.empty:
        return {
            "league": league, "matched_rows": 0,
            "unmatched_predictions": int(len(p)),
            "unmatched_results": int(len(r)),
            "status": "NO_MATCHED_RESULTS",
        }

    if league == "NPB":
        required = {"home_win_probability", "draw_probability", "away_win_probability", "actual_home_score", "actual_away_score"}
        if not required.issubset(merged.columns):
            raise ValueError("NPB audit requires explicit three-way probabilities and final scores")
        probs = _validate_probability_rows(merged, ["home_win_probability", "draw_probability", "away_win_probability"])
        home = pd.to_numeric(merged["actual_home_score"], errors="coerce").to_numpy(float)
        away = pd.to_numeric(merged["actual_away_score"], errors="coerce").to_numpy(float)
        if not np.isfinite(home).all() or not np.isfinite(away).all():
            raise ValueError("final scores must be finite")
        y = np.where(home == away, 1, np.where(home > away, 0, 2)).astype(int)
        cls = classification_metrics(y, probs, classes=[0, 1, 2])
        cls["DrawRecall"] = float(((np.argmax(probs, axis=1) == 1) & (y == 1)).sum() / max(1, (y == 1).sum()))
    else:
        required = {"home_win_probability", "away_win_probability", "actual_home_score", "actual_away_score"}
        if not required.issubset(merged.columns):
            raise ValueError("MLB audit requires binary probabilities and final scores")
        probs = _validate_probability_rows(merged, ["home_win_probability", "away_win_probability"])
        home = pd.to_numeric(merged["actual_home_score"], errors="coerce").to_numpy(float)
        away = pd.to_numeric(merged["actual_away_score"], errors="coerce").to_numpy(float)
        if not np.isfinite(home).all() or not np.isfinite(away).all():
            raise ValueError("final scores must be finite")
        y = (home > away).astype(int)
        cls = classification_metrics(y, probs, classes=[0, 1])

    out: dict[str, Any] = {
        "league": league,
        "matched_rows": int(len(merged)),
        "unmatched_predictions": int(len(p) - len(merged)),
        "unmatched_results": int(len(r) - len(merged)),
        "classification": cls,
    }
    if {"pred_home_score", "pred_away_score"}.issubset(merged.columns):
        out["score"] = score_metrics(home, away, merged.pred_home_score, merged.pred_away_score)
    out["prediction_error_summary"] = {
        "home_probability_mae": float(np.mean(np.abs(probs[:, 0] - (y == 0).astype(float)))),
        "draw_probability_mae": float(np.mean(np.abs(probs[:, 1] - (y == 1).astype(float)))) if league == "NPB" else None,
    }
    return out


def audit_csv(prediction_csv: str | Path, result_csv: str | Path, *, league: str,
              output: str | Path | None = None) -> dict[str, Any]:
    report = audit_predictions(pd.read_csv(prediction_csv), pd.read_csv(result_csv), league=league)
    if output:
        p = Path(output); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
