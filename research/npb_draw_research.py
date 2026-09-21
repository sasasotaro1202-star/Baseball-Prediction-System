"""Research metrics for the separate NPB top-draw prediction.

For each Japan-local calendar date, the system selects exactly one eligible NPB
game: the one with the highest predicted Draw probability. This is evaluated
separately from the normal Home/Draw/Away winner metric.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _binary_brier(y: pd.Series, p: pd.Series) -> float:
    return float(((p - y.astype(float)) ** 2).mean())


def _daily_top_draw_threshold_scan(df: pd.DataFrame) -> dict[str, Any]:
    """Tune a draw-probability threshold on the supplied development slice only.

    This is research evidence, not a production override. Threshold selection is
    intentionally isolated from the later chronological evaluation slice.
    """
    rows = []
    for threshold in [i / 100 for i in range(1, 31)]:
        eligible = df[df["_draw_p"] >= threshold]
        if eligible.empty:
            continue
        selected = eligible.loc[eligible.groupby("_game_date")["_draw_p"].idxmax()].copy()
        hit = selected["_actual_draw"].mean()
        brier = _binary_brier(selected["_actual_draw"], selected["_draw_p"])
        rows.append({
            "threshold": threshold,
            "slates": int(len(selected)),
            "hit_rate": float(hit),
            "brier": float(brier),
            "mean_probability": float(selected["_draw_p"].mean()),
        })
    if not rows:
        return {"status": "UNAVAILABLE", "reason": "no threshold candidates"}
    best = min(rows, key=lambda x: (x["brier"], -x["hit_rate"], x["threshold"]))
    return {"status": "PASS", "best": best, "scan": rows}


def _daily_top_draw_with_threshold(df: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Evaluate one fixed threshold without re-selecting it on this slice."""
    eligible = df[df["_draw_p"] >= float(threshold)]
    if eligible.empty:
        return {"status": "UNAVAILABLE", "reason": "no games above selected threshold"}
    selected = eligible.loc[eligible.groupby("_game_date")["_draw_p"].idxmax()].copy()
    selected["draw_probability_error"] = (
        selected["_draw_p"] - selected["_actual_draw"].astype(float)
    ).abs()
    return {
        "status": "PASS",
        "threshold": float(threshold),
        "slates": int(len(selected)),
        "hit_rate": float(selected["_actual_draw"].mean()),
        "brier": _binary_brier(selected["_actual_draw"], selected["_draw_p"]),
        "probability_mae": float(selected["draw_probability_error"].mean()),
        "mean_probability": float(selected["_draw_p"].mean()),
    }


def evaluate_top_draw_from_backtest(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path or RESULTS / "npb_backtest_results.csv")
    if not path.exists():
        return {"status": "UNAVAILABLE", "reason": "npb_backtest_results.csv missing"}
    df = pd.read_csv(path)
    required = {"pred_draw", "actual_home_score", "actual_away_score"}
    missing = required - set(df.columns)
    if missing:
        return {"status": "UNAVAILABLE", "reason": "missing columns: " + ", ".join(sorted(missing))}
    date_col = next((c for c in ("datetime", "date", "game_date") if c in df.columns), None)
    if date_col is None:
        return {"status": "UNAVAILABLE", "reason": "no game date column"}
    df = df.copy()
    df["_draw_p"] = pd.to_numeric(df["pred_draw"], errors="coerce")
    raw_dt = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df["_game_date"] = raw_dt.dt.tz_convert("Asia/Tokyo").dt.date
    df["_actual_draw"] = df["actual_home_score"] == df["actual_away_score"]
    df = df.dropna(subset=["_draw_p", "_game_date"])
    df = df[df["_draw_p"].between(0.0, 1.0)]
    if df.empty:
        return {"status": "UNAVAILABLE", "reason": "no valid NPB prediction rows"}

    dates = sorted(df["_game_date"].dropna().unique())
    if len(dates) < 2:
        return {"status": "UNAVAILABLE", "reason": "fewer than two Japan-local slates"}
    split_idx = max(1, min(len(dates) - 1, int(len(dates) * 0.70)))
    dev_dates = set(dates[:split_idx])
    holdout_dates = set(dates[split_idx:])
    dev = df[df["_game_date"].isin(dev_dates)].copy()
    holdout = df[df["_game_date"].isin(holdout_dates)].copy()

    threshold_research = _daily_top_draw_threshold_scan(dev)
    fixed_threshold_eval = (
        _daily_top_draw_with_threshold(holdout, threshold_research["best"]["threshold"])
        if threshold_research.get("status") == "PASS"
        else {"status": "UNAVAILABLE", "reason": "threshold selection unavailable"}
    )

    selected = df.loc[df.groupby("_game_date")["_draw_p"].idxmax()].copy()
    selected["draw_probability_error"] = (
        selected["_draw_p"] - selected["_actual_draw"].astype(float)
    ).abs()
    result = {
        "status": "PASS",
        "prediction_type": "NPB_TOP_DRAW_PROBABILITY_V1",
        "threshold_research": threshold_research,
        "threshold_selection_split": {
            "method": "chronological_jst_date_70_30",
            "development_slates": int(len(dev_dates)),
            "evaluation_slates": int(len(holdout_dates)),
            "holdout_untouched_during_threshold_selection": True,
        },
        "fixed_threshold_holdout_evaluation": fixed_threshold_eval,
        "slates": int(len(selected)),
        "top_draw_hit_rate": float(selected["_actual_draw"].mean()),
        "top_draw_probability_mae": float(selected["draw_probability_error"].mean()),
        "mean_selected_draw_probability": float(selected["_draw_p"].mean()),
        "actual_draw_rate_all_games": float(df["_actual_draw"].mean()),
        "lift_vs_all_game_draw_rate": float(
            selected["_actual_draw"].mean() - df["_actual_draw"].mean()
        ),
        "selection_rule": "one game per Japan-local calendar date, maximum pred_draw",
        "timezone": "Asia/Tokyo",
    }
    return result

def write_top_draw_research(path: str | Path | None = None) -> dict[str, Any]:
    result = evaluate_top_draw_from_backtest(path)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "npb_top_draw_research.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
