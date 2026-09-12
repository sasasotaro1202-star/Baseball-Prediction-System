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
    # The backtest's datetime is normalized to UTC. Selection must use the
    # Japan-local calendar date, otherwise games around 00:00 JST can be put
    # into the wrong daily slate.
    df["_game_date"] = raw_dt.dt.tz_convert("Asia/Tokyo").dt.date
    df["_actual_draw"] = df["actual_home_score"] == df["actual_away_score"]
    df = df.dropna(subset=["_draw_p", "_game_date"])
    df = df[df["_draw_p"].between(0.0, 1.0)]
    if df.empty:
        return {"status": "UNAVAILABLE", "reason": "no valid NPB prediction rows"}

    selected = df.loc[df.groupby("_game_date")["_draw_p"].idxmax()].copy()
    selected["draw_probability_error"] = (selected["_draw_p"] - selected["_actual_draw"].astype(float)).abs()
    result = {
        "status": "PASS",
        "prediction_type": "NPB_TOP_DRAW_PROBABILITY_V1",
        "slates": int(len(selected)),
        "top_draw_hit_rate": float(selected["_actual_draw"].mean()),
        "top_draw_probability_mae": float(selected["draw_probability_error"].mean()),
        "mean_selected_draw_probability": float(selected["_draw_p"].mean()),
        "actual_draw_rate_all_games": float(df["_actual_draw"].mean()),
        "lift_vs_all_game_draw_rate": float(selected["_actual_draw"].mean() - df["_actual_draw"].mean()),
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
