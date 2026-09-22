"""Normalize walk-forward outputs to the canonical prediction contract.

The historical backtest engine predates the shared production score contract.
This adapter is deliberately downstream of model inference: it does not alter
win/draw/away probabilities, but regenerates exact-score and Low/High outputs
from the same game-specific run means and adds the separate highest-draw
selection required for draw-capable competitions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from prediction.game_ranking import rank_games
from prediction.score_distribution import build_score_outputs


def _repair_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = {"lambda_home", "lambda_away"}
    if not required.issubset(out.columns):
        raise ValueError(f"missing run-distribution columns: {sorted(required - set(out.columns))}")

    score_rows: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        generated = build_score_outputs(
            float(row["lambda_home"]),
            float(row["lambda_away"]),
            shared_lambda=float(row.get("shared_lambda", 0.0)),
        )
        score_rows.append(generated)
    out["score1"] = [x["score_candidates"][0]["score"] for x in score_rows]
    out["score1_prob"] = [x["score_candidates"][0]["probability"] for x in score_rows]
    out["score2"] = [x["score_candidates"][1]["score"] for x in score_rows]
    out["score2_prob"] = [x["score_candidates"][1]["probability"] for x in score_rows]
    out["score3"] = [x["score_candidates"][2]["score"] for x in score_rows]
    out["score3_prob"] = [x["score_candidates"][2]["probability"] for x in score_rows]
    out["score4"] = [x["score_candidates"][3]["score"] for x in score_rows]
    out["score4_prob"] = [x["score_candidates"][3]["probability"] for x in score_rows]
    out["low"] = [x["low_probability"] for x in score_rows]
    out["high"] = [x["high_probability"] for x in score_rows]
    return out


def add_top_draw_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Add one highest-draw selection per Japan-local calendar date.

    Only NPB rows are draw-capable here.  The selection is made among all rows
    for the date using the model's already-produced ``pred_draw`` probability;
    it never forces the selected game to be the normal 3-way argmax prediction.
    """
    if df.empty or "pred_draw" not in df.columns:
        raise ValueError("NPB output requires pred_draw for Top-Draw selection")
    required = {"game_id", "datetime", "pred_home", "pred_draw", "pred_away"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing draw-ranking columns: {sorted(missing)}")

    work = df.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce", utc=True)
    if work["datetime"].isna().any():
        raise ValueError("invalid datetime in walk-forward output")
    rows = []
    for _, r in work.iterrows():
        rows.append({
            "event_id": str(r["game_id"]),
            "home_probability": float(r["pred_home"]),
            "draw_probability": float(r["pred_draw"]),
            "away_probability": float(r["pred_away"]),
        })
    # Validate every game through the shared ranking contract first.
    rank_games(rows, draw_capable=True)
    work["top_draw_selection"] = False
    work["top_draw_probability"] = pd.NA
    work["top_draw_rank"] = pd.NA
    work["top_draw_status"] = "NOT_SELECTED"
    work["date_jst"] = work["datetime"].dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m-%d")

    for date, idx in work.groupby("date_jst", sort=True).groups.items():
        subset = work.loc[idx]
        ranked = rank_games(
            [
                {
                    "event_id": str(r["game_id"]),
                    "home_probability": float(r["pred_home"]),
                    "draw_probability": float(r["pred_draw"]),
                    "away_probability": float(r["pred_away"]),
                }
                for _, r in subset.iterrows()
            ],
            draw_capable=True,
        )
        candidate = ranked["draw_candidate"]
        candidate_id = str(candidate["event_id"])
        mask = work.loc[idx, "game_id"].astype(str).eq(candidate_id)
        selected_idx = work.loc[idx].index[mask][0]
        work.at[selected_idx, "top_draw_selection"] = True
        work.at[selected_idx, "top_draw_probability"] = float(candidate["draw_probability"])
        work.at[selected_idx, "top_draw_rank"] = 1
        work.at[selected_idx, "top_draw_status"] = "SELECTED"

    return work


def normalize_walkforward(path: Path, *, npb: bool) -> Path:
    df = pd.read_csv(path)
    df = _repair_scores(df)
    if npb:
        df = add_top_draw_selection(df)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/checkpoints")
    args = parser.parse_args()
    root = Path(args.results_dir)
    targets = [(root / "npb_walkforward.csv", True), (root / "mlb_walkforward.csv", False)]
    for path, is_npb in targets:
        if not path.exists():
            continue
        normalize_walkforward(path, npb=is_npb)
        print(f"canonical backtest output repaired: {path}")


if __name__ == "__main__":
    main()
