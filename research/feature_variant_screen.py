"""Fast chronological screening for explicit baseball feature-set variants.

This is a research screen, not a promotion path. It evaluates each variant on
the newest chronological holdout after model-selection validation on the older
prefix. Holdout metrics are reported only after the variant is fixed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from baseball_backtest import BaseballBacktest
from evaluation.metrics import expected_calibration_error, multiclass_brier
from research.feature_set_variants import SCREENING_VARIANTS


def _metrics(y: np.ndarray, p: np.ndarray, league: str) -> dict[str, float]:
    if league == "NPB":
        return {
            "Accuracy": float(accuracy_score(y, np.argmax(p, axis=1))),
            "LogLoss": float(log_loss(y, p, labels=[0, 1, 2])),
            "Brier": float(multiclass_brier(y, p, classes=[0, 1, 2])),
            "ECE": float(expected_calibration_error(y, p, classes=[0, 1, 2])),
            "rows": int(len(y)),
        }
    positive = p[:, 0]
    return {
        "Accuracy": float(accuracy_score(y, (positive >= 0.5).astype(int))),
        "LogLoss": float(log_loss(y, p, labels=[0, 1])),
        "Brier": float(brier_score_loss((y == 0).astype(int), positive)),
        "ECE": float(expected_calibration_error(y, p, classes=[0, 1])),
        "rows": int(len(y)),
    }


def _load_games(bt: BaseballBacktest, league: str, mlb_start: int, mlb_end: int):
    if league == "NPB":
        raw = bt.load_npb_pbp()
        games = bt.aggregate_npb_games(raw)
        if "game_class" in games:
            games = games[games["game_class"].astype(str).eq("official")]
        if "season_type" in games:
            games = games[games["season_type"].astype(str).eq("regular_season")]
    else:
        games = bt.load_mlb(mlb_start, mlb_end)
        if "game_class" in games:
            games = games[games["game_class"].astype(str).eq("official")]
        if "season_type" in games:
            games = games[games["season_type"].astype(str).eq("regular_season")]
    games = games.sort_values(["datetime", "game_id"]).reset_index(drop=True)
    if len(games) < 260:
        raise RuntimeError(f"{league} feature screening requires >=260 chronological games; got {len(games)}")
    return games


def run(*, league: str, data_dir: str = "data", holdout_fraction: float = 0.20,
        variants: list[str] | None = None, mlb_start: int = 2020, mlb_end: int = 2026) -> dict:
    if not 0.10 <= holdout_fraction <= 0.40:
        raise ValueError("holdout_fraction must be between 0.10 and 0.40")
    chosen = variants or list(SCREENING_VARIANTS)
    unknown = sorted(set(chosen) - set(SCREENING_VARIANTS))
    if unknown:
        raise ValueError("unknown feature variants: " + ", ".join(unknown))

    results = []
    for variant in chosen:
        os.environ["BASEBALL_FEATURE_SET_VARIANT"] = variant
        os.environ.setdefault("BASEBALL_FAST_OOS", "1")
        os.environ.setdefault("BASEBALL_FAST_MODEL_POOL", "Logistic,HistGB,RandomForest,ExtraTrees")
        bt = BaseballBacktest(Path(data_dir))
        games = _load_games(bt, league, mlb_start, mlb_end)

        # Feature construction remains fully chronological. For the final holdout
        # rows, prior holdout games may legitimately update state because their
        # outcomes would be known before later holdout games.
        X, y, meta = bt.build_features(games)
        cut = int(len(X) * (1.0 - holdout_fraction))
        if cut < 180 or len(X) - cut < 60:
            raise RuntimeError(f"{variant}: invalid train/holdout split sizes")
        X_train, X_holdout = X.iloc[:cut], X.iloc[cut:]
        y_train, y_holdout = y[:cut], y[cut:]

        fitted, validation_scores, selected = bt.fit_ensemble(
            X_train, y_train, league, fast_oos=True,
            context_keys=meta.iloc[:cut]["competition_key"]
            if "competition_key" in meta.columns else None,
        )
        if not fitted:
            raise RuntimeError(f"{variant}: ensemble fitting returned no models")
        p = bt.ensemble_proba(
            fitted, X_holdout, league,
            context_keys=meta.iloc[cut:]["competition_key"]
            if "competition_key" in meta.columns else None,
        )
        metrics = _metrics(y_holdout, p, league)
        feature_meta = dict(getattr(bt, "_feature_set_metadata", {}))
        results.append({
            "variant": variant,
            "status": "SCREENED",
            "feature_meta": feature_meta,
            "train_rows": int(cut),
            "holdout_rows": int(len(y_holdout)),
            "validation_scores": {k: float(v) for k, v in validation_scores.items()},
            "selected_model_by_validation": selected,
            "holdout_metrics": metrics,
            "holdout_period": {
                "start": str(meta.iloc[cut]["datetime"]),
                "end": str(meta.iloc[-1]["datetime"]),
            },
        })

    results.sort(key=lambda x: (
        float(x["holdout_metrics"]["LogLoss"]),
        float(x["holdout_metrics"]["Brier"]),
        -float(x["holdout_metrics"]["Accuracy"]),
        str(x["variant"]),
    ))
    return {
        "schema_version": "baseball-feature-variant-screen-v1",
        "status": "RESEARCH_SCREEN_ONLY",
        "league": league,
        "holdout_fraction": holdout_fraction,
        "variants": results,
        "decision": "NO_AUTO_ADOPTION",
        "adoption_rule": "screen winner -> repeat on full chronological WFO -> calibration -> ablation -> robustness -> locked holdout -> explicit adoption",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", choices=["NPB", "MLB"], required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--mlb-start", type=int, default=2020)
    parser.add_argument("--mlb-end", type=int, default=2026)
    args = parser.parse_args()
    payload = run(
        league=args.league,
        data_dir=args.data_dir,
        holdout_fraction=args.holdout_fraction,
        variants=args.variants,
        mlb_start=args.mlb_start,
        mlb_end=args.mlb_end,
    )
    out = Path(args.data_dir).parent / "results" / f"feature_variant_screen_{args.league.lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
