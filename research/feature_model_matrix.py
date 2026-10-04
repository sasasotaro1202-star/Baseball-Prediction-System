"""Large, deterministic baseball feature/model matrix experiment.

This runner is research-only. It intentionally separates Development OOS
selection from the newest locked holdout. A holdout score can never affect the
winner selection.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from baseball_backtest import BaseballBacktest
from evaluation.metrics import expected_calibration_error, multiclass_brier
from research.feature_set_variants import SCREENING_VARIANTS, select_feature_set


MODEL_POOLS: dict[str, str] = {
    "LINEAR_TREE": "Logistic,HistGB",
    "BROAD_TREE": "Logistic,HistGB,RandomForest,ExtraTrees",
    "DIVERSE": "Logistic,HistGB,RandomForest,ExtraTrees,KNNAnalog",
}


@dataclass(frozen=True)
class Config:
    variant: str
    half_life: int
    model_pool: str

    @property
    def id(self) -> str:
        return f"{self.variant}|hl={self.half_life}|pool={self.model_pool}"


def _metrics(y: np.ndarray, p: np.ndarray, league: str) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
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


def _load_games(bt: BaseballBacktest, league: str, start: int, end: int):
    if league == "NPB":
        games = bt.aggregate_npb_games(bt.load_npb_pbp())
    else:
        games = bt.load_mlb(start, end)
    if "game_class" in games:
        games = games[games["game_class"].astype(str).eq("official")]
    if "season_type" in games:
        games = games[games["season_type"].astype(str).eq("regular_season")]
    games = games.sort_values(["datetime", "game_id"]).reset_index(drop=True)
    if len(games) < 500:
        raise RuntimeError(f"{league}: insufficient chronological games for matrix experiment: {len(games)}")
    return games


def _development_folds(locked_start: int) -> list[tuple[int, int]]:
    # Three expanding chronological evaluation blocks. The last 20% of all
    # games remains untouched and is therefore a genuine locked holdout.
    ratios = ((0.55, 0.68), (0.68, 0.81), (0.81, 1.00))
    folds = []
    for start_ratio, end_ratio in ratios:
        cut = int(locked_start * start_ratio)
        stop = int(locked_start * end_ratio)
        if cut >= 240 and stop > cut + 40:
            folds.append((cut, min(stop, locked_start)))
    if len(folds) < 2:
        raise RuntimeError(f"development chronology too short for 2 folds: locked_start={locked_start}")
    return folds


def _run_config(
    *,
    bt: BaseballBacktest,
    X,
    y,
    meta,
    league: str,
    config: Config,
    folds: list[tuple[int, int]],
) -> dict[str, Any]:
    os.environ["BASEBALL_RECENCY_HALF_LIFE_GAMES"] = str(config.half_life)
    os.environ["BASEBALL_FAST_MODEL_POOL"] = MODEL_POOLS[config.model_pool]
    os.environ["BASEBALL_FAST_OOS"] = "1"

    Xv, feature_meta = select_feature_set(X, league, variant=config.variant)
    fold_metrics = []
    all_y: list[np.ndarray] = []
    all_p: list[np.ndarray] = []

    for cut, stop in folds:
        fitted, validation_scores, selected = bt.fit_ensemble(
            Xv.iloc[:cut],
            y[:cut],
            league,
            fast_oos=True,
            context_keys=meta.iloc[:cut]["competition_key"]
            if "competition_key" in meta.columns else None,
        )
        if not fitted:
            raise RuntimeError(f"{config.id}: no ensemble at fold cut={cut}")
        p = bt.ensemble_proba(
            fitted,
            Xv.iloc[cut:stop],
            league,
            context_keys=meta.iloc[cut:stop]["competition_key"]
            if "competition_key" in meta.columns else None,
        )
        yy = y[cut:stop]
        fm = _metrics(yy, p, league)
        fm.update({
            "cut": int(cut),
            "stop": int(stop),
            "period_start": str(meta.iloc[cut]["datetime"]),
            "period_end": str(meta.iloc[stop - 1]["datetime"]),
        })
        fold_metrics.append(fm)
        all_y.append(yy)
        all_p.append(p)

    y_dev = np.concatenate(all_y)
    p_dev = np.vstack(all_p)
    aggregate = _metrics(y_dev, p_dev, league)
    fold_logloss = np.asarray([m["LogLoss"] for m in fold_metrics], dtype=float)

    return {
        "config_id": config.id,
        "variant": config.variant,
        "half_life": int(config.half_life),
        "model_pool": config.model_pool,
        "feature_meta": feature_meta,
        "development_metrics": aggregate,
        "fold_metrics": fold_metrics,
        "mean_fold_LogLoss": float(np.mean(fold_logloss)),
        "max_fold_LogLoss": float(np.max(fold_logloss)),
        "min_fold_LogLoss": float(np.min(fold_logloss)),
        "selected_models_last_fold": selected,
        "status": "DEVELOPMENT_OOS_EXECUTED",
    }


def _select(results: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        r for r in results
        if r.get("status") == "DEVELOPMENT_OOS_EXECUTED"
        and np.isfinite(float(r["mean_fold_LogLoss"]))
        and np.isfinite(float(r["development_metrics"]["ECE"]))
    ]
    if not usable:
        raise RuntimeError("matrix experiment produced no usable Development OOS candidates")
    usable.sort(
        key=lambda r: (
            float(r["mean_fold_LogLoss"]),
            float(r["development_metrics"]["Brier"]),
            float(r["development_metrics"]["ECE"]),
            -float(r["development_metrics"]["Accuracy"]),
            str(r["config_id"]),
        )
    )
    return usable[0]


def run(
    *,
    league: str,
    data_dir: str = "data",
    holdout_fraction: float = 0.20,
    variants: list[str] | None = None,
    half_lives: list[int] | None = None,
    model_pools: list[str] | None = None,
    max_configs: int = 160,
    mlb_start: int = 2020,
    mlb_end: int = 2026,
) -> dict[str, Any]:
    if not 0.15 <= holdout_fraction <= 0.30:
        raise ValueError("holdout_fraction must be between 0.15 and 0.30")
    if max_configs < 1:
        raise ValueError("max_configs must be >= 1")

    chosen_variants = variants or list(SCREENING_VARIANTS)
    unknown = sorted(set(chosen_variants) - set(SCREENING_VARIANTS))
    if unknown:
        raise ValueError("unknown feature variants: " + ", ".join(unknown))

    chosen_hl = sorted(set(half_lives or [600, 900, 1800, 3600, 7200]))
    if any(x < 100 for x in chosen_hl):
        raise ValueError("half-lives must be >= 100")
    chosen_pools = model_pools or list(MODEL_POOLS)
    unknown_pools = sorted(set(chosen_pools) - set(MODEL_POOLS))
    if unknown_pools:
        raise ValueError("unknown model pools: " + ", ".join(unknown_pools))

    # One full PIT-safe chronological feature construction. Downstream variants
    # only remove columns, so every candidate sees identical row identity.
    os.environ["BASEBALL_FEATURE_SET_VARIANT"] = "FULL_VALIDATED_ENSEMBLE"
    os.environ["BASEBALL_FAST_OOS"] = "1"
    bt = BaseballBacktest(Path(data_dir))
    games = _load_games(bt, league, mlb_start, mlb_end)
    X, y, meta = bt.build_features(games)

    locked_start = int(len(X) * (1.0 - holdout_fraction))
    folds = _development_folds(locked_start)

    configs = [
        Config(v, h, p)
        for v in chosen_variants
        for h in chosen_hl
        for p in chosen_pools
    ]
    if len(configs) > max_configs:
        # Deterministic coverage-preserving truncation: keep every variant,
        # spreading the budget across half-life/model-pool combinations.
        configs = configs[:max_configs]

    baseline_ids = {f"BASELINE_TEAM_STATE|hl={h}|pool={p}" for h in chosen_hl for p in chosen_pools}
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for idx, config in enumerate(configs, start=1):
        bt._check_time_budget(f"feature_matrix:{idx}/{len(configs)}")
        try:
            results.append(_run_config(
                bt=bt, X=X, y=y, meta=meta, league=league, config=config, folds=folds
            ))
        except Exception as exc:
            failures.append({
                "config_id": config.id,
                "error": f"{type(exc).__name__}: {exc}",
            })

    winner = _select(results)
    winner_baselines = [
        r for r in results if r.get("config_id") in baseline_ids
    ]
    matched_baseline = min(
        winner_baselines,
        key=lambda r: (
            abs(int(r["half_life"]) - int(winner["half_life"])),
            0 if r["model_pool"] == winner["model_pool"] else 1,
            float(r["mean_fold_LogLoss"]),
        ),
        default=None,
    )

    # Locked holdout is scored exactly once for the Development-selected winner.
    wcfg = Config(
        winner["variant"],
        int(winner["half_life"]),
        str(winner["model_pool"]),
    )
    Xw, winner_feature_meta = select_feature_set(X, league, variant=wcfg.variant)
    y_train, y_holdout = y[:locked_start], y[locked_start:]
    fitted, validation_scores, selected = bt.fit_ensemble(
        Xw.iloc[:locked_start],
        y_train,
        league,
        fast_oos=False,
        context_keys=meta.iloc[:locked_start]["competition_key"]
        if "competition_key" in meta.columns else None,
    )
    if not fitted:
        raise RuntimeError("winner holdout fit produced no ensemble")
    p_holdout = bt.ensemble_proba(
        fitted,
        Xw.iloc[locked_start:],
        league,
        context_keys=meta.iloc[locked_start:]["competition_key"]
        if "competition_key" in meta.columns else None,
    )
    holdout_metrics = _metrics(y_holdout, p_holdout, league)

    return {
        "schema_version": "baseball-feature-model-matrix-v1",
        "status": "RESEARCH_ONLY",
        "selection_basis": "Development OOS only",
        "holdout_locked_before_selection": True,
        "league": league,
        "total_games": int(len(X)),
        "development_rows": int(locked_start),
        "locked_holdout_rows": int(len(y_holdout)),
        "development_folds": [
            {"start_row": int(a), "end_row": int(b), "start": str(meta.iloc[a]["datetime"]), "end": str(meta.iloc[b - 1]["datetime"])}
            for a, b in folds
        ],
        "matrix_size_requested": int(len(chosen_variants) * len(chosen_hl) * len(chosen_pools)),
        "matrix_size_executed": int(len(results) + len(failures)),
        "successful_configs": int(len(results)),
        "failed_configs": int(len(failures)),
        "candidate_results": sorted(
            results,
            key=lambda r: (
                float(r["mean_fold_LogLoss"]),
                float(r["development_metrics"]["Brier"]),
                str(r["config_id"]),
            ),
        ),
        "failures": failures,
        "winner": {
            "config_id": winner["config_id"],
            "feature_meta": winner_feature_meta,
            "development_metrics": winner["development_metrics"],
            "mean_fold_LogLoss": winner["mean_fold_LogLoss"],
            "max_fold_LogLoss": winner["max_fold_LogLoss"],
        },
        "matched_baseline": None if matched_baseline is None else {
            "config_id": matched_baseline["config_id"],
            "development_metrics": matched_baseline["development_metrics"],
            "mean_fold_LogLoss": matched_baseline["mean_fold_LogLoss"],
            "relative_LogLoss_improvement": (
                1.0 - float(winner["mean_fold_LogLoss"]) / float(matched_baseline["mean_fold_LogLoss"])
            ),
        },
        "locked_holdout": {
            "winner_only": True,
            "metrics": holdout_metrics,
            "selected_model": selected,
            "validation_scores": {k: float(v) for k, v in validation_scores.items()},
            "period": {
                "start": str(meta.iloc[locked_start]["datetime"]),
                "end": str(meta.iloc[-1]["datetime"]),
            },
        },
        "decision": "NO_AUTO_ADOPTION",
        "next_gate": "repeat winner on full chronological WFO -> calibration -> ablation -> robustness -> frozen holdout -> explicit adoption",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", choices=["NPB", "MLB"], required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--half-life", action="append", dest="half_lives", type=int)
    parser.add_argument("--model-pool", action="append", dest="model_pools")
    parser.add_argument("--max-configs", type=int, default=160)
    parser.add_argument("--mlb-start", type=int, default=2020)
    parser.add_argument("--mlb-end", type=int, default=2026)
    args = parser.parse_args()
    payload = run(
        league=args.league,
        data_dir=args.data_dir,
        holdout_fraction=args.holdout_fraction,
        variants=args.variants,
        half_lives=args.half_lives,
        model_pools=args.model_pools,
        max_configs=args.max_configs,
        mlb_start=args.mlb_start,
        mlb_end=args.mlb_end,
    )
    out = Path(args.data_dir).parent / "results" / f"feature_model_matrix_{args.league.lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
