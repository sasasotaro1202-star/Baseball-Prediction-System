"""Fast, fair feature-set screening on identical chronological folds.

This is a development/OOS screening tool, not a promotion mechanism. It builds
the PIT-safe feature matrix once, applies one feature-set variant at a time,
then fits the same fixed model panel on identical chronological folds.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from baseball_backtest import BaseballBacktest
from evaluation.metrics import expected_calibration_error
from research.feature_set_variants import SCREENING_VARIANTS, select_feature_set

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.asarray(p, dtype=float)
    p = np.maximum(p, 1e-12)
    p /= p.sum(axis=1, keepdims=True)
    y = np.asarray(y, dtype=int)
    ll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())
    acc = float(np.mean(np.argmax(p, axis=1) == y))
    onehot = np.eye(p.shape[1])[y]
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    ece = float(expected_calibration_error(y, p, classes=list(range(p.shape[1]))))
    return {"LogLoss": ll, "Brier": brier, "Accuracy": acc, "ECE": ece}


def _folds(n: int) -> list[tuple[int, int]]:
    if n < 600:
        raise ValueError(f"insufficient games for screening: n={n}, need >=600")
    candidates = [(int(n * 0.55), int(n * 0.10)), (int(n * 0.65), int(n * 0.10)), (int(n * 0.75), int(n * 0.10))]
    folds = [(tr, va) for tr, va in candidates if tr >= 250 and tr + va <= n]
    if len(folds) != 3:
        raise RuntimeError(f"could not construct exactly 3 chronological screening folds for n={n}")
    return folds


def _make_models(seed: int = 42):
    return [
        LogisticRegression(
            max_iter=1200,
            solver="lbfgs",
            multi_class="auto",
            random_state=seed,
        ),
        HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=1.0,
            random_state=seed,
        ),
    ]


def run_screen(*, data_dir: str | Path = "data") -> dict[str, object]:
    started = time.time()
    bt = BaseballBacktest(Path(data_dir))
    pbp = bt.load_npb_pbp()
    games = bt.aggregate_npb_games(pbp)
    games = games.sort_values(["datetime", "game_id"]).reset_index(drop=True)
    X_full, y, meta = bt.build_features(games)
    if len(X_full) != len(games):
        raise RuntimeError("feature/game row alignment failed")
    folds = _folds(len(X_full))
    results: list[dict[str, object]] = []

    for variant in SCREENING_VARIANTS:
        X, feature_meta = select_feature_set(X_full, "NPB", variant=variant)
        fold_metrics: list[dict[str, float]] = []
        for train_end, val_len in folds:
            Xtr = X.iloc[:train_end]
            Xva = X.iloc[train_end:train_end + val_len]
            ytr = y[:train_end]
            yva = y[train_end:train_end + val_len]
            members = []
            for model in _make_models():
                if isinstance(model, LogisticRegression):
                    scaler = StandardScaler()
                    xtr = scaler.fit_transform(Xtr)
                    xva = scaler.transform(Xva)
                    model.fit(xtr, ytr)
                    members.append(model.predict_proba(xva))
                else:
                    model.fit(Xtr, ytr)
                    members.append(model.predict_proba(Xva))
            if not members:
                raise RuntimeError(f"no model predictions for {variant}")
            p = np.mean(np.stack(members, axis=0), axis=0)
            fold_metrics.append(_metrics(yva, p))

        mean_metrics = {
            key: float(np.mean([fold[key] for fold in fold_metrics]))
            for key in fold_metrics[0]
        }
        newest = fold_metrics[-1]
        results.append({
            "feature_set_variant": variant,
            "feature_set_id": feature_meta["feature_set_id"],
            "feature_count": int(feature_meta["feature_count"]),
            "feature_schema_hash": feature_meta["feature_schema_hash"],
            "feature_family_counts": feature_meta["feature_family_counts"],
            "context_mode": feature_meta["feature_context_mode"],
            "folds": fold_metrics,
            "mean_metrics": mean_metrics,
            "newest_fold_metrics": newest,
            "evaluation_games": int(sum(v for _, v in folds)),
        })

    results.sort(key=lambda x: (float(x["mean_metrics"]["LogLoss"]), -float(x["mean_metrics"]["Accuracy"])))
    baseline = next(x for x in results if x["feature_set_variant"] == "BASELINE_TEAM_STATE")
    baseline_ll = float(baseline["mean_metrics"]["LogLoss"])
    for row in results:
        ll = float(row["mean_metrics"]["LogLoss"])
        row["relative_logloss_improvement_vs_baseline"] = float((baseline_ll - ll) / max(abs(baseline_ll), 1e-12))
        row["screening_status"] = "SCREENING_ONLY"

    output = {
        "schema_version": "baseball-feature-set-screen-v1",
        "status": "SCREENING_COMPLETED",
        "evidence_scope": "development_oos_screening",
        "league": "NPB",
        "objective": "compare feature variants on identical chronological folds",
        "folds": [{"train_end": tr, "validation_len": va} for tr, va in folds],
        "variants_tested": len(results),
        "winner_by_mean_logloss": results[0]["feature_set_variant"],
        "baseline_variant": baseline["feature_set_variant"],
        "baseline_mean_logloss": baseline_ll,
        "results": results,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": float(time.time() - started),
        "production_promotion": "NOT_RUN",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "feature_set_screening.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([
        {
            "feature_set_variant": r["feature_set_variant"],
            "feature_count": r["feature_count"],
            "LogLoss": r["mean_metrics"]["LogLoss"],
            "Brier": r["mean_metrics"]["Brier"],
            "Accuracy": r["mean_metrics"]["Accuracy"],
            "ECE": r["mean_metrics"]["ECE"],
            "NewestLogLoss": r["newest_fold_metrics"]["LogLoss"],
            "relative_logloss_improvement_vs_baseline": r["relative_logloss_improvement_vs_baseline"],
        }
        for r in results
    ]).to_csv(RESULTS / "feature_set_screening.csv", index=False)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    output = run_screen(data_dir=args.data_dir)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
