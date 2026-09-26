"""Run the Future Generalization v2 controller on real NPB Development-OOS data.

Research-only. The frozen holdout is never used to fit or tune the controller.
Development-forward ablation is used for A-J comparison; the full J controller
is then frozen and applied once to the independent holdout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from baseball_backtest import BaseballBacktest
from research.future_generalization_v2 import (
    FutureGeneralizationController,
    metrics,
    paired_block_bootstrap,
    selective_metrics,
)
from research.npb_candidate_replay import ReplayConfig, _candidate_probability, _development_compare, _fit_candidate


RESULT = Path("research_outputs/future_generalization_v2_npb.json")


def _temperature_blend(p):
    q = np.asarray(p, dtype=float)
    q = q / q.sum(axis=1, keepdims=True)
    return q


def main() -> None:
    data_dir = Path("data")
    bt = BaseballBacktest(data_dir)
    raw = bt.load_npb_pbp()
    games = bt.aggregate_npb_games(raw).copy()

    pit_safe = (
        __import__("os").getenv("PIT_SAFE_STARTER_DATA", "0") == "1"
        and "starter_evidence_status" in games.columns
        and bool((games["starter_evidence_status"] == "pit_safe").all())
        and "confirmed_starters" in games.columns
        and bool(games["confirmed_starters"].all())
    )
    if not pit_safe:
        games["home_starter"] = ""
        games["away_starter"] = ""
        games["confirmed_starters"] = False
        games["starter_evidence_status"] = "not_pit_safe"

    X, y, _ = bt.build_features(games)
    cfg = ReplayConfig()
    n = len(X)
    holdout_start = int(n * (1.0 - cfg.holdout_fraction))
    dev_start = max(cfg.min_train_rows, int(n * 0.10))
    if holdout_start <= dev_start or n - holdout_start < cfg.min_holdout_rows:
        raise RuntimeError("insufficient NPB rows for independent holdout")

    candidate_names = list(bt.models("NPB").keys())
    development, windows, dev_predictions = _development_compare(
        bt, X, y, dev_start, holdout_start, candidate_names, cfg.block_size, cfg.retrain_every
    )
    names = [x for x in candidate_names if x in dev_predictions]
    if len(names) < 2:
        raise RuntimeError("not enough base models with Development-OOS predictions")

    y_dev = y[dev_start:holdout_start]
    prod_dev = dev_predictions.get("ProductionEnsemble")
    if prod_dev is None:
        raise RuntimeError("ProductionEnsemble Development-OOS predictions missing")

    # Development-forward ablation: train on first 65% of Development-OOS,
    # evaluate only on the later suffix. No frozen holdout is touched.
    cut = int(len(y_dev) * 0.65)
    fit_y, eval_y = y_dev[:cut], y_dev[cut:]
    fit_p = {k: dev_predictions[k][:cut] for k in names}
    eval_p = {k: dev_predictions[k][cut:] for k in names}
    ctl = FutureGeneralizationController(block_size=max(30, cfg.block_size // 2))
    ctl.fit(fit_y, fit_p)

    modes = ["B", "C", "D", "E", "F", "G", "H", "I", "J"]
    ablation = {
        "A_CurrentProduction": metrics(eval_y, prod_dev[cut:]),
    }
    for mode in modes:
        routed = ctl.route(eval_p, mode=mode)
        ablation[mode] = metrics(eval_y, routed["probabilities"])
    full_dev = ctl.route(eval_p, mode="J")
    ablation_selective = selective_metrics(eval_y, full_dev["probabilities"], full_dev["confidence"])

    # Freeze the controller using the complete Development-OOS stream.
    frozen = FutureGeneralizationController(block_size=max(30, cfg.block_size // 2))
    frozen.fit(y_dev, {k: dev_predictions[k] for k in names})

    X_train = X.iloc[:holdout_start]
    y_train = y[:holdout_start]
    X_holdout = X.iloc[holdout_start:].reset_index(drop=True)
    y_holdout = y[holdout_start:]
    holdout_probs = {}
    for name in names:
        model = _fit_candidate(bt, name, X_train, y_train)
        holdout_probs[name] = _candidate_probability(bt, model, X_holdout)

    base_fit, _, _ = bt.fit_ensemble(X_train, y_train, "NPB")
    if not base_fit:
        raise RuntimeError("production ensemble could not be fitted for holdout")
    production_holdout = bt.ensemble_proba(base_fit, X_holdout, "NPB")
    full = frozen.route(holdout_probs, mode="J")
    routed_holdout = full["probabilities"]

    result = {
        "schema_version": "future-generalization-v2-npb-v1",
        "git_commit": __import__("os").getenv("GITHUB_SHA", "unknown"),
        "dataset_hash": hashlib.sha256(
            __import__("pandas").util.hash_pandas_object(games, index=True).values.tobytes()
        ).hexdigest(),
        "pit": {
            "status": "PASS",
            "starter_mode": "pit_safe" if pit_safe else "not_pit_safe_and_blank",
            "future_holdout_excluded_from_fit": True,
        },
        "development_oos": {
            "rows": int(len(y_dev)),
            "validation_windows": int(windows),
            "model_names": names,
            "ablation": ablation,
            "full_selective": ablation_selective,
        },
        "holdout": {
            "rows": int(len(y_holdout)),
            "baseline": metrics(y_holdout, production_holdout),
            "full_architecture": metrics(y_holdout, routed_holdout),
            "selective": selective_metrics(y_holdout, routed_holdout, full["confidence"]),
            "statistical_validation": paired_block_bootstrap(
                y_holdout,
                production_holdout,
                routed_holdout,
                block_size=max(10, min(30, len(y_holdout) // 10)),
                replications=400,
                seed=42,
            ),
            "drift_levels": {str(k): int(v) for k, v in __import__("collections").Counter(full["drift_level"]).items()},
            "mean_predictability": float(np.mean(full["predictability"])),
            "mean_confidence": float(np.mean(full["confidence"])),
            "abstain_rate": float(np.mean(full["abstain"])),
        },
        "controller": {
            "config": frozen.audit,
            "safety": full["safety"],
            "production_changed": False,
            "promotion": "HOLD",
        },
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({
        "development_ablation": ablation,
        "holdout_baseline": result["holdout"]["baseline"],
        "holdout_full": result["holdout"]["full_architecture"],
        "statistical_validation": result["holdout"]["statistical_validation"],
        "pit": result["pit"],
        "promotion": "HOLD",
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
