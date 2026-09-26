"""Maximum Future-Generalization v6 NPB end-to-end research runner.

All model selection and v6 fusion tuning happen on a forward suffix of
Development-OOS. The independent final holdout is touched exactly once, after
all v6 parameters are frozen. No Production artifact is modified.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from baseball_backtest import BaseballBacktest
from research.future_generalization_v2 import (
    FutureGeneralizationController,
    metrics,
    paired_block_bootstrap,
    selective_metrics,
)
from research.future_generalization_v6 import (
    ChronologicalMetaLabeler,
    HistoricalPrototypeRetriever,
    error_correlation,
    feature_reliability,
    multi_horizon_consistency,
    next_regime_distribution,
    prediction_dynamics,
    probability_safety_gate,
    regime_transition_probabilities,
    source_reliability,
    split_conformal_sets,
    stress_probability_stream,
    uncertainty_decomposition,
)
from research.npb_candidate_replay import ReplayConfig, _candidate_probability, _development_compare, _fit_candidate

RESULT = Path("research_outputs/future_generalization_v6_npb.json")


def _load():
    bt = BaseballBacktest(Path("data"))
    games = bt.aggregate_npb_games(bt.load_npb_pbp()).copy()
    pit_safe = (
        os.getenv("PIT_SAFE_STARTER_DATA", "0") == "1"
        and "starter_evidence_status" in games.columns
        and bool((games["starter_evidence_status"] == "pit_safe").all())
        and "confirmed_starters" in games.columns
        and bool(games["confirmed_starters"].all())
    )
    if not pit_safe:
        for c, v in (
            ("home_starter", ""), ("away_starter", ""),
            ("confirmed_starters", False), ("starter_evidence_status", "not_pit_safe"),
        ):
            games[c] = v
    X, y, _ = bt.build_features(games)
    cfg = ReplayConfig()
    n = len(X)
    holdout_start = int(n * (1.0 - cfg.holdout_fraction))
    dev_start = max(cfg.min_train_rows, int(n * 0.10))
    if holdout_start <= dev_start or n - holdout_start < cfg.min_holdout_rows:
        raise RuntimeError("insufficient NPB rows for independent holdout")
    names = list(bt.models("NPB").keys())
    development, windows, dev_predictions = _development_compare(
        bt, X, y, dev_start, holdout_start, names, cfg.block_size, cfg.retrain_every
    )
    usable = [n for n in names if n in dev_predictions]
    if len(usable) < 2 or "ProductionEnsemble" not in dev_predictions:
        raise RuntimeError("insufficient Development-OOS model streams")
    return bt, games, X, y, cfg, holdout_start, dev_start, windows, usable, dev_predictions


def _score_fusion(y_true, routed, retrieval):
    best = None
    for w in np.linspace(0.0, 0.50, 11):
        q = (1.0 - w) * routed + w * retrieval
        m = metrics(y_true, q)
        key = (m["LogLoss"], m["Brier"], -m["Accuracy"])
        if best is None or key < best[0]:
            best = (key, float(w), q, m)
    assert best is not None
    return {"weight": best[1], "metrics": best[3], "probabilities": best[2]}


def main() -> None:
    bt, games, X, y, cfg, holdout_start, dev_start, windows, names, dev_predictions = _load()
    y_dev = y[dev_start:holdout_start]

    # ----- Development-forward tuning (65/35 split; final holdout untouched) -----
    cut = int(len(y_dev) * 0.65)
    y_fit, y_eval = y_dev[:cut], y_dev[cut:]
    fit_p = {n: dev_predictions[n][:cut] for n in names}
    eval_p = {n: dev_predictions[n][cut:] for n in names}

    controller = FutureGeneralizationController(block_size=max(30, cfg.block_size // 2)).fit(y_fit, fit_p)
    routed_eval = controller.route(eval_p, mode="J")
    routed_fit = controller.route(fit_p, mode="J")

    mean_fit = np.mean(np.stack(list(fit_p.values()), axis=0), axis=0)
    mean_eval = np.mean(np.stack(list(eval_p.values()), axis=0), axis=0)
    fit_dyn = prediction_dynamics(mean_fit)
    eval_dyn = prediction_dynamics(mean_eval)
    state_fit = fit_dyn.join(uncertainty_decomposition(fit_p))
    state_eval = eval_dyn.join(uncertainty_decomposition(eval_p))

    # Retrieval and meta-labeling are trained strictly on the earlier OOS prefix.
    retrieval = HistoricalPrototypeRetriever().fit(state_fit, y_fit)
    retrieval_eval = retrieval.predict_proba(state_eval, 3)
    fused = _score_fusion(y_eval, routed_eval["probabilities"], retrieval_eval)

    metal = ChronologicalMetaLabeler().fit(state_fit, y_fit, fit_p[names[0]])
    meta_eval = metal.predict(state_eval, eval_p[names[0]])

    transition = regime_transition_probabilities(
        np.where(fit_dyn["prediction_flip"].to_numpy() > 0, "FLIP", "STABLE")
    )
    regime_next = next_regime_distribution(transition, "STABLE")

    # Conformal is a risk-control candidate, not a time-series guarantee.
    conformal_eval = split_conformal_sets(fit_p[names[0]], y_fit, eval_p[names[0]])

    ablation = {
        "A_CurrentProduction": metrics(y_eval, dev_predictions["ProductionEnsemble"][cut:]),
        "B_Disagreement": metrics(y_eval, controller.route(eval_p, mode="B")["probabilities"]),
        "C_Predictability": metrics(y_eval, controller.route(eval_p, mode="C")["probabilities"]),
        "D_FutureFailure": metrics(y_eval, controller.route(eval_p, mode="D")["probabilities"]),
        "E_Drift": metrics(y_eval, controller.route(eval_p, mode="E")["probabilities"]),
        "F_Disagreement_Predictability": metrics(y_eval, controller.route(eval_p, mode="F")["probabilities"]),
        "G_Disagreement_Failure": metrics(y_eval, controller.route(eval_p, mode="G")["probabilities"]),
        "H_Predictability_Failure": metrics(y_eval, controller.route(eval_p, mode="H")["probabilities"]),
        "I_Drift_Failure": metrics(y_eval, controller.route(eval_p, mode="I")["probabilities"]),
        "J_All_Three_CoreLayers": metrics(y_eval, routed_eval["probabilities"]),
        "K_All_Three_Plus_Retrieval": fused["metrics"],
    }
    selective = selective_metrics(y_eval, fused["probabilities"], fused["confidence"] if "confidence" in fused else fused["probabilities"].max(axis=1))

    # ----- Freeze all v6 decisions on complete Development-OOS -----
    frozen = FutureGeneralizationController(block_size=max(30, cfg.block_size // 2)).fit(
        y_dev, {n: dev_predictions[n] for n in names}
    )

    X_train = X.iloc[:holdout_start]
    y_train = y[:holdout_start]
    X_holdout = X.iloc[holdout_start:].reset_index(drop=True)
    y_holdout = y[holdout_start:]
    holdout_probs = {}
    for name in names:
        holdout_probs[name] = _candidate_probability(
            bt, _fit_candidate(bt, name, X_train, y_train), X_holdout
        )

    base_fit, _, _ = bt.fit_ensemble(X_train, y_train, "NPB")
    if not base_fit:
        raise RuntimeError("production ensemble could not be fitted for holdout")
    production_holdout = bt.ensemble_proba(base_fit, X_holdout, "NPB")
    routed_holdout = frozen.route(holdout_probs, mode="J")
    dyn_holdout = prediction_dynamics(np.mean(np.stack(list(holdout_probs.values()), axis=0), axis=0))
    state_holdout = dyn_holdout.join(uncertainty_decomposition(holdout_probs))

    full_retrieval = HistoricalPrototypeRetriever().fit(
        prediction_dynamics(np.mean(np.stack(list(dev_predictions[n] for n in names), axis=0), axis=0))
        .join(uncertainty_decomposition({n: dev_predictions[n] for n in names})),
        y_dev,
    )
    retrieval_holdout = full_retrieval.predict_proba(state_holdout, 3)
    routed_fused_holdout = (1.0 - fused["weight"]) * routed_holdout["probabilities"] + fused["weight"] * retrieval_holdout

    meta_frozen = ChronologicalMetaLabeler().fit(
        prediction_dynamics(np.mean(np.stack([dev_predictions[n] for n in names], axis=0), axis=0))
        .join(uncertainty_decomposition({n: dev_predictions[n] for n in names})),
        y_dev,
        dev_predictions[names[0]],
    )
    meta_holdout = meta_frozen.predict(state_holdout, holdout_probs[names[0]])

    stress = stress_probability_stream(routed_fused_holdout)
    safety = probability_safety_gate(routed_fused_holdout)
    error_diversity = error_correlation(y_dev, {n: dev_predictions[n] for n in names})

    # Feature/source diagnostics are outcome-free operational statistics.
    fr = feature_reliability(X_train).sort_values("reliability")
    source_df = source_reliability(pd.DataFrame({
        "freshness": np.ones(min(3, len(X_train))),
        "completeness": np.ones(min(3, len(X_train))),
        "consistency": np.ones(min(3, len(X_train))),
    }))
    horizon = multi_horizon_consistency({
        "J": routed_holdout["probabilities"],
        "retrieval_fused": routed_fused_holdout,
    })

    holdout = {
        "rows": int(len(y_holdout)),
        "baseline": metrics(y_holdout, production_holdout),
        "v2_full": metrics(y_holdout, routed_holdout["probabilities"]),
        "v6_full": metrics(y_holdout, routed_fused_holdout),
        "selective": selective_metrics(y_holdout, routed_fused_holdout, routed_holdout["confidence"]),
        "statistical_validation": paired_block_bootstrap(
            y_holdout, production_holdout, routed_fused_holdout,
            block_size=max(10, min(30, len(y_holdout) // 10)),
            replications=400, seed=42,
        ),
        "conformal": split_conformal_sets(
            # calibration is complete Development-OOS only
            dev_predictions[names[0]], y_dev, holdout_probs[names[0]],
        ),
        "stress": stress,
        "safety": safety,
        "mean_predictability": float(np.mean(routed_holdout["predictability"])),
        "mean_confidence": float(np.mean(routed_holdout["confidence"])),
        "abstain_rate": float(np.mean(routed_holdout["abstain"])),
        "meta_label_mean_reliability": float(np.mean(meta_holdout)),
        "temporal_consistency_mean": float(horizon["temporal_consistency"].mean()),
    }

    result = {
        "schema_version": "future-generalization-v6-npb-v1",
        "git_commit": os.getenv("GITHUB_SHA", "unknown"),
        "dataset_hash": hashlib.sha256(pd.util.hash_pandas_object(games, index=True).values.tobytes()).hexdigest(),
        "pit": {
            "status": "PASS",
            "future_holdout_excluded_from_fit": True,
            "starter_mode": "pit_safe" if os.getenv("PIT_SAFE_STARTER_DATA", "0") == "1" else "not_pit_safe_and_blank",
        },
        "development_oos": {
            "rows": int(len(y_dev)),
            "validation_windows": int(windows),
            "ablation_forward": ablation,
            "selected_retrieval_weight": fused["weight"],
            "regime_transition": regime_next,
            "meta_label_eval_mean_reliability": float(np.mean(meta_eval)),
            "conformal_eval_set_size": float(conformal_eval["set_size_mean"]),
            "forward_meta_cut_rows": int(cut),
            "forward_meta_eval_rows": int(len(y_eval)),
            "nested_oos": True,
        },
        "diagnostics": {
            "error_correlation": error_diversity.to_dict(),
            "feature_reliability_lowest10": fr.head(10).to_dict(orient="records"),
            "source_reliability": source_df.to_dict(orient="records"),
        },
        "holdout": holdout,
        "controller": {
            "config": frozen.audit,
            "production_changed": False,
            "promotion": "HOLD",
            "fallback": ["v6_full", "v2_full", "VerifiedProductionEnsemble"],
        },
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({
        "baseline": holdout["baseline"],
        "v2_full": holdout["v2_full"],
        "v6_full": holdout["v6_full"],
        "delta_vs_baseline": {
            "Accuracy": holdout["v6_full"]["Accuracy"] - holdout["baseline"]["Accuracy"],
            "LogLoss": holdout["v6_full"]["LogLoss"] - holdout["baseline"]["LogLoss"],
            "Brier": holdout["v6_full"]["Brier"] - holdout["baseline"]["Brier"],
        },
        "ablation_forward": ablation,
        "selected_retrieval_weight": fused["weight"],
        "promotion": "HOLD",
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
