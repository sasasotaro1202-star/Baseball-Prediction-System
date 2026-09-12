"""MLB Development-OOS -> Candidate-Lock -> Independent-Holdout lifecycle.

This is the MLB counterpart to the existing NPB replay. It uses the existing
BaseballBacktest model pool, selects candidates only on chronological
Development OOS, locks the candidate, then evaluates it on a later holdout.
Promotion is fail-closed when score/Low-High evidence is unavailable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from baseball_backtest import BaseballBacktest
from evaluation.metrics import classification_metrics
from research.candidates import CandidateSpec, lock_candidate
from research.validation_pipeline import run_validation_pipeline

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class MLBReplayConfig:
    holdout_fraction: float = 0.20
    min_train_rows: int = 180
    min_holdout_rows: int = 200
    block_size: int = 40


def _proba(bt: BaseballBacktest, model: Any, X):
    return bt.align_proba(model.predict_proba(X), model.classes_, "MLB")


def _fit(bt: BaseballBacktest, name: str, X, y):
    models = bt.models("MLB")
    if name not in models:
        raise ValueError(f"unknown MLB model candidate: {name}")
    model = models[name]
    bt._fit_model(model, X, y, bt._sample_weights(len(X)), "MLB")
    return model


def _development(bt, X, y, start, end, names, block):
    actual, outputs = [], {"ProductionEnsemble": []}
    for name in names:
        outputs[name] = []
    windows = 0
    for cut in range(start, end, block):
        stop = min(end, cut + block)
        fitted, _, _ = bt.fit_ensemble(X.iloc[:cut], y[:cut], "MLB")
        if not fitted:
            continue
        windows += 1
        actual.append(y[cut:stop])
        outputs["ProductionEnsemble"].append(bt.ensemble_proba(fitted, X.iloc[cut:stop], "MLB"))
        for name in names:
            outputs[name].append(_proba(bt, _fit(bt, name, X.iloc[:cut], y[:cut]), X.iloc[cut:stop]))
    if not actual:
        raise RuntimeError("MLB Development OOS produced no valid windows")
    yy = np.concatenate(actual)
    metrics = {}
    for name, chunks in outputs.items():
        if chunks:
            pp = np.vstack(chunks)
            metrics[name] = classification_metrics(yy[:len(pp)], pp, classes=[0, 1])
    return metrics, windows


def run_mlb_candidate_cycle(*, data_dir: str | Path = "data", git_commit: str,
                            mlb_start: int = 2020, mlb_end: int = 2026,
                            feature_version: str = "baseball-features-v1",
                            config: MLBReplayConfig = MLBReplayConfig()) -> dict[str, Any]:
    bt = BaseballBacktest(Path(data_dir))
    games = bt.load_mlb(mlb_start, mlb_end)
    X, y, _meta = bt.build_features(games)
    n = len(X)
    holdout_start = int(n * (1.0 - config.holdout_fraction))
    dev_start = max(config.min_train_rows, int(n * 0.10))
    if holdout_start <= dev_start or n - holdout_start < config.min_holdout_rows:
        raise RuntimeError("MLB replay does not have enough rows for an independent holdout")

    names = list(bt.models("MLB").keys())
    development, windows = _development(bt, X, y, dev_start, holdout_start, names, config.block_size)
    baseline = development["ProductionEnsemble"]
    candidates = [(k, v) for k, v in development.items() if k != "ProductionEnsemble"]
    candidates.sort(key=lambda kv: (kv[1]["LogLoss"], kv[1]["Brier"], -kv[1]["Accuracy"], kv[0]))
    selected_name, selected_metrics = candidates[0]
    if baseline["LogLoss"] - selected_metrics["LogLoss"] <= 0:
        result = {"stage": "development_evaluated", "decision": "NO_CHANGE", "baseline": baseline,
                  "candidate": selected_metrics, "candidate_model": selected_name, "validation_windows": windows}
        (RESULTS / "mlb_candidate_development.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    dataset_hash = hashlib.sha256(str(len(games)).encode() + str(games.index.tolist()).encode()).hexdigest()
    spec = CandidateSpec(
        candidate_id="cand-" + hashlib.sha256(f"MLB|{selected_name}|{feature_version}|{git_commit}|{dataset_hash}".encode()).hexdigest()[:20],
        league="MLB", objective="win", model_version=selected_name, feature_version=feature_version,
        development_metrics=selected_metrics,
        selection_reason="Development OOS only; lowest LogLoss, then Brier, then highest Accuracy.",
        git_commit=git_commit, dataset_hash=dataset_hash,
    )
    locked = lock_candidate(spec)

    X_train, X_holdout = X.iloc[:holdout_start], X.iloc[holdout_start:]
    y_train, y_holdout = y[:holdout_start], y[holdout_start:]
    base_fit, _, _ = bt.fit_ensemble(X_train, y_train, "MLB")
    if not base_fit:
        raise RuntimeError("MLB production ensemble could not be fitted for holdout")
    base_p = bt.ensemble_proba(base_fit, X_holdout, "MLB")
    cand_p = _proba(bt, _fit(bt, selected_name, X_train, y_train), X_holdout)
    base = classification_metrics(y_holdout, base_p, classes=[0, 1])
    cand = classification_metrics(y_holdout, cand_p, classes=[0, 1])

    # Score and market-line checks are intentionally fail-closed until the
    # production OOS artifact contains PIT historical line evidence.
    lifecycle = run_validation_pipeline(
        candidate_id=spec.candidate_id,
        development_metrics=selected_metrics,
        holdout_baseline=base,
        holdout_candidate=cand,
        validation_windows=windows,
        calibration_ok=cand["ECE"] <= base["ECE"] + 0.005,
        no_future_target_data=True,
        reproducible=True,
        holdout_score_baseline=None,
        holdout_score_candidate=None,
        holdout_hilo_baseline=None,
        holdout_hilo_candidate=None,
        league="MLB",
    )
    # The common policy requires score/Low-High evidence, so this remains
    # non-promotable until those datasets are actually wired in.
    out = {"stage": "locked_holdout_evaluated", "candidate": locked,
           "holdout": {"baseline": base, "candidate": cand},
           "validation": asdict(lifecycle), "decision": lifecycle.decision,
           "score_hilo_status": "REQUIRED_EVIDENCE_NOT_CONNECTED"}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "mlb_candidate_development.json").write_text(json.dumps(development, indent=2), encoding="utf-8")
    (RESULTS / "mlb_locked_holdout.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
