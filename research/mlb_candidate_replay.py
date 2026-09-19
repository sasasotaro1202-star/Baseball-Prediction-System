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

from baseball_backtest import BaseballBacktest, low_high_probs, score_candidates
from core.atomic_io import atomic_write_json
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
    retrain_every: int = 150


def _proba(bt: BaseballBacktest, model: Any, X):
    return bt.align_proba(model.predict_proba(X), model.classes_, "MLB")


def _fit(bt: BaseballBacktest, name: str, X, y):
    models = bt.models("MLB")
    if name not in models:
        raise ValueError(f"unknown MLB model candidate: {name}")
    model = models[name]
    bt._fit_model(model, X, y, bt._sample_weights(len(X)), "MLB")
    return model


def _target_metrics(bt, X_train, games_train, games_eval, X_eval, p):
    score_fit = bt.fit_score_ensemble(
        X_train,
        games_train["home_score"].astype(float).to_numpy(),
        games_train["away_score"].astype(float).to_numpy(),
        "MLB",
    )
    home_true = games_eval["home_score"].astype(float).to_numpy()
    away_true = games_eval["away_score"].astype(float).to_numpy()
    expected_home, expected_away, score_choices = [], [], []
    hilo_actual, hilo_prob = [], []
    for i in range(len(games_eval)):
        lam_h, lam_a = bt.predict_scores(score_fit, X_eval.iloc[[i]], "MLB")
        split = float(np.clip(p[i, 0] - 0.5, -0.35, 0.35))
        lam_h *= 1.0 + 0.08 * split
        lam_a *= 1.0 - 0.08 * split
        expected_home.append(lam_h)
        expected_away.append(lam_a)
        choices = score_candidates(lam_h, lam_a, 4)
        score_choices.append(choices)
        _, high = low_high_probs(lam_h, lam_a)
        hilo_prob.append(high)
        hilo_actual.append(int(home_true[i] + away_true[i] >= 7))
    eh, ea = np.asarray(expected_home), np.asarray(expected_away)
    score_mae = float((np.mean(np.abs(eh-home_true)) + np.mean(np.abs(ea-away_true))) / 2.0)
    score_top4 = float(np.mean([
        (h + a >= 7 and any(x == "その他" for x, _ in choices))
        or (h + a < 7 and any(x == f"{int(h)}-{int(a)}" for x, _ in choices))
        for h, a, choices in zip(home_true, away_true, score_choices)
    ]))
    hp = np.clip(np.asarray(hilo_prob), 1e-9, 1-1e-9)
    ya = np.asarray(hilo_actual)
    return {"ScoreMAE": score_mae, "Top4HitRate": score_top4, "rows": int(len(home_true))}, {
        "Accuracy": float(np.mean((hp >= 0.5).astype(int) == ya)),
        "LogLoss": float(-np.mean(ya*np.log(hp) + (1-ya)*np.log(1-hp))),
        "Brier": float(np.mean((hp-ya)**2)),
        "rows": int(len(home_true)),
    }


def _development(bt, X, y, start, end, names, block, retrain_every):
    actual, outputs = [], {"ProductionEnsemble": []}
    for name in names:
        outputs[name] = []
    if block <= 0 or retrain_every <= 0:
        raise ValueError("block and retrain_every must be > 0")
    windows = 0
    fitted = None
    candidate_fitted = {}
    last_fit_cut = -10**9
    for cut in range(start, end, block):
        stop = min(end, cut + block)
        # Periodic retraining remains strictly chronological: every reused
        # model was trained only on observations before its first OOS block.
        if fitted is None or cut - last_fit_cut >= retrain_every:
            fitted, _, _ = bt.fit_ensemble(X.iloc[:cut], y[:cut], "MLB")
            if not fitted:
                continue
            candidate_fitted = {
                name: _fit(bt, name, X.iloc[:cut], y[:cut])
                for name in names
            }
            last_fit_cut = cut
        windows += 1
        actual.append(y[cut:stop])
        outputs["ProductionEnsemble"].append(bt.ensemble_proba(fitted, X.iloc[cut:stop], "MLB"))
        for name in names:
            outputs[name].append(_proba(bt, candidate_fitted[name], X.iloc[cut:stop]))
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
    development, windows = _development(bt, X, y, dev_start, holdout_start, names, config.block_size, config.retrain_every)
    baseline = development["ProductionEnsemble"]
    candidates = [(k, v) for k, v in development.items() if k != "ProductionEnsemble"]
    candidates.sort(key=lambda kv: (kv[1]["LogLoss"], kv[1]["Brier"], -kv[1]["Accuracy"], kv[0]))
    selected_name, selected_metrics = candidates[0]
    if baseline["LogLoss"] - selected_metrics["LogLoss"] <= 0:
        result = {"stage": "development_evaluated", "decision": "NO_CHANGE", "baseline": baseline,
                  "candidate": selected_metrics, "candidate_model": selected_name, "validation_windows": windows}
        atomic_write_json(RESULTS / "mlb_candidate_development.json", result)
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
    atomic_write_json(RESULTS / "mlb_candidate_development.json", development)
    atomic_write_json(RESULTS / "mlb_locked_holdout.json", out)
    return out
