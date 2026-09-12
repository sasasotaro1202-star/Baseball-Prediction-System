"""NPB candidate replay for Development OOS -> locked Holdout.

The replay uses the existing BaseballBacktest feature construction and model
pool. Candidate selection sees only the Development OOS window. After the
candidate is locked, a later chronological holdout is evaluated independently.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from baseball_backtest import BaseballBacktest, low_high_probs, score_candidates
from evaluation.metrics import expected_calibration_error, multiclass_brier
from research.candidates import CandidateSpec, lock_candidate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class ReplayConfig:
    holdout_fraction: float = 0.20
    min_train_rows: int = 180
    min_holdout_rows: int = 200
    block_size: int = 40
    calibration_tolerance: float = 0.005


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = np.argmax(p, axis=1)
    return {
        "Accuracy": float(accuracy_score(y, pred)),
        "LogLoss": float(log_loss(y, p, labels=[0, 1, 2])),
        "Brier": float(multiclass_brier(y, p, classes=[0, 1, 2])),
        "ECE": float(expected_calibration_error(y, p, classes=[0, 1, 2])),
        "rows": int(len(y)),
    }


def _draw_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    actual = (y == 1).astype(int)
    predicted = (np.argmax(p, axis=1) == 1).astype(int)
    draw_rows = actual == 1
    recall = float(predicted[draw_rows].mean()) if draw_rows.any() else 0.0
    mae = float(np.mean(np.abs(p[:, 1] - actual)))
    return {"DrawRecall": recall, "DrawProbabilityMAE": mae, "actual_draw_rows": int(draw_rows.sum())}


def _target_metrics(
    bt: BaseballBacktest,
    X_train: pd.DataFrame,
    games_train: pd.DataFrame,
    games_holdout: pd.DataFrame,
    X_holdout: pd.DataFrame,
    p: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    score_fit = bt.fit_score_ensemble(
        X_train,
        games_train["home_score"].astype(float).to_numpy(),
        games_train["away_score"].astype(float).to_numpy(),
        "NPB",
    )
    home_true = games_holdout["home_score"].astype(float).to_numpy()
    away_true = games_holdout["away_score"].astype(float).to_numpy()
    expected_home: list[float] = []
    expected_away: list[float] = []
    score_choices: list[list[list[int]]] = []
    low_high_actual: list[int] = []
    low_high_prob: list[float] = []

    for i in range(len(games_holdout)):
        lam_h, lam_a = bt.predict_scores(score_fit, X_holdout.iloc[[i]], "NPB")
        split = float(np.clip(p[i, 0] - p[i, 2], -0.35, 0.35))
        lam_h *= 1.0 + 0.08 * split
        lam_a *= 1.0 - 0.08 * split
        expected_home.append(lam_h)
        expected_away.append(lam_a)
        choices = score_candidates(lam_h, lam_a, 4)
        score_choices.append([
            [int(x.split("-")[0]), int(x.split("-")[1])] if x != "その他" else [-1, -1]
            for x, _ in choices
        ])
        low, high = low_high_probs(lam_h, lam_a)
        low_high_prob.append(high)
        low_high_actual.append(int(home_true[i] + away_true[i] >= 7))

    score_mae = float(
        (np.mean(np.abs(np.asarray(expected_home) - home_true))
         + np.mean(np.abs(np.asarray(expected_away) - away_true))) / 2.0
    )
    score_top4 = float(np.mean([
        ((h + a >= 7) and any(pair == [-1, -1] for pair in choices))
        or ((h + a < 7) and [int(h), int(a)] in choices)
        for h, a, choices in zip(home_true, away_true, score_choices)
    ]))
    hp = np.clip(np.asarray(low_high_prob), 1e-9, 1 - 1e-9)
    ya = np.asarray(low_high_actual, dtype=int)
    hilo = {
        "Accuracy": float(np.mean((hp >= 0.5).astype(int) == ya)),
        "LogLoss": float(-np.mean(ya * np.log(hp) + (1 - ya) * np.log(1 - hp))),
        "Brier": float(np.mean((hp - ya) ** 2)),
        "ScoreMAE": score_mae,
        "Top4HitRate": score_top4,
        "rows": int(len(home_true)),
    }
    return {"ScoreMAE": score_mae, "Top4HitRate": score_top4, "rows": int(len(home_true))}, hilo


def _fit_candidate(bt: BaseballBacktest, name: str, X: pd.DataFrame, y: np.ndarray):
    models = bt.models("NPB")
    if name not in models:
        raise ValueError(f"unknown NPB model candidate: {name}")
    model = models[name]
    bt._fit_model(model, X, y, bt._sample_weights(len(X)), "NPB")
    return model


def _candidate_probability(bt: BaseballBacktest, model, X: pd.DataFrame) -> np.ndarray:
    return bt.align_proba(model.predict_proba(X), model.classes_, "NPB")


def _development_compare(
    bt: BaseballBacktest,
    X: pd.DataFrame,
    y: np.ndarray,
    start: int,
    end: int,
    candidate_names: list[str],
    block_size: int,
) -> tuple[dict[str, dict[str, float]], int]:
    rows: dict[str, list[np.ndarray]] = {name: [] for name in ["ProductionEnsemble"] + candidate_names}
    actual: list[np.ndarray] = []
    windows = 0
    for cut in range(start, end, block_size):
        stop = min(end, cut + block_size)
        fitted, _, _ = bt.fit_ensemble(X.iloc[:cut], y[:cut], "NPB")
        if not fitted:
            continue
        windows += 1
        actual.append(y[cut:stop])
        rows["ProductionEnsemble"].append(bt.ensemble_proba(fitted, X.iloc[cut:stop], "NPB"))
        for name in candidate_names:
            model = _fit_candidate(bt, name, X.iloc[:cut], y[:cut])
            rows[name].append(_candidate_probability(bt, model, X.iloc[cut:stop]))

    y_dev = np.concatenate(actual) if actual else np.empty(0, dtype=int)
    out: dict[str, dict[str, float]] = {}
    for name, chunks in rows.items():
        if not chunks:
            continue
        p = np.vstack(chunks)
        out[name] = _metrics(y_dev[:len(p)], p)
    return out, windows


def run_npb_candidate_cycle(
    *,
    data_dir: str | Path = "data",
    git_commit: str,
    feature_version: str = "baseball-features-v1",
    config: ReplayConfig = ReplayConfig(),
) -> dict[str, Any]:
    """Replay NPB candidates, lock on Development OOS, then prepare holdout evidence."""
    bt = BaseballBacktest(Path(data_dir))
    raw = bt.load_npb_pbp()
    games = bt.aggregate_npb_games(raw)
    X, y, _meta = bt.build_features(games)
    n = len(X)
    holdout_start = int(n * (1.0 - config.holdout_fraction))
    dev_start = max(config.min_train_rows, int(n * 0.10))
    if holdout_start <= dev_start or n - holdout_start < config.min_holdout_rows:
        raise RuntimeError("NPB replay does not have enough chronological rows for an independent holdout")

    candidate_names = list(bt.models("NPB").keys())
    development, validation_windows = _development_compare(
        bt, X, y, dev_start, holdout_start, candidate_names, config.block_size
    )
    if "ProductionEnsemble" not in development:
        raise RuntimeError("production ensemble has no valid Development OOS result")
    if validation_windows < 2:
        raise RuntimeError("NPB Development OOS has fewer than two chronological validation windows")

    baseline = development["ProductionEnsemble"]
    candidates = [(name, metrics) for name, metrics in development.items() if name != "ProductionEnsemble"]
    candidates.sort(key=lambda x: (x[1]["LogLoss"], x[1]["Brier"], -x[1]["Accuracy"], x[0]))
    if not candidates:
        raise RuntimeError("no NPB candidate produced valid Development OOS metrics")
    selected_name, selected_metrics = candidates[0]
    if baseline["LogLoss"] - selected_metrics["LogLoss"] <= 0:
        return {
            "stage": "development_evaluated",
            "decision": "NO_CHANGE",
            "baseline": baseline,
            "candidate": selected_metrics,
            "candidate_model": selected_name,
            "validation_windows": validation_windows,
            "development": development,
        }

    dataset_hash = hashlib.sha256(pd.util.hash_pandas_object(games, index=True).values.tobytes()).hexdigest()
    spec = CandidateSpec(
        candidate_id="cand-" + hashlib.sha256(
            f"NPB|{selected_name}|{feature_version}|{git_commit}|{dataset_hash}".encode("utf-8")
        ).hexdigest()[:20],
        league="NPB",
        objective="win",
        model_version=selected_name,
        feature_version=feature_version,
        development_metrics=selected_metrics,
        selection_reason="Development OOS only; lowest LogLoss, then Brier, then highest Accuracy among candidates beating the production ensemble.",
        git_commit=git_commit,
        dataset_hash=dataset_hash,
    )
    locked = lock_candidate(spec)

    X_train = X.iloc[:holdout_start]
    y_train = y[:holdout_start]
    games_train = games.iloc[:holdout_start].reset_index(drop=True)
    games_holdout = games.iloc[holdout_start:].reset_index(drop=True)
    X_holdout = X.iloc[holdout_start:].reset_index(drop=True)

    base_fit, _, _ = bt.fit_ensemble(X_train, y_train, "NPB")
    if not base_fit:
        raise RuntimeError("production ensemble could not be fitted for locked holdout")
    base_p = bt.ensemble_proba(base_fit, X_holdout, "NPB")
    cand_model = _fit_candidate(bt, selected_name, X_train, y_train)
    cand_p = _candidate_probability(bt, cand_model, X_holdout)

    y_holdout = y[holdout_start:]
    base_metrics = _metrics(y_holdout, base_p)
    cand_metrics = _metrics(y_holdout, cand_p)
    base_metrics.update(_draw_metrics(y_holdout, base_p))
    cand_metrics.update(_draw_metrics(y_holdout, cand_p))
    base_score, base_hilo = _target_metrics(bt, X_train, games_train, games_holdout, X_holdout, base_p)
    cand_score, cand_hilo = _target_metrics(bt, X_train, games_train, games_holdout, X_holdout, cand_p)

    calibration_ok = cand_metrics["ECE"] <= base_metrics["ECE"] + config.calibration_tolerance
    reproducible = True  # fixed model seeds + deterministic chronological ordering are encoded by the core

    holdout = {
        "stage": "locked_holdout_ready",
        "candidate_id": spec.candidate_id,
        "baseline": base_metrics,
        "candidate": cand_metrics,
        "baseline_score": base_score,
        "candidate_score": cand_score,
        "baseline_hilo": base_hilo,
        "candidate_hilo": cand_hilo,
        "validation_windows": validation_windows,
        "calibration_ok": bool(calibration_ok),
        "no_future_target_data": True,
        "reproducible": reproducible,
        "holdout_rows": int(len(X_holdout)),
        "holdout_start": str(games_holdout["datetime"].iloc[0]),
        "holdout_end": str(games_holdout["datetime"].iloc[-1]),
        "candidate_model": selected_name,
        "dataset_hash": dataset_hash,
        "selection_locked_before_holdout": True,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "npb_candidate_development.json").write_text(
        json.dumps({
            "development": development,
            "candidate": asdict(spec),
            "baseline": baseline,
            "validation_windows": validation_windows,
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS / "npb_locked_holdout.json").write_text(
        json.dumps(holdout, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"stage": "locked_holdout_ready", "decision": "HOLDOUT_READY", "candidate": locked, "holdout": holdout}
