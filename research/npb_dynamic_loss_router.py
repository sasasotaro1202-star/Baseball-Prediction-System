"""Research-only dynamic model weighting for NPB.

Each expert gets a meta-model that predicts the loss it is likely to incur for
the current game. The expert weights are then adjusted per game from those
predicted losses. Crucially, meta-models are trained only on earlier
out-of-sample rows, never on the current prediction block or holdout.

This is inspired by arbitrated dynamic ensembles / mixture-of-experts, but is
implemented conservatively for the Baseball repository:
- PIT/OOS chronology is preserved.
- The current production runtime is not modified.
- Per-game weights and correctness are written for auditability.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

from baseball_backtest import BaseballBacktest, low_high_probs, score_candidates
from evaluation.metrics import expected_calibration_error, multiclass_brier

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DEFAULT_EXPERTS = (
    "RandomForest",
    "ExtraTrees",
    "CatBoost",
    "KNNAnalog",
    "XGBoost",
)

CONTEXT_COLUMNS = (
    "d_elo",
    "starter_x_quality_proxy",
    "starter_kbb_gap",
    "starter_recent_form_gap",
    "d_gd_10",
    "d_win_10",
    "d_draw_10",
    "d_matches",
    "d_rest_days",
    "expected_env",
    "offense_power_gap_10",
    "offense_walk_gap_10",
    "offense_contact_gap_10",
    "bullpen_fatigue_diff",
    "bullpen_quality_era_diff",
    "bullpen_whip_diff",
    "run_volatility_gap_20",
    "run_trend_gap_20",
    "d_bat_avg_10",
    "d_bat_so_rate_10",
    "d_bp_app_10",
)


def _stable_context_columns(X: pd.DataFrame) -> list[str]:
    cols = [c for c in CONTEXT_COLUMNS if c in X.columns]
    if len(cols) < 5:
        numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        cols = numeric[: min(30, len(numeric))]
    if len(cols) < 5:
        raise RuntimeError("dynamic router has insufficient numeric context features")
    return cols


def _feature_matrix(
    X: pd.DataFrame,
    expert_probabilities: dict[str, np.ndarray],
    expert_order: list[str],
    context_columns: list[str],
) -> np.ndarray:
    context = X.loc[:, context_columns].to_numpy(dtype=float)
    parts = [context]
    for name in expert_order:
        p = np.asarray(expert_probabilities[name], dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) != len(X):
            raise ValueError(f"invalid expert probabilities for {name}")
        p = np.clip(p, 1e-6, 1 - 1e-6)
        parts.append(p)
        parts.append(np.log(p))
        entropy = -np.sum(p * np.log(p), axis=1, keepdims=True)
        parts.append(entropy)
    out = np.hstack(parts)
    if not np.isfinite(out).all():
        raise ValueError("dynamic router feature matrix contains non-finite values")
    return out


class DynamicLossRouter:
    """Predict each expert's future log-loss and turn it into per-row weights."""

    def __init__(
        self,
        *,
        alpha: float = 10.0,
        min_train_rows: int = 300,
        softmax_temperature: float = 1.0,
        uniform_floor: float = 0.15,
    ) -> None:
        if alpha <= 0 or not np.isfinite(alpha):
            raise ValueError("alpha must be positive and finite")
        if min_train_rows < 1:
            raise ValueError("min_train_rows must be >= 1")
        if softmax_temperature <= 0 or not np.isfinite(softmax_temperature):
            raise ValueError("softmax_temperature must be positive and finite")
        if not (0.0 <= uniform_floor < 1.0):
            raise ValueError("uniform_floor must be in [0,1)")
        self.alpha = float(alpha)
        self.min_train_rows = int(min_train_rows)
        self.softmax_temperature = float(softmax_temperature)
        self.uniform_floor = float(uniform_floor)
        self.models: dict[str, Pipeline] = {}
        self.expert_order: list[str] = []
        self.context_columns: list[str] = []
        self.fitted_rows = 0

    @staticmethod
    def _row_loss(probabilities: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = np.asarray(probabilities, dtype=float)
        yy = np.asarray(y, dtype=int)
        if p.shape != (len(yy), 3):
            raise ValueError("probabilities/labels shape mismatch")
        chosen = np.clip(p[np.arange(len(yy)), yy], 1e-8, 1.0)
        return -np.log(chosen)

    def fit(
        self,
        X_history: pd.DataFrame,
        y_history: np.ndarray,
        expert_probabilities: dict[str, np.ndarray],
        expert_order: list[str],
    ) -> "DynamicLossRouter":
        if len(X_history) != len(y_history):
            raise ValueError("router history X/y length mismatch")
        if len(X_history) < self.min_train_rows:
            raise ValueError("router history is too short")
        self.context_columns = _stable_context_columns(X_history)
        self.expert_order = list(expert_order)
        if len(self.expert_order) < 2:
            raise ValueError("dynamic router needs at least two experts")
        Z = _feature_matrix(
            X_history,
            expert_probabilities,
            self.expert_order,
            self.context_columns,
        )
        yy = np.asarray(y_history, dtype=int)
        self.models = {}
        for name in self.expert_order:
            target = self._row_loss(expert_probabilities[name], yy)
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=self.alpha)),
                ]
            )
            model.fit(Z, target)
            self.models[name] = model
        self.fitted_rows = len(yy)
        return self

    def predict_weights(
        self,
        X: pd.DataFrame,
        expert_probabilities: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        if not self.models:
            raise RuntimeError("dynamic router is not fitted")
        Z = _feature_matrix(
            X,
            expert_probabilities,
            self.expert_order,
            self.context_columns,
        )
        predicted_losses = np.vstack(
            [self.models[name].predict(Z) for name in self.expert_order]
        ).T
        predicted_losses = np.nan_to_num(
            predicted_losses, nan=10.0, posinf=10.0, neginf=10.0
        )
        predicted_losses = np.clip(predicted_losses, -2.0, 10.0)
        logits = -predicted_losses / self.softmax_temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        raw = np.exp(logits)
        raw /= np.sum(raw, axis=1, keepdims=True)
        k = len(self.expert_order)
        floor = self.uniform_floor / k
        weights = floor + (1.0 - self.uniform_floor) * raw
        return {
            name: weights[:, i].copy()
            for i, name in enumerate(self.expert_order)
        }

    def predict(
        self,
        X: pd.DataFrame,
        expert_probabilities: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        weights = self.predict_weights(X, expert_probabilities)
        p = np.zeros((len(X), 3), dtype=float)
        for name in self.expert_order:
            p += weights[name][:, None] * np.asarray(expert_probabilities[name], dtype=float)
        p = np.maximum(p, 1e-12)
        p /= p.sum(axis=1, keepdims=True)
        return p, weights


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(p, axis=1)
    return {
        "Accuracy": float(accuracy_score(y, pred)),
        "LogLoss": float(log_loss(y, p, labels=[0, 1, 2])),
        "Brier": float(multiclass_brier(y, p, classes=[0, 1, 2])),
        "ECE": float(expected_calibration_error(y, p, classes=[0, 1, 2])),
        "DrawRecall": float(
            np.mean(pred[np.asarray(y) == 1] == 1)
            if np.any(np.asarray(y) == 1)
            else 0.0
        ),
        "rows": int(len(y)),
    }


def _expert_predictions_from_fit(
    bt: BaseballBacktest,
    fitted: list[tuple[Any, float, str]],
    X: pd.DataFrame,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for model, _weight, name in fitted:
        if name not in DEFAULT_EXPERTS:
            continue
        raw = model.predict_proba(X)
        out[name] = bt.align_proba(raw, model.classes_, "NPB")
    return out


def _weighted_top_score_metrics(
    bt: BaseballBacktest,
    X_train: pd.DataFrame,
    games_train: pd.DataFrame,
    games_eval: pd.DataFrame,
    X_eval: pd.DataFrame,
    p: np.ndarray,
) -> dict[str, float]:
    score_fit = bt.fit_score_ensemble(
        X_train,
        games_train["home_score"].astype(float).to_numpy(),
        games_train["away_score"].astype(float).to_numpy(),
        "NPB",
    )
    true_h = games_eval["home_score"].astype(float).to_numpy()
    true_a = games_eval["away_score"].astype(float).to_numpy()
    expected_h: list[float] = []
    expected_a: list[float] = []
    top4_hits: list[bool] = []
    hilo_probs: list[float] = []
    hilo_actual: list[int] = []
    for i in range(len(games_eval)):
        lh, la, shared = bt.predict_scores(score_fit, X_eval.iloc[[i]], "NPB")
        split = float(np.clip(p[i, 0] - p[i, 2], -0.35, 0.35))
        lh *= 1.0 + 0.08 * split
        la *= 1.0 - 0.08 * split
        expected_h.append(lh)
        expected_a.append(la)
        choices = score_candidates(lh, la, shared)
        labels = {label for label, _ in choices}
        top4_hits.append(
            (true_h[i] + true_a[i] < 7)
            and (f"{int(true_h[i])}-{int(true_a[i])}" in labels)
        )
        _low, high = low_high_probs(lh, la, shared)
        hilo_probs.append(float(high))
        hilo_actual.append(int(true_h[i] + true_a[i] >= 7))
    score_mae = float(
        (
            np.mean(np.abs(np.asarray(expected_h) - true_h))
            + np.mean(np.abs(np.asarray(expected_a) - true_a))
        )
        / 2.0
    )
    hp = np.clip(np.asarray(hilo_probs), 1e-9, 1.0 - 1e-9)
    ya = np.asarray(hilo_actual, dtype=int)
    hilo_acc = float(np.mean((hp >= 0.5).astype(int) == ya))
    return {
        "ScoreMAE": score_mae,
        "Top4HitRate": float(np.mean(top4_hits)),
        "LowHighAccuracy": hilo_acc,
    }


def _run_dynamic_development(
    bt: BaseballBacktest,
    X: pd.DataFrame,
    y: np.ndarray,
    dev_start: int,
    holdout_start: int,
    development_predictions: dict[str, np.ndarray],
    block_size: int,
    retrain_every: int,
    min_train_rows: int,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    expert_order = [name for name in DEFAULT_EXPERTS if name in development_predictions]
    if len(expert_order) < 2:
        raise RuntimeError(f"insufficient expert pool for dynamic routing: {expert_order}")

    start = dev_start
    stop = holdout_start
    dev_X = X.iloc[start:stop].reset_index(drop=True)
    dev_y = np.asarray(y[start:stop], dtype=int)
    expert_dev = {
        name: np.asarray(development_predictions[name], dtype=float)
        for name in expert_order
    }
    baseline_dev = np.asarray(development_predictions["ProductionEnsemble"], dtype=float)
    pred_chunks: list[np.ndarray] = []
    weight_rows: list[dict[str, Any]] = []
    active = 0
    router: DynamicLossRouter | None = None
    last_fit = 0
    for cut in range(0, len(dev_X), block_size):
        end = min(len(dev_X), cut + block_size)
        if router is None or cut - last_fit >= retrain_every:
            if cut >= min_train_rows:
                hist_slice = slice(0, cut)
                router = DynamicLossRouter(min_train_rows=min_train_rows)
                hist_probs = {
                    name: expert_dev[name][hist_slice]
                    for name in expert_order
                }
                router.fit(dev_X.iloc[hist_slice], dev_y[hist_slice], hist_probs, expert_order)
                last_fit = cut
            else:
                router = None
        cur_probs = {name: expert_dev[name][cut:end] for name in expert_order}
        if router is None:
            pred = baseline_dev[cut:end]
            weights = {
                name: np.full(end - cut, np.nan, dtype=float)
                for name in expert_order
            }
            active_flag = False
        else:
            pred, weights = router.predict(dev_X.iloc[cut:end], cur_probs)
            active += end - cut
            active_flag = True
        pred_chunks.append(pred)
        for i in range(end - cut):
            row = {
                "row_index": int(start + cut + i),
                "active_router": bool(active_flag),
                "actual_class": int(dev_y[cut + i]),
                "predicted_class": int(np.argmax(pred[i])),
                "correct": bool(int(np.argmax(pred[i])) == int(dev_y[cut + i])),
                "home_probability": float(pred[i, 0]),
                "draw_probability": float(pred[i, 1]),
                "away_probability": float(pred[i, 2]),
            }
            for name in expert_order:
                row[f"weight_{name}"] = (
                    float(weights[name][i]) if np.isfinite(weights[name][i]) else None
                )
                row[f"p_{name}_home"] = float(cur_probs[name][i, 0])
                row[f"p_{name}_draw"] = float(cur_probs[name][i, 1])
                row[f"p_{name}_away"] = float(cur_probs[name][i, 2])
            weight_rows.append(row)
    pred = np.vstack(pred_chunks)
    return {
        "full_development": _metrics(dev_y, pred),
        "active_router_rows": int(active),
        "active_router_share": float(active / max(1, len(dev_y))),
        "active_router": _metrics(
            dev_y[-active:] if active else np.empty(0, dtype=int),
            pred[-active:] if active else np.empty((0, 3), dtype=float),
        ) if active else {},
    }, pred, weight_rows


def run_experiment(data_dir: str | Path = "data") -> dict[str, Any]:
    bt = BaseballBacktest(Path(data_dir))
    raw = bt.load_npb_pbp()
    games = bt.aggregate_npb_games(raw).copy()
    if len(games) < 1000:
        raise RuntimeError(f"NPB corpus unexpectedly small: {len(games)} games")

    # Historical starter identities are not PIT-safe in ordinary public PBP
    # replays; use the same conservative candidate contract.
    games["home_starter"] = ""
    games["away_starter"] = ""
    games["confirmed_starters"] = False
    games["starter_evidence_status"] = "not_pit_safe"

    X, y, _meta = bt.build_features(games)
    n = len(X)
    holdout_start = int(n * 0.80)
    dev_start = max(180, int(n * 0.10))
    if holdout_start <= dev_start + 600:
        raise RuntimeError("insufficient chronology for dynamic router experiment")

    from research.npb_candidate_replay import _development_compare

    candidate_names = list(bt.models("NPB").keys())
    development, validation_windows, development_predictions = _development_compare(
        bt, X, y, dev_start, holdout_start,
        candidate_names, block_size=60, retrain_every=180
    )
    if "ProductionEnsemble" not in development:
        raise RuntimeError("missing ProductionEnsemble development OOS")

    summary, dev_pred, ledger = _run_dynamic_development(
        bt, X, y, dev_start, holdout_start, development_predictions,
        block_size=60, retrain_every=180, min_train_rows=300
    )
    baseline = development["ProductionEnsemble"]
    summary["baseline_production_ensemble"] = baseline
    summary["validation_windows"] = int(validation_windows)
    summary["expert_pool"] = [x for x in DEFAULT_EXPERTS if x in development_predictions]

    # Locked holdout: fit the baseline production ensemble on the pre-holdout
    # prefix, then use its actual fitted expert members as the dynamic expert
    # pool. The router itself only sees Development OOS outcomes.
    X_train = X.iloc[:holdout_start]
    y_train = y[:holdout_start]
    X_holdout = X.iloc[holdout_start:].reset_index(drop=True)
    y_holdout = np.asarray(y[holdout_start:], dtype=int)
    games_train = games.iloc[:holdout_start].reset_index(drop=True)
    games_holdout = games.iloc[holdout_start:].reset_index(drop=True)
    base_fit, _, _ = bt.fit_ensemble(X_train, y_train, "NPB")
    if not base_fit:
        raise RuntimeError("failed to fit production ensemble for holdout")

    hold_experts = _expert_predictions_from_fit(bt, base_fit, X_holdout)
    hold_order = [name for name in DEFAULT_EXPERTS if name in hold_experts]
    if len(hold_order) < 2:
        raise RuntimeError(f"holdout expert pool too small: {hold_order}")

    dev_X = X.iloc[dev_start:holdout_start].reset_index(drop=True)
    dev_y = np.asarray(y[dev_start:holdout_start], dtype=int)
    dev_probs = {
        name: np.asarray(development_predictions[name], dtype=float)
        for name in hold_order
        if name in development_predictions
    }
    common_order = [name for name in hold_order if name in dev_probs]
    if len(common_order) < 2:
        raise RuntimeError("insufficient common development/holdout expert pool")
    router = DynamicLossRouter(min_train_rows=300)
    router.fit(dev_X, dev_y, {name: dev_probs[name] for name in common_order}, common_order)
    hold_p, hold_weights = router.predict(X_holdout, {name: hold_experts[name] for name in common_order})
    hold_base_p = bt.ensemble_proba(base_fit, X_holdout, "NPB")

    summary["locked_holdout_dynamic_router"] = _metrics(y_holdout, hold_p)
    summary["locked_holdout_production_ensemble"] = _metrics(y_holdout, hold_base_p)
    summary["locked_holdout_dynamic_router_score_hilo"] = _weighted_top_score_metrics(
        bt, X_train, games_train, games_holdout, X_holdout, hold_p
    )
    summary["locked_holdout_production_score_hilo"] = _weighted_top_score_metrics(
        bt, X_train, games_train, games_holdout, X_holdout, hold_base_p
    )

    hold_ledger: list[dict[str, Any]] = []
    for i in range(len(hold_p)):
        actual = int(y_holdout[i])
        row = {
            "row_index": int(holdout_start + i),
            "active_router": True,
            "actual_class": actual,
            "predicted_class": int(np.argmax(hold_p[i])),
            "correct": bool(int(np.argmax(hold_p[i])) == actual),
            "home_probability": float(hold_p[i, 0]),
            "draw_probability": float(hold_p[i, 1]),
            "away_probability": float(hold_p[i, 2]),
        }
        for name in common_order:
            row[f"weight_{name}"] = float(hold_weights[name][i])
            row[f"p_{name}_home"] = float(hold_experts[name][i, 0])
            row[f"p_{name}_draw"] = float(hold_experts[name][i, 1])
            row[f"p_{name}_away"] = float(hold_experts[name][i, 2])
        hold_ledger.append(row)

    RESULTS.mkdir(parents=True, exist_ok=True)
    dev_csv = RESULTS / "npb_dynamic_loss_router_development_ledger.csv"
    hold_csv = RESULTS / "npb_dynamic_loss_router_holdout_ledger.csv"
    pd.DataFrame(ledger).to_csv(dev_csv, index=False)
    pd.DataFrame(hold_ledger).to_csv(hold_csv, index=False)

    summary["artifacts"] = {
        "development_ledger": str(dev_csv),
        "holdout_ledger": str(hold_csv),
    }
    (RESULTS / "npb_dynamic_loss_router.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    result = run_experiment(args.data_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
