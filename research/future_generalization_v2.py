"""Research-only Future Generalization v2 control layer.

This module is deliberately decoupled from the Production pipeline. It provides:
- model disagreement features
- predictability score
- chronological future-failure risk models
- distribution drift / regime-shift scoring
- dynamic soft routing with weight smoothing
- temperature-neutral probability validation
- selective prediction / abstention metrics
- fail-closed safety fallback
- paired block-bootstrap validation helpers

All fit operations consume a strictly chronological development stream. Prediction
methods do not accept labels, preventing accidental access to future outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


EPS = 1e-12


def _validate_probs(model_probs: Mapping[str, np.ndarray], n_rows: int | None = None) -> tuple[list[str], int, int]:
    if not model_probs:
        raise ValueError("model_probs must not be empty")
    names = list(model_probs)
    arrays = [np.asarray(model_probs[name], dtype=float) for name in names]
    shape = arrays[0].shape
    if len(shape) != 2 or shape[1] < 2:
        raise ValueError("each probability matrix must be 2D with at least two classes")
    if any(a.shape != shape for a in arrays):
        raise ValueError("all model probability matrices must share the same shape")
    if n_rows is not None and shape[0] != int(n_rows):
        raise ValueError("probability row count mismatch")
    for name, a in zip(names, arrays):
        if not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(a > 1):
            raise ValueError(f"invalid probability matrix: {name}")
        sums = a.sum(axis=1)
        if np.any(sums <= 0):
            raise ValueError(f"non-positive probability mass: {name}")
    return names, shape[0], shape[1]


def normalize_probs(p: np.ndarray) -> np.ndarray:
    q = np.asarray(p, dtype=float)
    if q.ndim != 2 or q.shape[1] < 2 or not np.all(np.isfinite(q)):
        raise ValueError("probabilities must be a finite 2D array")
    if np.any(q < 0):
        raise ValueError("probabilities must be non-negative")
    s = q.sum(axis=1, keepdims=True)
    if np.any(s <= 0):
        raise ValueError("probability rows must have positive mass")
    return q / s


def probability_entropy(p: np.ndarray) -> np.ndarray:
    q = normalize_probs(p)
    return -np.sum(q * np.log(np.clip(q, EPS, 1.0)), axis=1)


def disagreement_features(
    model_probs: Mapping[str, np.ndarray],
    *,
    rolling_window: int = 5,
) -> pd.DataFrame:
    names, n, k = _validate_probs(model_probs)
    stack = np.stack([normalize_probs(np.asarray(model_probs[name])) for name in names], axis=0)
    mean_p = stack.mean(axis=0)
    pred_cls = np.argmax(stack, axis=2)
    mean_cls = np.argmax(mean_p, axis=1)
    agreement = (pred_cls == mean_cls[None, :]).mean(axis=0)
    pairwise = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairwise.append(0.5 * np.abs(stack[i] - stack[j]).sum(axis=1))
    pairwise_disagreement = np.mean(pairwise, axis=0) if pairwise else np.zeros(n)
    rank_dis = np.zeros(n, dtype=float)
    if k >= 2 and len(names) >= 2:
        ranks = np.argsort(np.argsort(-stack, axis=2), axis=2)
        rank_dis = np.mean(np.abs(ranks[1:] - ranks[:1]), axis=(0, 2)) / max(k - 1, 1)
    stats = {
        "mean_probability": mean_p.max(axis=1),
        "std_probability": stack.std(axis=0).mean(axis=1),
        "min_probability": stack.min(axis=0).max(axis=1),
        "max_probability": stack.max(axis=0).max(axis=1),
        "probability_range": (stack.max(axis=0) - stack.min(axis=0)).max(axis=1),
        "prediction_entropy": probability_entropy(mean_p),
        "top_class_agreement_rate": agreement,
        "majority_margin": np.sort(mean_p, axis=1)[:, -1] - np.sort(mean_p, axis=1)[:, -2],
        "rank_disagreement": rank_dis,
        "pairwise_disagreement": pairwise_disagreement,
    }
    # Time-local dynamics, computed only from prior rows.
    d = pd.DataFrame(stats)
    rw = max(2, int(rolling_window))
    d["recent_disagreement"] = d["std_probability"].shift(1).rolling(rw, min_periods=1).mean().fillna(d["std_probability"].iloc[0])
    d["disagreement_change_rate"] = d["std_probability"].diff().fillna(0.0)
    d["rolling_disagreement"] = d["pairwise_disagreement"].shift(1).rolling(rw, min_periods=1).mean().fillna(d["pairwise_disagreement"].iloc[0])
    d["regime_conditioned_disagreement"] = d["pairwise_disagreement"] * (1.0 + d["probability_range"])
    d.index = range(n)
    return d


def _entropy_normalized(p: np.ndarray) -> np.ndarray:
    k = p.shape[1]
    return probability_entropy(p) / max(np.log(k), EPS)


@dataclass(frozen=True)
class DriftConfig:
    reference_blocks: int = 8
    rolling_blocks: int = 3
    watch_quantile: float = 0.80
    shift_quantile: float = 0.95
    severe_quantile: float = 0.99


class DriftDetector:
    def __init__(self, config: DriftConfig = DriftConfig()):
        self.config = config
        self.reference_: np.ndarray | None = None
        self.thresholds_: tuple[float, float, float] | None = None

    def fit(self, state_blocks: pd.DataFrame) -> "DriftDetector":
        x = np.asarray(state_blocks, dtype=float)
        if x.ndim != 2 or len(x) < max(4, self.config.reference_blocks + self.config.rolling_blocks):
            raise ValueError("not enough chronological blocks to fit drift detector")
        ref_n = min(self.config.reference_blocks, max(2, len(x) // 3))
        ref = x[:ref_n]
        mu = np.nanmean(ref, axis=0)
        sd = np.nanstd(ref, axis=0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        self.reference_ = np.column_stack([mu, sd])
        # Historical self-distance provides PIT-safe thresholds.
        scores = []
        for i in range(ref_n, len(x)):
            z = np.abs((x[i] - mu) / sd)
            scores.append(float(np.mean(np.clip(z, 0.0, 10.0))))
        scores = np.asarray(scores, dtype=float)
        if len(scores) == 0:
            scores = np.array([0.0])
        self.thresholds_ = (
            float(np.quantile(scores, self.config.watch_quantile)),
            float(np.quantile(scores, self.config.shift_quantile)),
            float(np.quantile(scores, self.config.severe_quantile)),
        )
        return self

    def score(self, state_blocks: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        if self.reference_ is None or self.thresholds_ is None:
            raise RuntimeError("drift detector is not fitted")
        x = np.asarray(state_blocks, dtype=float)
        mu, sd = self.reference_[:, 0], self.reference_[:, 1]
        z = np.abs((x - mu) / sd)
        scores = np.mean(np.clip(z, 0.0, 10.0), axis=1)
        w, s, sev = self.thresholds_
        levels = ["Normal" if v < w else "Watch" if v < s else "Shift" if v < sev else "Severe Shift" for v in scores]
        return scores, levels


@dataclass(frozen=True)
class RouterConfig:
    routing_temperature: float = 0.75
    failure_penalty: float = 2.0
    disagreement_penalty: float = 0.75
    low_predictability_flatten: float = 0.65
    drift_fallback_pull: float = 0.70
    max_daily_shift: float = 0.20
    confidence_abstain: float = 0.48
    risk_abstain: float = 0.70


class FutureFailurePredictor:
    """Predict future degradation risk from historical states only."""

    def __init__(self, *, threshold: float = 0.02, future_horizon_blocks: int = 2):
        self.threshold = float(threshold)
        self.future_horizon_blocks = int(future_horizon_blocks)
        self.models: dict[str, LogisticRegression | float] = {}
        self.feature_columns: list[str] = []
        self.base_rate: dict[str, float] = {}

    def fit(
        self,
        state_blocks: pd.DataFrame,
        block_logloss: Mapping[str, Sequence[float]],
        *,
        train_blocks: int | None = None,
    ) -> "FutureFailurePredictor":
        if self.future_horizon_blocks <= 0:
            raise ValueError("future_horizon_blocks must be positive")
        x = state_blocks.reset_index(drop=True).copy()
        if x.empty:
            raise ValueError("state_blocks is empty")
        self.feature_columns = list(x.columns)
        n = len(x) if train_blocks is None else int(train_blocks)
        if n <= self.future_horizon_blocks + 2:
            raise ValueError("not enough training blocks for future failure prediction")
        x = x.iloc[:n]
        for name, values in block_logloss.items():
            loss = np.asarray(values, dtype=float)
            if len(loss) != n:
                raise ValueError(f"logloss length mismatch for {name}")
            rows: list[int] = []
            labels: list[int] = []
            for t in range(n - self.future_horizon_blocks):
                past_start = max(0, t - self.future_horizon_blocks + 1)
                past = loss[past_start : t + 1]
                future = loss[t + 1 : t + 1 + self.future_horizon_blocks]
                if len(past) == 0 or len(future) < self.future_horizon_blocks:
                    continue
                base = float(np.mean(past))
                fwd = float(np.mean(future))
                rows.append(t)
                labels.append(int(fwd - base > self.threshold))
            if not rows:
                raise ValueError(f"no future failure labels for {name}")
            y = np.asarray(labels, dtype=int)
            xx = x.iloc[rows].to_numpy(dtype=float)
            rate = float(y.mean())
            self.base_rate[name] = rate
            if len(np.unique(y)) < 2:
                self.models[name] = rate
            else:
                model = LogisticRegression(C=0.5, max_iter=500, random_state=42)
                model.fit(xx, y)
                self.models[name] = model
        return self

    def predict_risk(self, state_blocks: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("failure predictor is not fitted")
        x = state_blocks[self.feature_columns].to_numpy(dtype=float)
        out: dict[str, np.ndarray] = {}
        for name, model in self.models.items():
            if isinstance(model, float):
                out[name] = np.full(len(x), model, dtype=float)
            else:
                out[name] = model.predict_proba(x)[:, 1]
        return pd.DataFrame(out, index=state_blocks.index)


def _blockify(values: np.ndarray, block_size: int) -> list[np.ndarray]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return [values[i : i + block_size] for i in range(0, len(values), block_size)]


def _state_blocks(
    model_probs: Mapping[str, np.ndarray],
    *,
    block_size: int,
    state_extra: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[slice]]:
    names, n, _ = _validate_probs(model_probs)
    row_features = disagreement_features(model_probs)
    blocks = _blockify(np.arange(n), block_size)
    states: list[dict[str, float]] = []
    slices: list[slice] = []
    for idxs in blocks:
        sl = slice(int(idxs[0]), int(idxs[-1]) + 1)
        row = row_features.iloc[idxs]
        item = {c: float(row[c].mean()) for c in row.columns}
        if state_extra is not None:
            extra = state_extra.iloc[sl]
            for c in extra.columns:
                vals = pd.to_numeric(extra[c], errors="coerce")
                item[c] = float(vals.mean()) if np.isfinite(vals).any() else 0.0
        states.append(item)
        slices.append(sl)
    return pd.DataFrame(states), slices


def _block_metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    q = normalize_probs(p)
    yy = np.asarray(y, dtype=int)
    ll = float(-np.mean(np.log(np.clip(q[np.arange(len(yy)), yy], EPS, 1.0))))
    brier = float(np.mean(np.sum((q - np.eye(q.shape[1])[yy]) ** 2, axis=1)))
    acc = float(np.mean(np.argmax(q, axis=1) == yy))
    return {"LogLoss": ll, "Brier": brier, "Accuracy": acc}


def _aggregate_block_metrics(
    y: np.ndarray,
    model_probs: Mapping[str, np.ndarray],
    block_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    _, n, _ = _validate_probs(model_probs, len(y))
    out = {"LogLoss": {}, "Brier": {}, "Accuracy": {}}
    for name, p in model_probs.items():
        vals = {"LogLoss": [], "Brier": [], "Accuracy": []}
        for idxs in _blockify(np.arange(n), block_size):
            m = _block_metric(y[idxs], np.asarray(p)[idxs])
            for key in vals:
                vals[key].append(m[key])
        for key in vals:
            out[key][name] = np.asarray(vals[key], dtype=float)
    return out


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    x = np.asarray(scores, dtype=float)
    x = x - np.max(x)
    e = np.exp(np.clip(x / temperature, -50, 50))
    return e / max(float(e.sum()), EPS)


class FutureGeneralizationController:
    """Fit on Development-OOS only; apply to a later unseen stream."""

    def __init__(self, config: RouterConfig = RouterConfig(), *, block_size: int = 90):
        self.config = config
        self.block_size = int(block_size)
        self.model_names: list[str] = []
        self.failure_predictor = FutureFailurePredictor()
        self.drift = DriftDetector()
        self.state_columns: list[str] = []
        self.base_weights_: np.ndarray | None = None
        self.fallback_weights_: np.ndarray | None = None
        self.audit: dict[str, Any] = {}

    def fit(self, y_dev: np.ndarray, model_probs_dev: Mapping[str, np.ndarray]) -> "FutureGeneralizationController":
        names, n, _ = _validate_probs(model_probs_dev, len(y_dev))
        if n < self.block_size * 6:
            raise ValueError("development stream too short for robust v2 controller")
        self.model_names = names
        states, _ = _state_blocks(model_probs_dev, block_size=self.block_size)
        self.state_columns = list(states.columns)
        metrics = _aggregate_block_metrics(np.asarray(y_dev, dtype=int), model_probs_dev, self.block_size)
        self.failure_predictor.fit(states, metrics["LogLoss"])
        self.drift.fit(states)
        # Quality is frozen from the latest development blocks only.
        tail = max(2, min(4, len(states) // 2))
        tail_losses = np.array([np.mean(metrics["LogLoss"][name][-tail:]) for name in names], dtype=float)
        self.base_weights_ = _softmax(-tail_losses, self.config.routing_temperature)
        self.fallback_weights_ = self.base_weights_.copy()
        self.audit = {
            "model_names": names,
            "development_rows": n,
            "development_blocks": len(states),
            "block_size": self.block_size,
            "future_horizon_blocks": self.failure_predictor.future_horizon_blocks,
            "failure_threshold_logloss": self.failure_predictor.threshold,
            "drift_thresholds": self.drift.thresholds_,
            "base_weights": self.base_weights_.tolist(),
        }
        return self

    def _routing_weights(
        self,
        state: pd.Series,
        probs_row: np.ndarray,
        failure_row: Mapping[str, float],
        drift_score: float,
        previous: np.ndarray | None,
        *,
        mode: str = "J",
    ) -> np.ndarray:
        if self.base_weights_ is None or self.fallback_weights_ is None:
            raise RuntimeError("controller is not fitted")
        p = normalize_probs(probs_row[None, :])[0]
        model_probs = probs_row
        consensus = model_probs.mean(axis=0)
        model_dist = 0.5 * np.abs(model_probs - consensus).sum(axis=1)
        risk = np.array([float(failure_row.get(name, 0.0)) for name in self.model_names], dtype=float)
        risk = np.clip(risk, 0.0, 1.0)
        pred_entropy = float(state.get("prediction_entropy", 0.0))
        pred_score = float(np.clip(1.0 - pred_entropy / max(np.log(len(p)), EPS), 0.0, 1.0))
        drift_state = float(np.clip(drift_score, 0.0, 10.0))

        use_dis = "B" in mode or mode in {"F", "G", "J"}
        use_pred = "C" in mode or mode in {"F", "H", "J"}
        use_fail = "D" in mode or mode in {"G", "H", "I", "J"}
        use_drift = "E" in mode or mode in {"I", "J"}

        score = np.log(np.clip(self.base_weights_, EPS, 1.0))
        if use_dis:
            score -= self.config.disagreement_penalty * model_dist
        if use_fail:
            score -= self.config.failure_penalty * risk
        w = _softmax(score, self.config.routing_temperature)

        if use_pred:
            alpha = np.clip(self.config.low_predictability_flatten * (1.0 - pred_score), 0.0, 1.0)
            w = (1.0 - alpha) * w + alpha * self.fallback_weights_
        if use_drift:
            pull = np.clip(self.config.drift_fallback_pull * (drift_state > 0.0), 0.0, 1.0)
            w = (1.0 - pull * min(drift_state / 5.0, 1.0)) * w + (pull * min(drift_state / 5.0, 1.0)) * self.fallback_weights_

        w = w / max(float(w.sum()), EPS)
        if previous is not None:
            delta = w - previous
            max_shift = float(self.config.max_daily_shift)
            if np.max(np.abs(delta)) > max_shift:
                w = previous + np.clip(delta, -max_shift, max_shift)
                w = w / max(float(w.sum()), EPS)
        return w

    def route(
        self,
        model_probs: Mapping[str, np.ndarray],
        *,
        mode: str = "J",
        state_extra: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        if list(model_probs) != self.model_names:
            raise ValueError("model names differ from fitted controller")
        names, n, k = _validate_probs(model_probs)
        states, slices = _state_blocks(model_probs, block_size=self.block_size, state_extra=state_extra)
        drift_scores, drift_levels = self.drift.score(states)
        risks = self.failure_predictor.predict_risk(states)
        stack = np.stack([normalize_probs(np.asarray(model_probs[name])) for name in names], axis=1)
        output = np.zeros((n, k), dtype=float)
        weights: list[list[float]] = []
        prev: np.ndarray | None = None
        predictability: list[float] = []
        confidence: list[float] = []
        for b, sl in enumerate(slices):
            fr = risks.iloc[b].to_dict()
            pmean = stack[sl].mean(axis=1)
            for i, row_i in enumerate(range(sl.start, sl.stop)):
                row_probs = stack[row_i]
                state = states.iloc[b]
                w = self._routing_weights(state, row_probs, fr, float(drift_scores[b]), prev, mode=mode)
                output[row_i] = np.sum(w[:, None] * row_probs, axis=0)
                prev = w
                weights.append(w.tolist())
                pe = float(state.get("prediction_entropy", probability_entropy(pmean[None, :])[0]))
                ps = float(np.clip(1.0 - pe / max(np.log(k), EPS), 0.0, 1.0))
                predictability.append(ps)
                confidence.append(float(np.max(output[row_i])))
        safety = SafetyMonitor().validate(output)
        return {
            "probabilities": output if safety["pass"] else np.sum(self.fallback_weights_[None, :, None] * stack, axis=1),
            "weights": np.asarray(weights, dtype=float),
            "predictability": np.asarray(predictability, dtype=float),
            "confidence": np.asarray(confidence, dtype=float),
            "drift_score": np.repeat(drift_scores, [s.stop - s.start for s in slices]),
            "drift_level": sum(([x] * (s.stop - s.start) for x, s in zip(drift_levels, slices)), []),
            "failure_risk": risks,
            "safety": safety,
            "abstain": (~safety["row_valid"]) | (np.asarray(confidence) < self.config.confidence_abstain),
            "mode": mode,
        }

    def predictability_score(self, routed: Mapping[str, Any]) -> np.ndarray:
        return np.asarray(routed["predictability"], dtype=float)


class SafetyMonitor:
    def validate(self, probabilities: np.ndarray) -> dict[str, Any]:
        p = np.asarray(probabilities, dtype=float)
        row_valid = (
            (p.ndim == 2)
            & np.all(np.isfinite(p), axis=1)
            & np.all(p >= 0, axis=1)
            & np.all(p <= 1, axis=1)
            & (np.sum(p, axis=1) > 0)
        )
        return {"pass": bool(np.all(row_valid)), "row_valid": row_valid}


def selective_metrics(y_true: np.ndarray, probabilities: np.ndarray, confidence: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = normalize_probs(probabilities)
    conf = np.asarray(confidence, dtype=float)
    if len(y) != len(p) or len(conf) != len(y):
        raise ValueError("selective metrics length mismatch")
    order = np.argsort(-conf)
    out: dict[str, Any] = {}
    for coverage in (1.00, 0.95, 0.90, 0.80, 0.70):
        take = max(1, int(np.floor(len(y) * coverage)))
        idx = order[:take]
        m = _block_metric(y[idx], p[idx])
        out[f"{int(coverage * 100)}%"] = {
            **m,
            "Coverage": float(len(idx) / len(y)),
            "HighConfidenceAccuracy": m["Accuracy"],
            "rows": int(len(idx)),
        }
    return out


def paired_block_bootstrap(
    y_true: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    block_size: int = 20,
    replications: int = 400,
    seed: int = 42,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    b = normalize_probs(baseline)
    c = normalize_probs(candidate)
    if len(y) != len(b) or b.shape != c.shape:
        raise ValueError("paired arrays mismatch")
    rng = np.random.default_rng(seed)
    blocks = _blockify(np.arange(len(y)), block_size)
    if len(blocks) < 2:
        raise ValueError("need at least two blocks")
    diffs = {"Accuracy": [], "LogLoss": [], "Brier": []}
    def vals(idx: np.ndarray, p: np.ndarray) -> dict[str, float]:
        return _block_metric(y[idx], p[idx])
    for _ in range(int(replications)):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[i] for i in chosen])
        bm, cm = vals(idx, b), vals(idx, c)
        diffs["Accuracy"].append(cm["Accuracy"] - bm["Accuracy"])
        diffs["LogLoss"].append(cm["LogLoss"] - bm["LogLoss"])
        diffs["Brier"].append(cm["Brier"] - bm["Brier"])
    out = {}
    for key, arr in diffs.items():
        a = np.asarray(arr, dtype=float)
        out[key] = {
            "mean_delta": float(np.mean(a)),
            "ci95": [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))],
            "improvement_positive_probability": float(np.mean(a > 0 if key == "Accuracy" else a < 0)),
        }
    return out


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return _block_metric(np.asarray(y_true, dtype=int), probabilities)


def as_dict(controller: FutureGeneralizationController) -> dict[str, Any]:
    return {
        "config": asdict(controller.config),
        "audit": controller.audit,
    }
