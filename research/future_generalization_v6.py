"""Maximum Future-Generalization v6 research extensions.

Research-only utilities. They intentionally separate:
- PIT-safe prediction-time state transforms (no labels required)
- historical outcome analysis / meta-label fitting (labels required, chronological only)

No function here mutates Production artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


EPS = 1e-12


def _norm(p: np.ndarray) -> np.ndarray:
    x = np.asarray(p, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("probabilities must be a finite non-negative 2D array")
    s = x.sum(axis=1, keepdims=True)
    if np.any(s <= 0):
        raise ValueError("probability rows must have positive mass")
    return x / s


def error_correlation(
    y_true: np.ndarray,
    model_probs: Mapping[str, np.ndarray],
    *,
    classes: int | None = None,
) -> dict[str, object]:
    """Measure whether models make correlated errors on historical OOS only."""
    y = np.asarray(y_true, dtype=int)
    names = list(model_probs)
    if len(names) < 2:
        raise ValueError("at least two models are required")
    ps = [_norm(np.asarray(model_probs[n])) for n in names]
    if any(len(p) != len(y) for p in ps):
        raise ValueError("y_true and probability arrays have different lengths")
    if classes is not None and any(p.shape[1] != classes for p in ps):
        raise ValueError("class dimension mismatch")
    errs = np.column_stack([(np.argmax(p, axis=1) != y).astype(float) for p in ps])
    corr = np.corrcoef(errs, rowvar=False)
    if not np.all(np.isfinite(corr)):
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
    overlap = {}
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            overlap[f"{a}::{b}"] = float(np.mean((errs[:, i] == 1) & (errs[:, j] == 1)))
    return {
        "models": names,
        "pairwise_error_correlation": {
            f"{names[i]}::{names[j]}": float(corr[i, j])
            for i in range(len(names)) for j in range(i + 1, len(names))
        },
        "error_overlap": overlap,
        "mean_error_rate": {n: float(errs[:, i].mean()) for i, n in enumerate(names)},
    }


def prediction_dynamics(model_probs: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """PIT-safe probability velocity/acceleration/flip features."""
    names = list(model_probs)
    if not names:
        raise ValueError("model_probs is empty")
    stack = np.stack([_norm(np.asarray(model_probs[n])) for n in names], axis=1)
    mean_p = stack.mean(axis=1)
    top = np.argmax(mean_p, axis=1)
    l1_velocity = np.zeros(len(mean_p), dtype=float)
    l1_acceleration = np.zeros(len(mean_p), dtype=float)
    if len(mean_p) > 1:
        l1_velocity[1:] = np.abs(mean_p[1:] - mean_p[:-1]).sum(axis=1)
        l1_acceleration[2:] = l1_velocity[2:] - l1_velocity[1:-1]
    flip = np.zeros(len(mean_p), dtype=float)
    if len(top) > 1:
        flip[1:] = (top[1:] != top[:-1]).astype(float)
    persistence = pd.Series((flip == 0).astype(float)).rolling(5, min_periods=1).mean().to_numpy()
    reversal = np.zeros(len(mean_p), dtype=float)
    if len(mean_p) > 2:
        reversal[2:] = (
            ((mean_p[2:] - mean_p[1:-1]) * (mean_p[1:-1] - mean_p[:-2])).sum(axis=1) < 0
        ).astype(float)
    return pd.DataFrame({
        "prediction_velocity": l1_velocity,
        "prediction_acceleration": l1_acceleration,
        "prediction_flip": flip,
        "prediction_persistence": persistence,
        "prediction_reversal": reversal,
    })


def information_shock(event_counts: Sequence[float], *, window: int = 10) -> pd.DataFrame:
    """Unlabeled update/event shock detector using only the historical prefix."""
    x = pd.Series(np.asarray(event_counts, dtype=float))
    if len(x) == 0 or (not np.all(np.isfinite(x))):
        raise ValueError("event_counts must be finite and non-empty")
    w = max(3, int(window))
    median = x.shift(1).rolling(w, min_periods=2).median()
    mad = (x.shift(1) - median).abs().rolling(w, min_periods=2).median()
    scale = (1.4826 * mad).fillna(1.0).clip(lower=1e-6)
    z = ((x - median.fillna(x.iloc[0])) / scale).abs().fillna(0.0)
    return pd.DataFrame({
        "event_rate": x,
        "event_rate_z": z,
        "shock_score": np.clip(z / 4.0, 0.0, 1.0),
        "information_shock": (z >= 4.0).astype(int),
    })


def feature_reliability(
    quality: pd.DataFrame,
    *,
    weights: Mapping[str, float] | None = None,
) -> pd.Series:
    """Combine feature reliability dimensions without outcomes."""
    required = ["reliability", "freshness", "completeness", "consistency", "drift"]
    q = quality.copy()
    missing = [c for c in required if c not in q.columns]
    if missing:
        raise ValueError("missing reliability columns: " + ",".join(missing))
    w = dict(weights or {"reliability": .30, "freshness": .15, "completeness": .20, "consistency": .20, "drift": .15})
    for c in required:
        if c not in w or w[c] < 0 or not np.isfinite(w[c]):
            raise ValueError("invalid reliability weights")
    total = sum(w.values())
    if total <= 0:
        raise ValueError("reliability weights must have positive total")
    score = (
        w["reliability"] * q["reliability"].astype(float)
        + w["freshness"] * q["freshness"].astype(float)
        + w["completeness"] * q["completeness"].astype(float)
        + w["consistency"] * q["consistency"].astype(float)
        + w["drift"] * (1.0 - q["drift"].astype(float))
    ) / total
    return score.clip(0.0, 1.0)


@dataclass(frozen=True)
class RegimeTransitionModel:
    states: tuple[str, ...]
    transition: np.ndarray

    @classmethod
    def fit(cls, historical_states: Sequence[str]) -> "RegimeTransitionModel":
        s = [str(x) for x in historical_states]
        if len(s) < 4:
            raise ValueError("not enough historical regimes")
        states = tuple(sorted(set(s)))
        idx = {v: i for i, v in enumerate(states)}
        mat = np.ones((len(states), len(states)), dtype=float)  # Laplace smoothing
        for a, b in zip(s[:-1], s[1:]):
            mat[idx[a], idx[b]] += 1.0
        mat /= mat.sum(axis=1, keepdims=True)
        return cls(states, mat)

    def predict_next(self, current_state: str) -> dict[str, float]:
        if current_state not in self.states:
            return {s: 1.0 / len(self.states) for s in self.states}
        row = self.transition[self.states.index(current_state)]
        return {s: float(p) for s, p in zip(self.states, row)}


class HistoricalRetrieval:
    """Nearest historical prototype/error retrieval.

    fit() must be called with rows strictly preceding any query rows.
    """

    def __init__(self, n_neighbors: int = 25):
        self.n_neighbors = int(n_neighbors)
        self.nn: NearestNeighbors | None = None
        self.X_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    def fit(self, X_history: pd.DataFrame, y_history: np.ndarray) -> "HistoricalRetrieval":
        x = np.asarray(X_history, dtype=float)
        y = np.asarray(y_history, dtype=int)
        if x.ndim != 2 or len(x) != len(y) or len(x) < 3:
            raise ValueError("invalid historical retrieval corpus")
        if not np.all(np.isfinite(x)):
            raise ValueError("historical retrieval features must be finite")
        n = min(self.n_neighbors, len(x))
        self.nn = NearestNeighbors(n_neighbors=n, metric="euclidean")
        self.nn.fit(x)
        self.X_, self.y_ = x, y
        return self

    def predict(self, X_query: pd.DataFrame) -> dict[str, object]:
        if self.nn is None or self.X_ is None or self.y_ is None:
            raise RuntimeError("retrieval is not fitted")
        xq = np.asarray(X_query, dtype=float)
        if xq.ndim != 2 or not np.all(np.isfinite(xq)):
            raise ValueError("query features must be finite")
        dist, ind = self.nn.kneighbors(xq)
        classes = int(max(self.y_)) + 1
        probs = np.zeros((len(xq), classes), dtype=float)
        for i, row in enumerate(ind):
            w = 1.0 / np.clip(dist[i], 1e-6, None)
            for wi, j in zip(w, row):
                probs[i, self.y_[j]] += wi
            probs[i] /= probs[i].sum()
        return {
            "probabilities": probs,
            "nearest_distance": dist[:, 0],
            "nearest_similarity": 1.0 / (1.0 + dist[:, 0]),
        }


class MetaLabeler:
    """Individual-prediction reliability model fit on chronological OOS only."""

    def __init__(self):
        self.model: LogisticRegression | None = None
        self.feature_columns: list[str] = []

    def fit(self, X_oos: pd.DataFrame, y_true: np.ndarray, base_probs: np.ndarray) -> "MetaLabeler":
        p = _norm(base_probs)
        y = np.asarray(y_true, dtype=int)
        if len(X_oos) != len(y) or len(p) != len(y):
            raise ValueError("meta-label input lengths differ")
        pred = np.argmax(p, axis=1)
        meta_y = (pred == y).astype(int)
        x = pd.DataFrame(X_oos).reset_index(drop=True).astype(float)
        x["base_confidence"] = p.max(axis=1)
        x["base_entropy"] = -np.sum(p * np.log(np.clip(p, EPS, 1.0)), axis=1)
        self.feature_columns = list(x.columns)
        if len(np.unique(meta_y)) < 2:
            self.model = None
        else:
            self.model = LogisticRegression(C=0.5, max_iter=500, random_state=42)
            self.model.fit(x.to_numpy(), meta_y)
        self._fallback = float(meta_y.mean())
        return self

    def predict_reliability(self, X: pd.DataFrame, base_probs: np.ndarray) -> np.ndarray:
        p = _norm(base_probs)
        x = pd.DataFrame(X).reset_index(drop=True).astype(float)
        x["base_confidence"] = p.max(axis=1)
        x["base_entropy"] = -np.sum(p * np.log(np.clip(p, EPS, 1.0)), axis=1)
        x = x[self.feature_columns]
        if self.model is None:
            return np.full(len(x), self._fallback, dtype=float)
        return self.model.predict_proba(x.to_numpy())[:, 1]


def uncertainty_decomposition(
    model_probs: Mapping[str, np.ndarray],
    *,
    data_quality: np.ndarray | None = None,
    drift_score: np.ndarray | None = None,
    information_uncertainty: np.ndarray | None = None,
) -> pd.DataFrame:
    names = list(model_probs)
    stack = np.stack([_norm(np.asarray(model_probs[n])) for n in names], axis=1)
    mean_p = stack.mean(axis=1)
    model_unc = np.var(stack, axis=1).mean(axis=1)
    entropy_unc = -np.sum(mean_p * np.log(np.clip(mean_p, EPS, 1.0)), axis=1)
    out = pd.DataFrame({
        "model_uncertainty": model_unc,
        "entropy_uncertainty": entropy_unc,
        "disagreement_uncertainty": np.mean(np.abs(stack - mean_p[:, None, :]), axis=(1, 2)),
    })
    if data_quality is not None:
        q = np.asarray(data_quality, dtype=float)
        if len(q) != len(out):
            raise ValueError("data_quality length mismatch")
        out["data_uncertainty"] = 1.0 - np.clip(q, 0.0, 1.0)
    else:
        out["data_uncertainty"] = 0.0
    if drift_score is not None:
        d = np.asarray(drift_score, dtype=float)
        if len(d) != len(out):
            raise ValueError("drift_score length mismatch")
        out["distribution_shift_uncertainty"] = np.clip(d, 0.0, 1.0)
    else:
        out["distribution_shift_uncertainty"] = 0.0
    if information_uncertainty is not None:
        u = np.asarray(information_uncertainty, dtype=float)
        if len(u) != len(out):
            raise ValueError("information_uncertainty length mismatch")
        out["information_uncertainty"] = np.clip(u, 0.0, 1.0)
    else:
        out["information_uncertainty"] = 0.0
    out["total_observable_uncertainty"] = out.drop(columns=[]).sum(axis=1)
    return out


def counterfactual_stability(
    X: pd.DataFrame,
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    *,
    noise_scale: float = 0.01,
    repeats: int = 3,
    seed: int = 42,
) -> dict[str, object]:
    """Label-free local sensitivity probe."""
    x = X.copy().reset_index(drop=True)
    base = _norm(np.asarray(predict_fn(x), dtype=float))
    rng = np.random.default_rng(seed)
    changes = []
    for _ in range(max(1, int(repeats))):
        pert = x.astype(float).copy()
        for c in pert.columns:
            scale = float(np.nanstd(pert[c].to_numpy(dtype=float)))
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            pert[c] = pert[c] + rng.normal(0.0, noise_scale * scale, size=len(pert))
        alt = _norm(np.asarray(predict_fn(pert), dtype=float))
        changes.append(np.abs(alt - base).sum(axis=1) * 0.5)
    c = np.vstack(changes)
    return {
        "mean_prediction_change": c.mean(axis=0),
        "p95_prediction_change": np.quantile(c, 0.95, axis=0),
        "unstable_rate": float(np.mean(c.mean(axis=0) > 0.10)),
    }


def split_conformal(
    calibration_probs: np.ndarray,
    calibration_y: np.ndarray,
    query_probs: np.ndarray,
    *,
    alpha: float = 0.10,
) -> dict[str, object]:
    """Basic split conformal prediction-set candidate.

    Time-series guarantee is NOT claimed; caller must preserve chronology.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    cp = _norm(calibration_probs)
    y = np.asarray(calibration_y, dtype=int)
    qp = _norm(query_probs)
    if len(cp) != len(y) or cp.shape[1] != qp.shape[1]:
        raise ValueError("conformal shape mismatch")
    scores = 1.0 - cp[np.arange(len(y)), y]
    q_level = min(1.0, (np.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores)))
    threshold = float(np.quantile(scores, q_level, method="higher"))
    sets = qp >= (1.0 - threshold - 1e-12)
    # Always include argmax to avoid an empty prediction set.
    sets[np.arange(len(sets)), np.argmax(qp, axis=1)] = True
    return {
        "prediction_sets": sets,
        "set_size": sets.sum(axis=1),
        "threshold": threshold,
        "alpha": float(alpha),
    }


def max_probability_safety(
    probabilities: np.ndarray,
    *,
    min_prob: float = 1e-6,
    max_prob: float = 1.0 - 1e-6,
    max_jump: float = 0.35,
) -> dict[str, object]:
    p = _norm(probabilities)
    valid = np.all(np.isfinite(p), axis=1)
    if len(p) > 1:
        valid[1:] &= np.max(np.abs(p[1:] - p[:-1]), axis=1) <= max_jump
    valid &= np.max(p, axis=1) < max_prob
    valid &= np.min(p, axis=1) > min_prob
    return {
        "pass": bool(np.all(valid)),
        "row_valid": valid,
        "invalid_rate": float(np.mean(~valid)),
    }
