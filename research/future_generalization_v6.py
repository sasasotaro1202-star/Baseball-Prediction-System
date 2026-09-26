"""Production-isolated v6 extensions built only from PIT-safe OOS streams.

The functions here never require future labels at prediction time. Training labels
are accepted only for Development-OOS fitting and historical retrieval.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


EPS=1e-12


def prediction_dynamics(probabilities: np.ndarray, rolling_window: int = 5) -> pd.DataFrame:
    p=np.asarray(probabilities,dtype=float)
    if p.ndim!=2 or not np.all(np.isfinite(p)) or np.any(p<0):
        raise ValueError("probabilities must be finite non-negative 2D")
    p=p/p.sum(axis=1,keepdims=True)
    cls=np.argmax(p,axis=1)
    conf=np.max(p,axis=1)
    vel=np.vstack([np.zeros((1,p.shape[1])),np.diff(p,axis=0)])
    accel=np.vstack([np.zeros((2,p.shape[1])),np.diff(p,n=2,axis=0)])
    flips=np.concatenate([[0],(cls[1:]!=cls[:-1]).astype(int)])
    rw=max(2,int(rolling_window))
    return pd.DataFrame({
        "prediction_confidence":conf,
        "prediction_velocity":np.linalg.norm(vel,axis=1),
        "prediction_acceleration":np.linalg.norm(accel,axis=1),
        "prediction_flip":flips,
        "flip_rate":pd.Series(flips).rolling(rw,min_periods=1).mean().to_numpy(),
        "confidence_shock":np.abs(pd.Series(conf).diff().fillna(0.0).to_numpy()),
    })


def error_correlation(y_true: np.ndarray, model_probs: Mapping[str,np.ndarray]) -> pd.DataFrame:
    y=np.asarray(y_true,dtype=int)
    errors={}
    for name,p in model_probs.items():
        q=np.asarray(p,dtype=float)
        if len(q)!=len(y):
            raise ValueError("error-correlation length mismatch")
        q=q/q.sum(axis=1,keepdims=True)
        errors[name]=(np.argmax(q,axis=1)!=y).astype(float)
    return pd.DataFrame(errors).corr().fillna(0.0)


def error_overlap(y_true: np.ndarray, model_probs: Mapping[str,np.ndarray]) -> pd.DataFrame:
    y=np.asarray(y_true,dtype=int)
    names=list(model_probs)
    e=np.column_stack([
        (np.argmax(np.asarray(model_probs[n]),axis=1)!=y).astype(int)
        for n in names
    ])
    out=np.zeros((len(names),len(names)),dtype=float)
    for i in range(len(names)):
        for j in range(len(names)):
            den=np.sum((e[:,i]>0)|(e[:,j]>0))
            out[i,j]=float(np.sum((e[:,i]>0)&(e[:,j]>0))/den) if den else 0.0
    return pd.DataFrame(out,index=names,columns=names)


def feature_reliability(X: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for col in X.columns:
        s=pd.to_numeric(X[col],errors="coerce")
        missing=float(s.isna().mean())
        finite=s.replace([np.inf,-np.inf],np.nan)
        if finite.notna().sum()>=2:
            q1,q3=finite.quantile([0.25,0.75]).to_numpy()
            iqr=float(q3-q1)
            robust=float(iqr/(abs(float(finite.median()))+1e-6))
        else:
            robust=0.0
        rows.append({
            "feature":col,
            "completeness":1.0-missing,
            "finite_rate":float(np.isfinite(s.to_numpy(dtype=float,nan=np.nan)).mean()),
            "robust_scale":robust,
            "reliability":float(np.clip((1.0-missing)*0.7+min(robust,1.0)*0.3,0.0,1.0))
        })
    return pd.DataFrame(rows)


def uncertainty_decomposition(model_probs: Mapping[str,np.ndarray]) -> pd.DataFrame:
    names=list(model_probs)
    stack=np.stack([np.asarray(model_probs[n],dtype=float) for n in names],axis=1)
    stack=stack/stack.sum(axis=2,keepdims=True)
    mean_p=stack.mean(axis=1)
    pred_entropy=-np.sum(mean_p*np.log(np.clip(mean_p,EPS,1.0)),axis=1)
    model_entropy=np.mean(-np.sum(stack*np.log(np.clip(stack,EPS,1.0)),axis=2),axis=1)
    disagreement=np.mean(0.5*np.abs(stack-mean_p[:,None,:]).sum(axis=2),axis=1)
    epistemic=np.clip(disagreement,0.0,1.0)
    aleatoric=np.clip(model_entropy/np.log(stack.shape[2]),0.0,1.0)
    total=np.clip(pred_entropy/np.log(stack.shape[2]),0.0,1.0)
    return pd.DataFrame({
        "uncertainty_total":total,
        "uncertainty_epistemic":epistemic,
        "uncertainty_aleatoric":aleatoric,
        "uncertainty_residual":np.clip(total-epistemic,0.0,1.0),
    })


@dataclass(frozen=True)
class RetrievalConfig:
    k: int=25
    temperature: float=0.20
    min_history: int=80


class HistoricalPrototypeRetriever:
    """Nearest-neighbour class distribution from Development-OOS history only."""

    def __init__(self, config: RetrievalConfig=RetrievalConfig()):
        self.config=config
        self.X_: np.ndarray|None=None
        self.y_: np.ndarray|None=None
        self.mu_: np.ndarray|None=None
        self.sd_: np.ndarray|None=None

    def fit(self, state: pd.DataFrame, y: np.ndarray) -> "HistoricalPrototypeRetriever":
        x=state.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        yy=np.asarray(y,dtype=int)
        if len(x)!=len(yy) or len(x)<self.config.min_history:
            raise ValueError("insufficient historical retrieval data")
        if not np.all(np.isfinite(x)):
            raise ValueError("retrieval state contains non-finite values")
        self.mu_=x.mean(axis=0)
        self.sd_=np.where(x.std(axis=0)>1e-9,x.std(axis=0),1.0)
        self.X_=(x-self.mu_)/self.sd_
        self.y_=yy
        return self

    def predict_proba(self, state: pd.DataFrame, n_classes: int) -> np.ndarray:
        if self.X_ is None or self.y_ is None or self.mu_ is None or self.sd_ is None:
            raise RuntimeError("retriever is not fitted")
        x=state.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError("query state contains non-finite values")
        x=(x-self.mu_)/self.sd_
        out=np.zeros((len(x),n_classes),dtype=float)
        k=min(max(1,int(self.config.k)),len(self.X_))
        for i,row in enumerate(x):
            dist=np.sqrt(np.sum((self.X_-row[None,:])**2,axis=1))
            idx=np.argpartition(dist,k-1)[:k]
            weights=np.exp(-dist[idx]/max(self.config.temperature,1e-6))
            weights/=max(float(weights.sum()),EPS)
            out[i]=np.bincount(self.y_[idx],weights=weights,minlength=n_classes)
        out/=out.sum(axis=1,keepdims=True)
        return out


def time_to_failure(risk: np.ndarray, warning: float=0.5, failure: float=0.7) -> np.ndarray:
    r=np.asarray(risk,dtype=float)
    if not np.all(np.isfinite(r)):
        raise ValueError("risk must be finite")
    # Discrete hazard proxy: 1 / expected periods to cross failure probability.
    out=np.full(r.shape,np.inf,dtype=float)
    out[r>=warning]=1.0/np.clip(r[r>=warning],1e-6,1.0)
    out[r>=failure]=np.minimum(out[r>=failure],1.0)
    return out



def source_reliability(quality: pd.DataFrame) -> pd.DataFrame:
    """Outcome-free source reliability summary from operational quality fields."""
    required = {"freshness", "completeness", "consistency"}
    missing = required.difference(quality.columns)
    if missing:
        raise ValueError("missing source quality columns: " + ",".join(sorted(missing)))
    out = quality.copy()
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["source_reliability"] = out[["freshness", "completeness", "consistency"]].mean(axis=1).clip(0.0, 1.0)
    return out


def information_shock_score(event_counts: Sequence[float], window: int = 10) -> np.ndarray:
    """Return a 0..1 PIT-safe shock score based on the prior rolling baseline."""
    x = pd.Series(np.asarray(event_counts, dtype=float))
    if len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("event_counts must be finite and non-empty")
    w = max(3, int(window))
    med = x.shift(1).rolling(w, min_periods=2).median()
    mad = (x.shift(1) - med).abs().rolling(w, min_periods=2).median()
    scale = (1.4826 * mad).replace(0.0, np.nan).fillna(1.0).clip(lower=1e-6)
    z = ((x - med.fillna(x.iloc[0])) / scale).abs().to_numpy()
    return np.clip(z / 4.0, 0.0, 1.0)


def regime_transition_probabilities(regimes: Sequence[str], *, smoothing: float = 1.0) -> pd.DataFrame:
    """Chronological Markov transition estimate; no target labels required."""
    s = [str(v) for v in regimes]
    if len(s) < 4:
        raise ValueError("not enough regimes")
    if smoothing <= 0 or not np.isfinite(smoothing):
        raise ValueError("smoothing must be positive and finite")
    states = sorted(set(s))
    idx = {v: i for i, v in enumerate(states)}
    mat = np.full((len(states), len(states)), float(smoothing))
    for a, b in zip(s[:-1], s[1:]):
        mat[idx[a], idx[b]] += 1.0
    mat /= mat.sum(axis=1, keepdims=True)
    return pd.DataFrame(mat, index=states, columns=states)


def next_regime_distribution(transition: pd.DataFrame, current: str) -> dict[str, float]:
    if current not in transition.index:
        p = np.full(len(transition.columns), 1.0 / len(transition.columns))
    else:
        p = transition.loc[current].to_numpy(dtype=float)
    return {str(k): float(v) for k, v in zip(transition.columns, p)}


def meta_label_features(model_probs: np.ndarray, state: pd.DataFrame | None = None) -> pd.DataFrame:
    """Construct individual-prediction meta features without outcomes."""
    p = np.asarray(model_probs, dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    out = pd.DataFrame({
        "base_confidence": p.max(axis=1),
        "base_entropy": -np.sum(p * np.log(np.clip(p, EPS, 1.0)), axis=1),
        "base_margin": np.sort(p, axis=1)[:, -1] - np.sort(p, axis=1)[:, -2],
    })
    if state is not None:
        s = state.reset_index(drop=True).select_dtypes(include=[np.number]).astype(float)
        if len(s) != len(out) or not np.all(np.isfinite(s.to_numpy())):
            raise ValueError("meta state mismatch or non-finite values")
        out = pd.concat([s, out], axis=1)
    return out


class ChronologicalMetaLabeler:
    """Fit prediction-level correctness risk on an earlier OOS prefix only."""

    def __init__(self):
        self.model: LogisticRegression | None = None
        self.columns: list[str] = []
        self.fallback: float = 0.5

    def fit(self, state: pd.DataFrame, y_true: np.ndarray, base_probs: np.ndarray) -> "ChronologicalMetaLabeler":
        y = np.asarray(y_true, dtype=int)
        x = meta_label_features(base_probs, state)
        if len(x) != len(y):
            raise ValueError("meta-label lengths differ")
        target = (np.argmax(_norm(base_probs), axis=1) == y).astype(int)
        self.columns = list(x.columns)
        self.fallback = float(target.mean()) if len(target) else 0.5
        if len(np.unique(target)) >= 2:
            self.model = LogisticRegression(C=0.5, max_iter=500, random_state=42)
            self.model.fit(x.to_numpy(), target)
        return self

    def predict(self, state: pd.DataFrame, base_probs: np.ndarray) -> np.ndarray:
        x = meta_label_features(base_probs, state)
        x = x[self.columns]
        if self.model is None:
            return np.full(len(x), self.fallback, dtype=float)
        return self.model.predict_proba(x.to_numpy())[:, 1]


def split_conformal_sets(
    calibration_probs: np.ndarray,
    calibration_y: np.ndarray,
    query_probs: np.ndarray,
    *,
    alpha: float = 0.10,
) -> dict[str, object]:
    """Chronology-compatible split-conformal candidate; no time-series guarantee claimed."""
    cp = _norm(calibration_probs)
    y = np.asarray(calibration_y, dtype=int)
    qp = _norm(query_probs)
    if len(cp) != len(y) or cp.shape[1] != qp.shape[1] or len(y) < 20:
        raise ValueError("invalid conformal inputs")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    scores = 1.0 - cp[np.arange(len(y)), y]
    q = float(np.quantile(scores, min(1.0, np.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores)), method="higher"))
    sets = qp >= (1.0 - q - EPS)
    sets[np.arange(len(sets)), np.argmax(qp, axis=1)] = True
    return {
        "sets": sets,
        "set_size_mean": float(sets.sum(axis=1).mean()),
        "threshold": q,
        "alpha": float(alpha),
    }


def multi_horizon_consistency(probability_streams: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Compare horizons on the same chronological prediction rows, label-free."""
    if len(probability_streams) < 2:
        raise ValueError("at least two horizons required")
    names = list(probability_streams)
    ps = [_norm(np.asarray(probability_streams[n])) for n in names]
    if any(p.shape != ps[0].shape for p in ps[1:]):
        raise ValueError("horizon shapes differ")
    mean_p = np.mean(np.stack(ps, axis=0), axis=0)
    mean_dist = np.mean([0.5 * np.abs(p - mean_p).sum(axis=1) for p in ps], axis=0)
    top_agreement = np.mean([np.argmax(p, axis=1) == np.argmax(mean_p, axis=1) for p in ps], axis=0)
    return pd.DataFrame({
        "temporal_consistency": 1.0 - np.clip(mean_dist, 0.0, 1.0),
        "horizon_top_class_agreement": top_agreement,
    })


def stress_probability_stream(
    probabilities: np.ndarray,
    *,
    temperature: float = 1.5,
    additive_noise: float = 0.02,
    seed: int = 42,
) -> dict[str, float]:
    """Distribution-level robustness probe without changing production outputs."""
    p = _norm(probabilities)
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    rng = np.random.default_rng(seed)
    logits = np.log(np.clip(p, EPS, 1.0)) / temperature
    noisy = np.clip(logits + rng.normal(0.0, additive_noise, size=logits.shape), -50, 50)
    noisy = np.exp(noisy - noisy.max(axis=1, keepdims=True))
    noisy /= noisy.sum(axis=1, keepdims=True)
    return {
        "mean_l1_change": float(np.mean(0.5 * np.abs(noisy - p).sum(axis=1))),
        "p95_l1_change": float(np.quantile(0.5 * np.abs(noisy - p).sum(axis=1), 0.95)),
        "unstable_rate": float(np.mean(0.5 * np.abs(noisy - p).sum(axis=1) > 0.10)),
    }


def probability_safety_gate(
    probabilities: np.ndarray,
    *,
    max_jump: float = 0.35,
    min_probability: float = 1e-5,
    max_probability: float = 1.0 - 1e-5,
) -> dict[str, object]:
    p = _norm(probabilities)
    valid = np.all(np.isfinite(p), axis=1)
    valid &= np.min(p, axis=1) >= float(min_probability)
    valid &= np.max(p, axis=1) <= float(max_probability)
    if len(p) > 1:
        valid[1:] &= np.max(np.abs(p[1:] - p[:-1]), axis=1) <= float(max_jump)
    return {
        "pass": bool(np.all(valid)),
        "row_valid": valid,
        "invalid_rate": float(np.mean(~valid)),
    }
