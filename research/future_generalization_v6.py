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
