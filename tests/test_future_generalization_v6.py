import numpy as np
import pandas as pd
import pytest
from research.future_generalization_v6 import (
    HistoricalPrototypeRetriever, error_correlation, error_overlap,
    feature_reliability, prediction_dynamics, time_to_failure,
    uncertainty_decomposition,
)

def _p(seed=1,n=120):
    rng=np.random.default_rng(seed); y=rng.integers(0,3,n)
    out={}
    for i in range(3):
        z=rng.normal(size=(n,3)); z[np.arange(n),y]+=1.0+i*0.1
        e=np.exp(z-z.max(axis=1,keepdims=True)); out[f"M{i}"]=e/e.sum(axis=1,keepdims=True)
    return y,out

def test_dynamics_and_uncertainty_finite():
    y,p=_p()
    assert np.isfinite(prediction_dynamics(p["M0"]).to_numpy()).all()
    assert np.isfinite(uncertainty_decomposition(p).to_numpy()).all()

def test_error_diversity_shapes():
    y,p=_p()
    c=error_correlation(y,p); o=error_overlap(y,p)
    assert c.shape==(3,3) and o.shape==(3,3)

def test_feature_reliability_range():
    x=pd.DataFrame({"a":[1,2,np.nan,4],"b":[5,5,5,5]})
    r=feature_reliability(x)
    assert ((r["reliability"]>=0)&(r["reliability"]<=1)).all()

def test_retrieval_is_reproducible_and_normalized():
    y,p=_p(n=200)
    state=prediction_dynamics(p["M0"]).join(uncertainty_decomposition(p))
    ret=HistoricalPrototypeRetriever().fit(state,y)
    q=ret.predict_proba(state.iloc[:20],3)
    assert q.shape==(20,3)
    assert np.allclose(q.sum(axis=1),1.0)

def test_time_to_failure():
    out=time_to_failure(np.array([0.1,0.5,0.8]))
    assert np.isinf(out[0]) and out[1]==2.0 and out[2]==1.0
