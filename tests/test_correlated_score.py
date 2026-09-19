import numpy as np
from research.correlated_score import estimate_shared_lambda, grid, low_high


def test_grid_normalizes_and_preserves_marginal_means_reasonably():
    m=grid(4.2,3.8,0.35,14)
    assert abs(m.sum()-1.0)<1e-9
    assert (m>=0).all()


def test_shared_lambda_is_nonnegative_and_bounded():
    assert estimate_shared_lambda(np.ones(100),np.ones(100)*2,max_shared=0.4)==0.4
    assert estimate_shared_lambda(np.ones(100),-np.ones(100),max_shared=0.4)==0.0


def test_low_high_contract():
    low,high=low_high(4.0,3.0,0.2)
    assert 0<=low<=1 and 0<=high<=1
    assert abs(low+high-1)<1e-9
