"""Focused tests for the leakage-safe regime router."""

import numpy as np
import pandas as pd

from research.regime_router import RegimeRouter


def test_router_uses_training_only_thresholds():
    train = pd.DataFrame({
        "d_elo": np.arange(120, dtype=float),
        "expected_env": np.linspace(3.0, 8.0, 120),
    })
    router = RegimeRouter(min_regime_rows=10).fit(train)
    labels_before = router.labels(train.iloc[-20:])

    future = pd.DataFrame({
        "d_elo": np.arange(10000, 10120, dtype=float),
        "expected_env": np.linspace(20.0, 30.0, 120),
    })
    # Changing future rows must not mutate already-fitted thresholds.
    thresholds = (router.strength_q, router.env_q)
    _ = router.labels(future)
    assert thresholds == (router.strength_q, router.env_q)
    assert len(labels_before) == 20


def test_small_regime_shrinks_to_global():
    router = RegimeRouter(min_regime_rows=35, shrinkage=80)
    weights = router.weights(
        {"A": 0.50, "B": 0.60},
        {"tiny": {"A": 0.10, "B": 2.0}},
        {"tiny": 2},
    )
    assert abs(weights["tiny"]["A"] - (1/0.50)/(1/0.50+1/0.60)) < 1e-9


def test_regime_weights_normalize():
    router = RegimeRouter(min_regime_rows=1)
    weights = router.weights(
        {"A": 0.50, "B": 1.00},
        {"s0_e0": {"A": 0.40, "B": 0.80}},
        {"s0_e0": 100},
    )
    assert abs(sum(weights["s0_e0"].values()) - 1.0) < 1e-9


def test_marginal_regime_edge_falls_back_to_global():
    router = RegimeRouter(min_regime_rows=10, shrinkage=0, min_relative_edge=0.03)
    weights = router.weights(
        {"A": 0.50, "B": 0.60},
        {"s0_e0": {"A": 0.49, "B": 0.59}},
        {"s0_e0": 100},
    )
    global_a = (1/0.50) / (1/0.50 + 1/0.60)
    assert abs(weights["s0_e0"]["A"] - global_a) < 1e-9


def test_material_regime_edge_allows_specialization():
    router = RegimeRouter(min_regime_rows=10, shrinkage=0, min_relative_edge=0.03)
    weights = router.weights(
        {"A": 0.50, "B": 0.60},
        {"s0_e0": {"A": 0.40, "B": 0.60}},
        {"s0_e0": 100},
    )
    assert weights["s0_e0"]["A"] > weights["s0_e0"]["B"]


def test_router_rejects_invalid_loss_evidence():
    import pytest
    router = RegimeRouter()
    with pytest.raises(ValueError):
        router.weights({"a": 0.0}, {"r": {"a": 1.0}}, {"r": 40})
    with pytest.raises(ValueError):
        router.weights({"a": 1.0}, {"r": {"a": float("nan")}}, {"r": 40})
    with pytest.raises(ValueError):
        router.weights({"a": 1.0}, {"r": {"a": 1.0}}, {"r": -1})
