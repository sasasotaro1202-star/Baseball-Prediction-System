import numpy as np

from research.regime_router import RegimeRouter


def test_adaptive_router_parameters_keep_weight_contract():
    router = RegimeRouter(min_regime_rows=25, shrinkage=40.0, min_relative_edge=0.02)
    weights = router.weights(
        {"A": 0.70, "B": 0.80, "C": 0.90},
        {
            "s0_e0_m0": {"A": 0.60, "B": 0.75, "C": 0.95},
            "s2_e2_m2": {"A": 0.72, "B": 0.70, "C": 0.88},
        },
        {"s0_e0_m0": 100, "s2_e2_m2": 12},
        power=1.5,
    )
    assert set(weights) == {"s0_e0_m0", "s2_e2_m2"}
    for row in weights.values():
        vals = np.asarray(list(row.values()), dtype=float)
        assert np.all(np.isfinite(vals))
        assert np.all(vals >= 0.0)
        assert np.isclose(vals.sum(), 1.0)
