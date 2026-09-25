import numpy as np
import pandas as pd
import pytest

from research.npb_dynamic_loss_router import DynamicLossRouter


def _toy(n=360):
    x = pd.DataFrame({
        "d_elo": np.linspace(-2, 2, n),
        "starter_x_quality_proxy": np.sin(np.arange(n) / 20.0),
        "expected_env": 4.5 + 1.2 * np.cos(np.arange(n) / 25.0),
        "d_gd_10": np.linspace(-0.4, 0.4, n),
        "d_matches": np.full(n, 20.0),
        "d_rest_days": np.sin(np.arange(n) / 17.0),
    })
    y = (x["d_elo"].to_numpy() > 0.6).astype(int)
    y[(x["d_gd_10"].to_numpy() < -0.25)] = 2
    y[(np.arange(n) % 11) == 0] = 1
    p1 = np.column_stack([
        np.where(y == 0, 0.72, 0.14),
        np.where(y == 1, 0.14, 0.08),
        np.where(y == 2, 0.14, 0.78),
    ])
    p2 = np.column_stack([
        np.where(y == 0, 0.60, 0.20),
        np.where(y == 1, 0.28, 0.14),
        np.where(y == 2, 0.12, 0.66),
    ])
    p3 = np.column_stack([
        np.where(y == 0, 0.82, 0.09),
        np.where(y == 1, 0.09, 0.12),
        np.where(y == 2, 0.09, 0.79),
    ])
    return x, y, {"a": p1, "b": p2, "c": p3}


def test_router_rejects_short_history():
    x, y, probs = _toy(100)
    with pytest.raises(ValueError):
        DynamicLossRouter(min_train_rows=120).fit(x, y, probs, ["a", "b", "c"])


def test_router_probabilities_are_valid_and_weights_sum_to_one():
    x, y, probs = _toy()
    router = DynamicLossRouter(min_train_rows=120)
    router.fit(x.iloc[:240], y[:240], {k: v[:240] for k, v in probs.items()}, ["a", "b", "c"])
    p, w = router.predict(x.iloc[240:], {k: v[240:] for k, v in probs.items()})
    assert p.shape == (120, 3)
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-10)
    ws = np.vstack([w[k] for k in ["a", "b", "c"]]).T
    assert np.isfinite(ws).all()
    assert np.allclose(ws.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(ws > 0.0)


def test_router_uses_context_and_can_change_weights():
    x, y, probs = _toy()
    router = DynamicLossRouter(min_train_rows=120)
    router.fit(x.iloc[:240], y[:240], {k: v[:240] for k, v in probs.items()}, ["a", "b", "c"])
    _, w = router.predict(x.iloc[240:], {k: v[240:] for k, v in probs.items()})
    a = w["a"]
    c = w["c"]
    assert float(np.std(a)) > 1e-6 or float(np.std(c)) > 1e-6
