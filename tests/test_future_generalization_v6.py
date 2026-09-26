import numpy as np
import pandas as pd

from research.future_generalization_v6 import (
    HistoricalPrototypeRetriever,
    ChronologicalMetaLabeler,
    error_correlation,
    error_overlap,
    feature_reliability,
    information_shock_score,
    multi_horizon_consistency,
    next_regime_distribution,
    prediction_dynamics,
    probability_safety_gate,
    regime_transition_probabilities,
    source_reliability,
    split_conformal_sets,
    stress_probability_stream,
    time_to_failure,
    uncertainty_decomposition,
)


def _p(seed=1, n=240):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    out = {}
    for i in range(3):
        z = rng.normal(size=(n, 3))
        z[np.arange(n), y] += 1.0 + i * 0.1
        e = np.exp(z - z.max(axis=1, keepdims=True))
        out[f"M{i}"] = e / e.sum(axis=1, keepdims=True)
    return y, out


def test_dynamics_and_uncertainty_finite():
    _, p = _p()
    assert np.isfinite(prediction_dynamics(p["M0"]).to_numpy()).all()
    assert np.isfinite(uncertainty_decomposition(p).to_numpy()).all()


def test_error_diversity_shapes():
    y, p = _p()
    c = error_correlation(y, p)
    o = error_overlap(y, p)
    assert c.shape == (3, 3)
    assert o.shape == (3, 3)


def test_feature_and_source_reliability():
    x = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, 5, 5, 5]})
    r = feature_reliability(x)
    assert ((r["reliability"] >= 0) & (r["reliability"] <= 1)).all()
    s = source_reliability(pd.DataFrame({
        "freshness": [1.0, .5], "completeness": [1.0, .8], "consistency": [.9, .6]
    }))
    assert ((s["source_reliability"] >= 0) & (s["source_reliability"] <= 1)).all()


def test_retrieval_and_regime_transition_are_normalized():
    y, p = _p()
    state = prediction_dynamics(p["M0"]).join(uncertainty_decomposition(p))
    ret = HistoricalPrototypeRetriever().fit(state, y)
    q = ret.predict_proba(state.iloc[:20], 3)
    assert q.shape == (20, 3)
    assert np.allclose(q.sum(axis=1), 1.0)
    t = regime_transition_probabilities(["A", "A", "B", "A", "B", "B"])
    assert np.allclose(t.sum(axis=1), 1.0)
    assert abs(sum(next_regime_distribution(t, "A").values()) - 1.0) < 1e-12


def test_meta_labeler_predicts_without_future_labels():
    y, p = _p()
    state = prediction_dynamics(p["M0"]).iloc[:, :3]
    m = ChronologicalMetaLabeler().fit(state.iloc[:140], y[:140], p["M0"][:140])
    r = m.predict(state.iloc[140:], p["M0"][140:])
    assert r.shape == (100,)
    assert np.all(np.isfinite(r))


def test_conformal_multi_horizon_and_shock():
    y, p = _p()
    c = split_conformal_sets(p["M0"][:120], y[:120], p["M0"][120:])
    assert c["sets"].shape == (120, 3)
    h = multi_horizon_consistency({"short": p["M0"], "medium": p["M1"]})
    assert np.isfinite(h.to_numpy()).all()
    shock = information_shock_score([1, 1, 1, 1, 2, 2, 20])
    assert np.all((shock >= 0) & (shock <= 1))


def test_stress_and_probability_safety():
    _, p = _p()
    stress = stress_probability_stream(p["M0"])
    assert 0 <= stress["mean_l1_change"] <= 1
    safe = probability_safety_gate(p["M0"])
    assert safe["row_valid"].shape == (240,)


def test_time_to_failure_contract():
    out = time_to_failure(np.array([0.1, 0.5, 0.8]))
    assert np.isinf(out[0])
    assert out[1] == 2.0
    assert out[2] == 1.0
