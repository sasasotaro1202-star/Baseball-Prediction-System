import numpy as np
import pandas as pd
import pytest

from research.future_generalization_v2 import (
    FutureGeneralizationController,
    SafetyMonitor,
    disagreement_features,
    paired_block_bootstrap,
    selective_metrics,
)


def _probs(n=720, seed=7):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    out = {}
    for i, bias in enumerate((0.10, 0.00, -0.08)):
        z = rng.normal(0, 0.7 + i * 0.05, size=(n, 3)) + bias
        z[np.arange(n), y] += 0.7 + i * 0.05
        e = np.exp(z - z.max(axis=1, keepdims=True))
        out[f"M{i}"] = e / e.sum(axis=1, keepdims=True)
    return y, out


def test_disagreement_features_have_required_fields_and_no_nan():
    y, p = _probs(120)
    d = disagreement_features(p)
    required = {
        "mean_probability", "std_probability", "min_probability",
        "max_probability", "probability_range", "prediction_entropy",
        "top_class_agreement_rate", "majority_margin", "rank_disagreement",
        "pairwise_disagreement", "recent_disagreement",
        "disagreement_change_rate", "rolling_disagreement",
        "regime_conditioned_disagreement",
    }
    assert required.issubset(d.columns)
    assert np.isfinite(d.to_numpy()).all()


def test_controller_fit_route_is_finite_and_weights_sum_to_one():
    y, p = _probs()
    ctl = FutureGeneralizationController(block_size=60)
    ctl.fit(y, p)
    routed = ctl.route(p, mode="J")
    assert routed["safety"]["pass"] is True
    assert np.isfinite(routed["probabilities"]).all()
    assert np.allclose(routed["probabilities"].sum(axis=1), 1.0)
    assert np.allclose(routed["weights"].sum(axis=1), 1.0)


def test_safety_fails_closed_on_nan():
    p = np.array([[0.4, 0.6], [np.nan, 0.5]])
    result = SafetyMonitor().validate(p)
    assert result["pass"] is False
    assert result["row_valid"].tolist() == [True, False]


def test_selective_metrics_reports_fixed_coverages():
    y, p = _probs(200)
    mean_p = np.mean(np.stack(list(p.values()), axis=0), axis=0)
    conf = mean_p.max(axis=1)
    out = selective_metrics(y, mean_p, conf)
    assert set(out) == {"100%", "95%", "90%", "80%", "70%"}
    assert out["100%"]["Coverage"] == 1.0
    assert out["70%"]["rows"] <= out["80%"]["rows"]


def test_block_bootstrap_is_reproducible():
    y, p = _probs(300)
    q = np.roll(p["M0"], 1, axis=0)
    a = paired_block_bootstrap(y, q, p["M0"], block_size=30, replications=80, seed=11)
    b = paired_block_bootstrap(y, q, p["M0"], block_size=30, replications=80, seed=11)
    assert a == b


def test_controller_rejects_short_development_stream():
    y, p = _probs(120)
    with pytest.raises(ValueError, match="too short"):
        FutureGeneralizationController(block_size=30).fit(y, p)
