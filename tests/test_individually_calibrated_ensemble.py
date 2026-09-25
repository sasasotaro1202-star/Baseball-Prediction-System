import numpy as np

from research.individually_calibrated_ensemble import (
    apply_calibrated_blend,
    fit_calibrated_blend,
    select_top_models,
)


def test_select_top_models_is_deterministic_and_uses_development_logloss_only():
    y = np.array([0, 1, 2, 0, 1, 2] * 5)
    good = np.array([
        [0.85, 0.10, 0.05] if t == 0 else
        [0.05, 0.85, 0.10] if t == 1 else
        [0.10, 0.05, 0.85] for t in y
    ])
    weaker = np.full((len(y), 3), 1 / 3)
    probs = {"Weaker": weaker, "Good": good, "GoodTie": good.copy()}
    assert select_top_models(y, probs, top_k=3) == ("Good", "GoodTie", "Weaker")


def test_fitted_probabilities_are_normalized_and_spec_is_frozen():
    y = np.array([0, 1, 2, 0, 1, 2] * 10)
    good = np.array([
        [0.80, 0.15, 0.05] if t == 0 else
        [0.05, 0.80, 0.15] if t == 1 else
        [0.15, 0.05, 0.80] for t in y
    ])
    alt = np.full((len(y), 3), 1 / 3)
    spec, dev = fit_calibrated_blend(y, {"Good": good, "Alt": alt}, top_k=2, weight_step=0.25)
    assert np.allclose(dev.sum(axis=1), 1.0)
    assert np.isfinite(dev).all()
    assert all(w >= 0 for w in spec.blend_weights)
    assert np.isclose(sum(spec.blend_weights), 1.0)
    before = spec.to_dict()
    holdout_inputs = {"Good": good[::-1], "Alt": alt}
    holdout = apply_calibrated_blend(spec, holdout_inputs)
    assert np.allclose(holdout.sum(axis=1), 1.0)
    assert spec.to_dict() == before


def test_apply_requires_all_locked_components_and_does_not_accept_labels():
    y = np.array([0, 1, 2] * 20)
    p = np.tile(np.array([[0.70, 0.20, 0.10], [0.10, 0.70, 0.20], [0.20, 0.10, 0.70]]), (20, 1))
    spec, _ = fit_calibrated_blend(y, {"A": p, "B": np.full_like(p, 1 / 3)}, top_k=2)
    try:
        apply_calibrated_blend(spec, {"A": p})
    except KeyError as exc:
        assert "B" in str(exc)
    else:
        raise AssertionError("missing locked component must fail closed")
