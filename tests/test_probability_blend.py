import numpy as np

from baseball_backtest import BaseballBacktest


def test_probability_blend_linear_and_log_pool_are_coherent():
    members = {
        "a": np.array([[0.80, 0.15, 0.05], [0.10, 0.20, 0.70]]),
        "b": np.array([[0.60, 0.30, 0.10], [0.25, 0.25, 0.50]]),
    }
    weights = {"a": 0.7, "b": 0.3}
    linear = BaseballBacktest._blend_probability_members(members, weights, "linear")
    log_pool = BaseballBacktest._blend_probability_members(members, weights, "log_pool")

    assert linear.shape == (2, 3)
    assert log_pool.shape == (2, 3)
    assert np.isfinite(linear).all()
    assert np.isfinite(log_pool).all()
    assert np.all(linear > 0)
    assert np.all(log_pool > 0)
    assert np.allclose(linear.sum(axis=1), 1.0)
    assert np.allclose(log_pool.sum(axis=1), 1.0)
    assert not np.allclose(linear, log_pool)


def test_probability_blend_rejects_zero_total_weight():
    members = {"a": np.array([[0.5, 0.3, 0.2]])}
    try:
        BaseballBacktest._blend_probability_members(members, {"a": 0.0}, "log_pool")
    except ValueError as exc:
        assert "positive finite sum" in str(exc)
    else:
        raise AssertionError("zero-total probability blend weights must fail closed")


def test_stack_features_preserve_member_order_and_normalize():
    members = {
        "a": np.array([[0.8, 0.1, 0.1]]),
        "b": np.array([[2.0, 1.0, 1.0]]),
    }
    features = BaseballBacktest._stack_features(members, ("a", "b"))
    assert features.shape == (1, 6)
    assert np.allclose(features[0, :3], [0.8, 0.1, 0.1])
    assert np.allclose(features[0, 3:], [0.5, 0.25, 0.25])


def test_stack_features_reject_missing_member():
    members = {"a": np.array([[0.7, 0.2, 0.1]])}
    try:
        BaseballBacktest._stack_features(members, ("a", "b"))
    except ValueError as exc:
        assert "missing stacker member" in str(exc)
    else:
        raise AssertionError("missing stacker members must fail closed")
