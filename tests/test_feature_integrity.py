import numpy as np
import pandas as pd
import pytest

from baseball_backtest import BaseballBacktest


def _bt():
    return BaseballBacktest()


def test_feature_integrity_accepts_consistent_differentials():
    bt = _bt()
    X = pd.DataFrame({
        "home_adv": [1.0, 1.0],
        "h_elo": [1500.0, 1510.0],
        "a_elo": [1490.0, 1505.0],
        "d_elo": [10.0, 5.0],
        "expected_env": [8.0, 7.0],
    })
    bt._validate_feature_matrix(X, "MLB")
    assert any(x.get("type") == "feature_integrity_pass" for x in bt.audit)


def test_feature_integrity_rejects_broken_differential():
    bt = _bt()
    X = pd.DataFrame({
        "home_adv": [1.0],
        "h_elo": [1500.0],
        "a_elo": [1490.0],
        "d_elo": [99.0],
        "expected_env": [8.0],
    })
    with pytest.raises(RuntimeError, match="differential feature invariant"):
        bt._validate_feature_matrix(X, "MLB")


def test_feature_integrity_rejects_invalid_environment():
    bt = _bt()
    X = pd.DataFrame({
        "home_adv": [1.0],
        "h_elo": [1500.0],
        "a_elo": [1490.0],
        "d_elo": [10.0],
        "expected_env": [99.0],
    })
    with pytest.raises(RuntimeError, match="expected_env outside"):
        bt._validate_feature_matrix(X, "MLB")
