import numpy as np
import pandas as pd

from baseball_backtest import BaseballBacktest


def _frame(values):
    return pd.DataFrame({
        "d_elo": values,
        "expected_env": np.linspace(3.0, 7.0, len(values)),
    })


def test_regime_validation_boundaries_ignore_future_rows():
    bt = BaseballBacktest.__new__(BaseballBacktest)
    base = _frame([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    shifted_future = base.copy()
    shifted_future.iloc[6:, shifted_future.columns.get_loc("d_elo")] = [100.0, 200.0]

    splits = [(4, 2)]
    before = bt._chronological_regime_labels(base, splits)[(4, 2)]
    after = bt._chronological_regime_labels(shifted_future, splits)[(4, 2)]
    assert np.array_equal(before, after)


def test_regime_validation_uses_training_prefix_not_full_frame():
    bt = BaseballBacktest.__new__(BaseballBacktest)
    frame = _frame([0.1, 0.2, 0.3, 0.4, 0.5, 50.0])
    labels = bt._chronological_regime_labels(frame, [(4, 2)])[(4, 2)]
    assert len(labels) == 2
