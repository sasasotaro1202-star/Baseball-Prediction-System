import numpy as np

from baseball_backtest import BaseballBacktest


def test_validation_error_redundancy_is_oos_only_and_deterministic(tmp_path):
    bt = BaseballBacktest(tmp_path)
    y = np.array([0, 1] * 40)
    good = np.column_stack([np.where(y == 0, 0.72, 0.28), np.where(y == 1, 0.72, 0.28)])
    redundant = good.copy()
    complementary = np.column_stack([
        np.where(y == 0, 0.60, 0.40),
        np.where(y == 1, 0.60, 0.40),
    ])
    validation_predictions = {
        (40, 20): {"A": good[:20], "B": redundant[:20], "C": complementary[:20]},
        (60, 20): {"A": good[20:40], "B": redundant[20:40], "C": complementary[20:40]},
        (80, 20): {"A": good[40:60], "B": redundant[40:60], "C": complementary[40:60]},
    }
    r1 = bt._validation_error_redundancy(validation_predictions, y, ("A", "B", "C"))
    r2 = bt._validation_error_redundancy(validation_predictions, y, ("A", "B", "C"))
    assert r1 == r2
    assert set(r1) == {"A", "B", "C"}
    assert all(0.0 <= value <= 1.0 for value in r1.values())
    assert r1["A"] > 0.9
    assert r1["B"] > 0.9
