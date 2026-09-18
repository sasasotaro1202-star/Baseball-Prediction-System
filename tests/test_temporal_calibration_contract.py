import numpy as np
import pytest

from evaluation.calibration import fit_temperature_on_prefix


def test_prefix_calibration_ignores_unseen_suffix():
    prefix_p = np.array([
        [0.80, 0.20],
        [0.75, 0.25],
        [0.70, 0.30],
        [0.65, 0.35],
    ])
    prefix_y = np.array([0, 0, 1, 0])
    suffix_a = np.array([
        [0.99, 0.01],
        [0.01, 0.99],
    ])
    suffix_b = np.array([
        [0.51, 0.49],
        [0.49, 0.51],
    ])

    a = fit_temperature_on_prefix(
        np.vstack([prefix_p, suffix_a]),
        np.concatenate([prefix_y, [1, 0]]),
        train_rows=len(prefix_y),
        grid=np.linspace(0.5, 3.0, 101),
    )
    b = fit_temperature_on_prefix(
        np.vstack([prefix_p, suffix_b]),
        np.concatenate([prefix_y, [0, 1]]),
        train_rows=len(prefix_y),
        grid=np.linspace(0.5, 3.0, 101),
    )

    assert a.temperature == b.temperature


def test_prefix_calibration_requires_unseen_suffix():
    p = np.array([[0.7, 0.3], [0.4, 0.6]])
    y = np.array([0, 1])
    with pytest.raises(ValueError, match="unseen suffix"):
        fit_temperature_on_prefix(p, y, train_rows=2)
