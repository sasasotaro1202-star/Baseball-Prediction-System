import numpy as np
import pandas as pd
import pytest

from baseball_backtest import clip_prob


def test_clip_prob_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        clip_prob([0.5, np.nan])
    with pytest.raises(ValueError):
        clip_prob([0.5, np.inf])
    with pytest.raises(ValueError):
        clip_prob([0.5, -0.1])


def test_clip_prob_normalizes_valid_values():
    p = clip_prob([0.2, 0.3, 0.5])
    assert np.allclose(p.sum(), 1.0)
    assert np.all(p > 0)

from research.closed_loop_execute import build_probabilities


def test_closed_loop_probability_artifact_fails_closed_on_invalid_rows():
    frame = pd.DataFrame({
        "pred_home": [0.7, float("nan")],
        "pred_draw": [0.1, 0.2],
        "pred_away": [0.2, 0.8],
    })
    with pytest.raises(RuntimeError, match="probability artifact contains 1 invalid rows"):
        build_probabilities(frame, "NPB")


def test_closed_loop_probability_artifact_accepts_valid_rows():
    frame = pd.DataFrame({
        "pred_home": [0.7],
        "pred_draw": [0.1],
        "pred_away": [0.2],
    })
    p, source = build_probabilities(frame, "NPB")
    assert source == "raw_classifier_strict"
    assert np.allclose(p[0], [0.7, 0.1, 0.2])


def test_low_high_metrics_uses_total_runs_threshold():
    from research.closed_loop_execute import hilo_metrics
    frame = pd.DataFrame({
        "actual_home_score": [4, 7, 2],
        "actual_away_score": [3, 0, 4],
    })
    # Totals: 7, 7, 6 -> HIGH, HIGH, LOW.
    p = np.array([[0.2, 0.8], [0.2, 0.8], [0.8, 0.2]], dtype=float)
    m = hilo_metrics(frame, p)
    assert m["Accuracy"] == 1.0
