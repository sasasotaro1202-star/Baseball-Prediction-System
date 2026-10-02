import numpy as np
import pandas as pd
import pytest

from research.retrieval_local_conformal import (
    retrieval_conformal_metrics,
    retrieval_local_conformal_sets,
)


def _fixture(n=80):
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    pt = np.array([base + pd.Timedelta(hours=i) for i in range(n)], dtype=object)
    mature = np.array([t + pd.Timedelta(hours=1) for t in pt], dtype=object)
    y = np.array([0, 1, 2] * ((n + 2) // 3))[:n]
    p = np.tile(
        np.array([[.80, .15, .05], [.10, .80, .10], [.05, .10, .85]], dtype=float),
        ((n + 2) // 3, 1),
    )[:n]
    X = np.column_stack([
        np.sin(np.arange(n) / 7.0),
        np.cos(np.arange(n) / 11.0),
        np.asarray(y, dtype=float) * 0.02,
    ])
    return y, p, X, pt, mature


def test_same_prediction_time_is_excluded():
    y, p, X, pt, mature = _fixture()
    pt[41] = pt[40]
    mature[41] = pt[41] + pd.Timedelta(hours=1)
    out = retrieval_local_conformal_sets(
        y, p, X, pt, mature, min_calibration=10, max_pool=40, k=10
    )
    assert out["calibration_count"][41] == out["calibration_count"][40]


def test_immature_outcome_is_excluded():
    y, p, X, pt, mature = _fixture()
    mature[20] = pt[50]
    out = retrieval_local_conformal_sets(
        y, p, X, pt, mature, min_calibration=10, max_pool=50, k=10
    )
    expected = int(((pt[:30] < pt[30]) & (mature[:30] <= pt[30])).sum())
    assert out["retrieval_pool_count"][30] == expected
    assert 20 not in np.flatnonzero(
        (pt[:30] < pt[30]) & (mature[:30] <= pt[30])
    )


def test_current_row_never_enters_retrieval():
    y, p, X, pt, mature = _fixture()
    out = retrieval_local_conformal_sets(
        y, p, X, pt, mature, min_calibration=10, max_pool=40, k=10
    )
    assert out["retrieval_pool_count"][0] == 0
    assert out["retrieved_calibration_count"][0] == 0


def test_bounded_pool_and_k():
    y, p, X, pt, mature = _fixture(100)
    out = retrieval_local_conformal_sets(
        y, p, X, pt, mature, min_calibration=10, max_pool=25, k=12
    )
    assert out["retrieval_pool_count"].max() <= 25
    assert out["retrieved_calibration_count"].max() <= 12


def test_insufficient_history_abstains():
    y, p, X, pt, mature = _fixture(20)
    out = retrieval_local_conformal_sets(
        y, p, X, pt, mature, min_calibration=10, max_pool=20, k=10
    )
    assert out["action"][0] == "ABSTAIN"
    assert out["action"][9] == "ABSTAIN"


def test_unsorted_and_invalid_maturity_fail_closed():
    y, p, X, pt, mature = _fixture(20)
    bad_pt = pt.copy()
    bad_pt[5] = bad_pt[4] - pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="monotonically"):
        retrieval_local_conformal_sets(y, p, X, bad_pt, mature)
    bad_mature = mature.copy()
    bad_mature[5] = pt[5] - pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="cannot precede"):
        retrieval_local_conformal_sets(y, p, X, pt, bad_mature)


def test_nonfinite_features_fail_closed():
    y, p, X, pt, mature = _fixture()
    X[10, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        retrieval_local_conformal_sets(y, p, X, pt, mature)


def test_deterministic():
    y, p, X, pt, mature = _fixture()
    a = retrieval_local_conformal_sets(
        y, p, X, pt, mature, min_calibration=10, max_pool=40, k=10
    )
    b = retrieval_local_conformal_sets(
        y, p, X, pt, mature, min_calibration=10, max_pool=40, k=10
    )
    assert np.array_equal(a["set_size"], b["set_size"])
    assert np.allclose(
        np.nan_to_num(a["pvalues"], nan=-1.0),
        np.nan_to_num(b["pvalues"], nan=-1.0),
    )
    assert a["contract"]["production_changed"] is False
    assert a["contract"]["promotion_allowed"] is False


def test_metrics():
    y, p, X, pt, mature = _fixture(50)
    out = retrieval_local_conformal_sets(
        y, p, X, pt, mature, min_calibration=10, max_pool=30, k=10
    )
    metrics = retrieval_conformal_metrics(out, y)
    assert metrics["eligible_rows"] > 0
    assert np.isfinite(metrics["coverage"])
    assert np.isfinite(metrics["mean_set_size"])


def test_same_game_revisions_are_collapsed_before_retrieval():
    y, p, X, pt, mature = _fixture(60)
    games = np.asarray([f"game_{i // 3}" for i in range(60)], dtype=object)
    out = retrieval_local_conformal_sets(
        y, p, X, pt, mature, game_ids=games,
        min_calibration=10, max_pool=40, k=10
    )
    # After warmup, at most one prior snapshot per game enters the pool.
    eligible_games = min(40, len(set(games[:50])))
    assert out["retrieval_pool_count"][50] <= eligible_games
    assert out["contract"]["same_game_revisions_collapsed"] is True


def test_game_id_contract_fails_closed():
    y, p, X, pt, mature = _fixture(40)
    with pytest.raises(ValueError, match="align"):
        retrieval_local_conformal_sets(
            y, p, X, pt, mature, game_ids=["g1"],
            min_calibration=10, max_pool=20, k=10
        )
