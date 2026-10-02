import numpy as np
import pandas as pd
import pytest

from research.retrieval_local_conformal_v2 import (
    retrieval_conformal_metrics_v2,
    retrieval_local_conformal_sets_v2,
)


def _fixture(n=90):
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
    games = np.asarray([f"game_{i}" for i in range(n)], dtype=object)
    return y, p, X, pt, mature, games


def test_game_id_is_required_and_aligned():
    y, p, X, pt, mature, games = _fixture(50)
    with pytest.raises(ValueError, match="align"):
        retrieval_local_conformal_sets_v2(
            y, p, X, pt, mature, games[:1],
            min_calibration_games=10, max_pool_games=20, k_neighbors=10,
        )


def test_insufficient_history_abstains():
    y, p, X, pt, mature, games = _fixture(20)
    out = retrieval_local_conformal_sets_v2(
        y, p, X, pt, mature, games,
        min_calibration_games=10, max_pool_games=20, k_neighbors=10,
    )
    assert out["action"][0] == "ABSTAIN"
    assert out["action"][9] == "ABSTAIN"


def test_immature_and_same_time_rows_are_excluded():
    y, p, X, pt, mature, games = _fixture(60)
    pt[41] = pt[40]
    mature[41] = pt[41] + pd.Timedelta(hours=1)
    mature[20] = pt[50]
    out = retrieval_local_conformal_sets_v2(
        y, p, X, pt, mature, games,
        min_calibration_games=10, max_pool_games=50, k_neighbors=10,
    )
    expected = int(((pt[:30] < pt[30]) & (mature[:30] <= pt[30])).sum())
    assert out["retrieval_pool_count"][30] == expected


def test_same_game_prior_revision_is_excluded():
    y, p, X, pt, mature, games = _fixture(50)
    games[20] = games[19]
    out = retrieval_local_conformal_sets_v2(
        y, p, X, pt, mature, games,
        min_calibration_games=10, max_pool_games=30, k_neighbors=10,
    )
    assert out["retrieval_pool_count"][20] == 19
    assert out["contract"]["target_game_excluded"] is True


def test_multiple_revisions_collapse_to_one_game():
    y, p, X, pt, mature, games = _fixture(70)
    games[:] = np.asarray([f"game_{i // 3}" for i in range(70)], dtype=object)
    out = retrieval_local_conformal_sets_v2(
        y, p, X, pt, mature, games,
        min_calibration_games=10, max_pool_games=20, k_neighbors=10,
    )
    assert out["retrieval_pool_count"][60] == 19
    assert out["retrieved_calibration_count"][60] == 10


def test_exact_distance_ties_have_stable_selection():
    y, p, X, pt, mature, games = _fixture(80)
    X[:] = 0.0
    a = retrieval_local_conformal_sets_v2(
        y, p, X, pt, mature, games,
        min_calibration_games=10, max_pool_games=40, k_neighbors=10,
    )
    b = retrieval_local_conformal_sets_v2(
        y, p, X, pt, mature, games,
        min_calibration_games=10, max_pool_games=40, k_neighbors=10,
    )
    assert np.array_equal(a["set_size"], b["set_size"])
    assert np.allclose(
        np.nan_to_num(a["pvalues"], nan=-1.0),
        np.nan_to_num(b["pvalues"], nan=-1.0),
    )


def test_pool_and_neighbor_bounds_are_game_based():
    y, p, X, pt, mature, games = _fixture(120)
    games[:] = np.asarray([f"game_{i // 2}" for i in range(120)], dtype=object)
    out = retrieval_local_conformal_sets_v2(
        y, p, X, pt, mature, games,
        min_calibration_games=10, max_pool_games=25, k_neighbors=12,
    )
    assert out["retrieval_pool_count"].max() <= 25
    assert out["retrieved_calibration_count"].max() <= 12


def test_invalid_maturity_and_nonfinite_features_fail_closed():
    y, p, X, pt, mature, games = _fixture(40)
    bad_mature = mature.copy()
    bad_mature[5] = pt[5] - pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="cannot precede"):
        retrieval_local_conformal_sets_v2(
            y, p, X, pt, bad_mature, games,
            min_calibration_games=10, max_pool_games=20, k_neighbors=10,
        )
    X[10, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        retrieval_local_conformal_sets_v2(
            y, p, X, pt, mature, games,
            min_calibration_games=10, max_pool_games=20, k_neighbors=10,
        )


def test_metrics_are_finite_on_eligible_cases():
    y, p, X, pt, mature, games = _fixture(60)
    out = retrieval_local_conformal_sets_v2(
        y, p, X, pt, mature, games,
        min_calibration_games=10, max_pool_games=30, k_neighbors=10,
    )
    metrics = retrieval_conformal_metrics_v2(out, y)
    assert metrics["eligible_rows"] > 0
    assert np.isfinite(metrics["coverage"])
    assert np.isfinite(metrics["mean_set_size"])


def test_production_and_promotion_remain_disabled():
    y, p, X, pt, mature, games = _fixture(60)
    out = retrieval_local_conformal_sets_v2(
        y, p, X, pt, mature, games,
        min_calibration_games=10, max_pool_games=30, k_neighbors=10,
    )
    assert out["contract"]["production_changed"] is False
    assert out["contract"]["promotion_allowed"] is False
