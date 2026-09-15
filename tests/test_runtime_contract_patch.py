from __future__ import annotations

import importlib


def test_runtime_patch_replaces_legacy_score_helpers():
    bt = importlib.import_module("baseball_backtest")
    patch = importlib.import_module("research.runtime_contract_patch")
    patch.apply()

    candidates = bt.score_candidates(3.2, 2.7, n=4)
    assert len(candidates) == 4
    assert all(score != "その他" for score, _ in candidates)
    assert len({score for score, _ in candidates}) == 4
    assert all(candidates[i][1] >= candidates[i + 1][1] for i in range(3))

    low, high = bt.low_high_probs(3.2, 2.7)
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert abs(low + high - 1.0) < 1e-12
