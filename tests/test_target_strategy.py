from __future__ import annotations

import pytest

from research.target_strategy import strategy_for_target


def test_npby_win_is_three_way_classification():
    s = strategy_for_target("NPB", "win_3way")
    assert s.probability_contract == "HOME_DRAW_AWAY"
    assert "classification" in s.model_family


def test_mlb_win_is_binary_classification():
    s = strategy_for_target("MLB", "win_2way")
    assert s.probability_contract == "HOME_AWAY"


def test_score_targets_use_distribution_not_win_classifier():
    for target in ("low_high", "exact_score"):
        s = strategy_for_target("MLB", target)
        assert s.model_family in {"run_distribution", "correlated_run_distribution"}


def test_three_way_contract_is_not_reused_for_mlb():
    with pytest.raises(ValueError):
        strategy_for_target("MLB", "win_3way")
