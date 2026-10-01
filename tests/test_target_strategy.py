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


def test_standard_target_bundle_matches_npb_production_contracts():
    from research.target_strategy import standard_target_strategies

    bundle = standard_target_strategies("NPB")
    assert set(bundle) == {"win_3way", "low_high", "exact_score"}
    assert bundle["win_3way"].probability_contract == "HOME_DRAW_AWAY"
    assert bundle["low_high"].probability_contract == "LOW_TOTAL_LE_6_HIGH_TOTAL_GE_7"
    assert bundle["exact_score"].probability_contract == "EXACT_SCORE_TOP4_FROM_FULL_DISTRIBUTION"


def test_standard_target_bundle_selects_two_way_for_mlb():
    from research.target_strategy import standard_target_strategies

    bundle = standard_target_strategies("MLB")
    assert set(bundle) == {"win_2way", "low_high", "exact_score"}
    assert bundle["win_2way"].probability_contract == "HOME_AWAY"


def test_standard_target_bundle_is_empty_for_unregistered_competition():
    from research.target_strategy import standard_target_strategies

    assert standard_target_strategies("KBO") == {}
