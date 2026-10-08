from research.prediction_oss_universe_gate import load_policy, validate_policy

def test_prediction_oss_universe_is_source_attributed():
    payload = load_policy()
    assert payload["source_type"] == "USER_SUPPLIED_RANKING"
    assert payload["ranking_basis"] == "github_stars_primary"

def test_prediction_oss_universe_is_not_performance_evidence():
    payload = load_policy()
    truth = payload["source_truth_policy"]
    assert truth["ranking_is_not_performance_evidence"] is True
    assert truth["ranking_is_not_production_eligibility"] is True

def test_prediction_oss_universe_is_fail_closed():
    payload = load_policy()
    assert validate_policy(payload) == []

def test_unknown_repo_is_not_guessed():
    payload = load_policy()
    assert payload["normalization"]["do_not_guess_repository_from_display_name"] is True

def test_production_requires_holdout_stage():
    payload = load_policy()
    ladder = payload["promotion_ladder"]
    assert ladder.index("FROZEN_HOLDOUT") < ladder.index("ADOPTED")
    assert ladder.index("ADOPTED") < ladder.index("PRODUCTION")
