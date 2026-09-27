from research.data_intelligence_candidates import (
    CANDIDATES,
    commercial_reference_candidates,
    free_research_candidates,
)


def test_candidate_registry_is_nonempty_and_unique():
    assert len(CANDIDATES) >= 20
    assert len({x.candidate_id for x in CANDIDATES}) == len(CANDIDATES)


def test_free_first_split_is_explicit():
    assert free_research_candidates()
    assert commercial_reference_candidates()
    assert all(x.free_usable_now for x in free_research_candidates())
    assert all(not x.free_usable_now for x in commercial_reference_candidates())


def test_commercial_sources_are_not_auto_acquire():
    assert all(not x.auto_acquire for x in commercial_reference_candidates())


def test_high_information_free_sources_are_present():
    ids = {x.candidate_id for x in free_research_candidates()}
    assert {"mlb_statcast", "sportsdataverse_mlb_models", "sportsdataverse_mlb_raw", "espn_mlb_public", "espn_college_baseball_public", "cpbl_trackman_open", "stormlight_baseball_api", "baseballcv", "openbiomechanics",
            "spaia_npb", "cpbl_public_2026", "omyu"} <= ids
