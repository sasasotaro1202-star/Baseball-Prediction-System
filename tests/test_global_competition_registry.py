from research.global_competition_registry import load_registry

def test_global_competition_registry_is_consistent():
    registry=load_registry()
    ids=[c["competition_id"] for c in registry["competitions"]]
    assert len(ids)==len(set(ids))
    for required in ("npb","mlb","asian_games_baseball","wbsc_u18","wbsc_u23"):
        assert required in ids

def test_registry_is_fail_closed_for_production():
    for c in load_registry()["competitions"]:
        if c["production_eligible"]:
            assert c["implementation_status"]=="IMPLEMENTED"
            assert c["pit_status"]=="PASS"
