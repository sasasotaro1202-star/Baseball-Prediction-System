from __future__ import annotations

from prediction.scope_router import build_scope


def test_scope_keeps_production_and_research_separate():
    obj = build_scope()
    assert obj["policy"]["production_promotion"] == "NEVER_BY_DISCOVERY"
    assert obj["policy"]["unknown_pit"] == "FAIL_CLOSED"
    # NPB is implemented and active in the registry, but current production is
    # blocked until its independent PIT/OOS/holdout gate is satisfied.
    assert "NPB" not in obj["current_production"]
    assert any(x["competition_id"].lower() == "npb"
               and x["state"] == "RESEARCH_ACTIVE"
               and x["current_production"] is False
               for x in obj["competitions"])
    # MLB follows the same production/research separation.
    assert any(x["competition_id"].lower() == "mlb" and x["state"] == "RESEARCH_ACTIVE"
               and x["current_production"] is False
               for x in obj["competitions"])


def test_deferred_competitions_are_not_current_production():
    obj = build_scope()
    for cid in obj["deferred"]:
        row = next(x for x in obj["competitions"] if x["competition_id"] == cid)
        assert row["current_production"] is False
