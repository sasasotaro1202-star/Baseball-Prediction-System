from __future__ import annotations

from prediction.scope_router import build_scope


def test_scope_keeps_production_and_research_separate():
    obj = build_scope()
    assert obj["policy"]["production_promotion"] == "NEVER_BY_DISCOVERY"
    assert obj["policy"]["unknown_pit"] == "FAIL_CLOSED"
    assert "NPB" in obj["current_production"] or "npb" in obj["current_production"]
    # MLB is implemented and active in the registry, but current production is
    # blocked until its independent PIT/OOS gate is satisfied.
    assert any(x["competition_id"].lower() == "mlb" and x["state"] == "RESEARCH_ACTIVE"
               for x in obj["competitions"])


def test_deferred_competitions_are_not_current_production():
    obj = build_scope()
    for cid in obj["deferred"]:
        row = next(x for x in obj["competitions"] if x["competition_id"] == cid)
        assert row["current_production"] is False
