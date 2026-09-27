from data.competition_registry import get, COMPETITIONS
from research import scope_discovery

def test_scope_registry_expands_beyond_initial_pro_leagues():
    ids = {spec.competition_id for spec in COMPETITIONS}
    for expected in {"KBO", "CPBL", "LMB", "ABL", "MILB", "JAPAN_INDEPENDENT", "LIDOM", "LMP"}:
        assert expected in ids

def test_new_competitions_remain_research_only():
    for competition_id in {"KBO", "CPBL", "LMB", "ABL", "MILB", "KBO_FUTURES"}:
        assert get(competition_id).status == "RESEARCH_ONLY"

def test_discovery_never_promotes(monkeypatch, tmp_path):
    monkeypatch.setattr(scope_discovery, "RESULTS", tmp_path)
    def fake_probe(url):
        return {"status": "REACHABLE", "signals": {"schedule": 1, "results": 1, "game": 1, "upcoming": 1, "date_tokens": 5}}
    monkeypatch.setattr(scope_discovery, "_probe", fake_probe)
    payload = scope_discovery.discover_scope()
    assert payload["status"] == "EXECUTED"
    assert payload["policy"]["production_promotion"] == "NEVER_BY_DISCOVERY"
    assert any(x["competition_id"] == "KBO" for x in payload["candidates"])
    assert all(
        x["production_eligible"] is False
        for x in payload["candidates"]
        if x["competition_id"] in {"KBO", "CPBL", "LMB", "ABL", "MILB"}
    )
    assert (tmp_path / "scope_discovery.json").exists()
def test_source_registry_creates_unregistered_frontier(monkeypatch, tmp_path):
    monkeypatch.setattr(scope_discovery, "RESULTS", tmp_path)
    monkeypatch.setattr(
        scope_discovery,
        "_probe",
        lambda url: {"status": "REACHABLE", "signals": {"schedule": 0, "results": 0, "game": 0, "upcoming": 0, "date_tokens": 0}},
    )
    payload = scope_discovery.discover_scope()
    ids = {x["competition_id"] for x in payload["unregistered_frontier"]}
    assert "NCAA" in ids


def test_catalog_scopes_are_exposed_as_frontier():
    payload = scope_discovery.discover_scope.__globals__
    assert callable(payload["catalog_scopes"])
