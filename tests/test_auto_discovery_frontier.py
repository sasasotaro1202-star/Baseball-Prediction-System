from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from research.auto_discovery_frontier import (
    _query_for_scope,
    _result_budget,
    _rotating_focus,
    _rotation_slot,
    _score_candidate,
    load_frontier,
)


def test_discovery_scope_query_is_competition_specific():
    kbo = next(s for s in __import__("research.competition_catalog", fromlist=["SCOPES"]).SCOPES if s.scope_id == "KBO")
    assert "kbo" in _query_for_scope(kbo).lower()
    assert "dataset" in _query_for_scope(kbo).lower()


def test_discovery_score_requires_pit_to_remain_unverified():
    scope = next(s for s in __import__("research.competition_catalog", fromlist=["SCOPES"]).SCOPES if s.scope_id == "KBO")
    score, meta = _score_candidate(
        scope=scope,
        platform="github",
        name="example/kbo-pbp",
        description="historical KBO play by play pitch by pitch csv dataset 2023 2024",
        url="https://github.com/example/kbo-pbp",
        license_name="CC-BY-4.0",
        stars=100,
    )
    assert score > 42
    assert meta["pit_status"] == "UNVERIFIED"
    assert meta["production_eligible"] is False
    assert "play_by_play" in meta["features"]


def test_discovery_frontier_schema_is_fail_closed(tmp_path, monkeypatch):
    import research.auto_discovery_frontier as mod
    path = tmp_path / "frontier.json"
    path.write_text(json.dumps({"schema_version": 1, "candidates": {}, "selection_history": []}), encoding="utf-8")
    monkeypatch.setattr(mod, "OUT", path)
    payload = load_frontier()
    assert payload["schema_version"] == 1


def test_rotating_focus_is_bounded():
    focus = _rotating_focus()
    assert 1 <= len(focus) <= 12
    assert len({s.scope_id for s in focus}) == len(focus)


def test_rotating_focus_changes_on_six_hour_boundary():
    from research.competition_catalog import SCOPES
    if len(SCOPES) <= 12:
        return
    first = datetime(2026, 9, 30, 0, tzinfo=timezone.utc)
    second = first + timedelta(hours=6)
    assert _rotation_slot(second) > _rotation_slot(first)
    assert [s.scope_id for s in _rotating_focus(first)] != [s.scope_id for s in _rotating_focus(second)]


def test_result_budget_distributes_global_cap_across_scope_platform_slots():
    assert _result_budget(12) == 5
