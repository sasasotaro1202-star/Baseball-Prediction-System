import pandas as pd
import pytest

from research.competition_catalog import get_scope, scopes
from research.universal_source_matrix import application_matrix, application_plan, pit_safe_rows


def test_catalog_covers_multiple_age_levels():
    levels = {s.level for s in scopes()}
    assert {"professional", "college", "high_school", "junior", "elementary", "international_age_group"} <= levels


def test_high_school_plan_includes_public_research_sources():
    rows = application_plan("Japan_HighSchool")
    source_ids = {r["source_id"] for r in rows}
    assert "omyu_high_school" in source_ids
    assert "jhbf_official" in source_ids
    assert all(r["pit_requirement"] == "explicit available_at <= prediction_time" for r in rows)


def test_matrix_contains_every_catalog_scope():
    rows = application_matrix()
    scopes_seen = {r["scope_id"] for r in rows}
    assert scopes_seen == {s.scope_id for s in scopes()}


def test_pit_filter_fails_closed_without_timestamps():
    with pytest.raises(ValueError):
        pit_safe_rows(pd.DataFrame([{"value": 1}]))


def test_pit_filter_rejects_future_and_unknown():
    frame = pd.DataFrame([
        {"id": "pass", "available_at": "2026-09-01T00:00:00Z", "prediction_time": "2026-09-02T00:00:00Z", "status": "KNOWN"},
        {"id": "future", "available_at": "2026-09-03T00:00:00Z", "prediction_time": "2026-09-02T00:00:00Z", "status": "KNOWN"},
        {"id": "unknown", "available_at": "2026-09-01T00:00:00Z", "prediction_time": "2026-09-02T00:00:00Z", "status": "UNVERIFIED"},
        {"id": "missing", "available_at": None, "prediction_time": "2026-09-02T00:00:00Z", "status": "KNOWN"},
    ])
    out = pit_safe_rows(frame)
    assert out["id"].tolist() == ["pass"]


def test_scope_rule_metadata_is_explicit():
    scope = get_scope("Japan_U12")
    assert scope.outcome_contract == "competition_defined"
    assert scope.rule_family == "wbsc_age_group_specific"
