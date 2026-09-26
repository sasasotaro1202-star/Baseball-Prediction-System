import pandas as pd
import pytest

from research.universal_dataset_contract import (
    UniversalContractError,
    normalize_research_frame,
    research_manifest,
)


def _frame(**kwargs):
    base = {
        "event_id": ["g1"],
        "event_time": ["2026-09-20T00:00:00Z"],
        "prediction_time": ["2026-09-21T00:00:00Z"],
        "available_at": ["2026-09-20T12:00:00Z"],
        "status": ["KNOWN"],
        "batting": [1.0],
    }
    base.update(kwargs)
    return pd.DataFrame(base)


def test_normalize_adds_scope_rule_and_provenance():
    out = normalize_research_frame(_frame(), scope_id="Japan_HighSchool", source_id="omyu_high_school")
    assert out.loc[0, "scope_id"] == "Japan_HighSchool"
    assert out.loc[0, "outcome_contract"] == "competition_defined"
    assert out.loc[0, "rule_family"] == "japan_high_school_specific"
    assert out.loc[0, "source_id"] == "omyu_high_school"


def test_future_data_is_rejected():
    with pytest.raises(UniversalContractError):
        normalize_research_frame(
            _frame(available_at=["2026-09-22T00:00:00Z"]),
            scope_id="Japan_HighSchool",
            source_id="omyu_high_school",
        )


def test_unknown_source_is_rejected():
    with pytest.raises(UniversalContractError):
        normalize_research_frame(_frame(), scope_id="Japan_HighSchool", source_id="does_not_exist")


def test_missing_status_is_rejected_fail_closed():
    bad = _frame().drop(columns=["status"])
    with pytest.raises(UniversalContractError):
        normalize_research_frame(bad, scope_id="Japan_HighSchool", source_id="omyu_high_school")


def test_manifest_is_auditable():
    m = research_manifest(_frame(), scope_id="Japan_HighSchool", source_id="omyu_high_school")
    assert m["rows"] == 1
    assert m["events"] == 1
    assert m["pit_status"] == "PASS"
    assert m["required_pit_rule"] == "available_at <= prediction_time"


def test_source_not_bound_to_scope_is_rejected():
    with pytest.raises(UniversalContractError):
        normalize_research_frame(
            _frame(),
            scope_id="Japan_HighSchool",
            source_id="statcast",
        )


def test_manifest_records_actual_capabilities():
    m = research_manifest(_frame(), scope_id="Japan_HighSchool", source_id="omyu_high_school")
    assert "play_by_play" in m["feature_columns_present"] or m["feature_columns_present"] == ["batting"]
