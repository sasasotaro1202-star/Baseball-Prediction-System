from __future__ import annotations

from research.pit_readiness_audit import _provenance_coverage


def test_source_level_pit_coverage_reports_completeness() -> None:
    rows = [
        {
            "league": "NPB",
            "source": "official",
            "available_at": "2026-10-01T04:00:00Z",
            "retrieved_at": "2026-10-01T04:05:00Z",
            "status": "KNOWN",
        },
        {
            "league": "NPB",
            "source": "official",
            "available_at": "2026-10-01T05:00:00Z",
            "retrieved_at": "2026-10-01T05:05:00Z",
            "status": "KNOWN",
        },
        {
            "league": "NPB",
            "source": "secondary",
            "available_at": "2026-10-01T04:00:00Z",
            "retrieved_at": None,
            "status": "UNVERIFIABLE",
        },
    ]

    result = _provenance_coverage(rows, "league", "source")
    official = result["groups"]["NPB|official"]
    secondary = result["groups"]["NPB|secondary"]

    assert result["available"] is True
    assert official["rows"] == 2
    assert official["available_at_coverage"] == 1.0
    assert official["retrieved_at_coverage"] == 1.0
    assert official["available_at_le_retrieved_at_coverage"] == 1.0

    assert secondary["rows"] == 1
    assert secondary["available_at_coverage"] == 1.0
    assert secondary["retrieved_at_coverage"] == 0.0
    assert secondary["available_at_le_retrieved_at_coverage"] is None
    assert secondary["availability_unverifiable"] == 1
    assert secondary["availability_status_counts"] == {"UNVERIFIABLE": 1}


def test_source_level_pit_coverage_does_not_turn_bad_order_into_pass() -> None:
    rows = [
        {
            "league": "MLB",
            "source": "official",
            "available_at": "2026-10-01T05:00:00Z",
            "retrieved_at": "2026-10-01T04:00:00Z",
            "status": "KNOWN",
        }
    ]
    result = _provenance_coverage(rows, "league", "source")
    item = result["groups"]["MLB|official"]
    assert item["available_at_coverage"] == 1.0
    assert item["retrieved_at_coverage"] == 1.0
    assert item["available_at_le_retrieved_at_coverage"] == 0.0


def test_source_level_pit_coverage_is_explicitly_unavailable_for_empty_input() -> None:
    result = _provenance_coverage([], "league", "source")
    assert result["available"] is False
    assert result["groups"] == {}
