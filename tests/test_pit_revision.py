from core.pit_revision import detect_revisions


def base(**overrides):
    row = {
        "league": "NPB",
        "entity_type": "game",
        "entity_id": "g1",
        "source": "source-a",
        "payload_hash": "aaa",
        "available_at": "2026-01-01T00:00:00+00:00",
        "retrieved_at": "2026-01-01T01:00:00+00:00",
    }
    row.update(overrides)
    return row


def test_same_payload_is_not_a_revision():
    assert detect_revisions([base(), base(retrieved_at="2026-01-01T02:00:00+00:00")]) == []


def test_changed_payload_is_revision():
    findings = detect_revisions([
        base(),
        base(payload_hash="bbb", retrieved_at="2026-01-01T02:00:00+00:00"),
    ])
    assert [f.kind for f in findings] == ["REVISION"]
    assert findings[0].previous_hash == "aaa"
    assert findings[0].current_hash == "bbb"


def test_earlier_available_time_is_backfill():
    findings = detect_revisions([
        base(),
        base(payload_hash="bbb", available_at="2025-12-31T23:00:00+00:00", retrieved_at="2026-01-01T02:00:00+00:00"),
    ])
    assert [f.kind for f in findings] == ["REVISION", "BACKFILL"]


def test_different_sources_are_independent_histories():
    rows = [base(), base(source="source-b", payload_hash="bbb", retrieved_at="2026-01-01T02:00:00+00:00")]
    assert detect_revisions(rows) == []
