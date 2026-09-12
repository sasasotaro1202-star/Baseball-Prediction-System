from pathlib import Path

from core.pit_snapshot import append_snapshot, make_snapshot
from core.pit_replay import replay


def test_replay_excludes_future_and_unverifiable(tmp_path: Path):
    p = tmp_path / "snapshots.jsonl"
    append_snapshot(make_snapshot(
        event_id="MLB:1", league="MLB", entity_type="game", entity_id="1",
        source="TEST", payload={"starter": "A"},
        prediction_cutoff="2026-09-12T10:00:00+00:00",
        available_at="2026-09-12T09:00:00+00:00",
        retrieved_at="2026-09-12T09:05:00+00:00"), p)
    append_snapshot(make_snapshot(
        event_id="MLB:1", league="MLB", entity_type="game", entity_id="1",
        source="TEST", payload={"starter": "B"},
        prediction_cutoff="2026-09-12T11:00:00+00:00",
        available_at="2026-09-12T11:00:00+00:00",
        retrieved_at="2026-09-12T11:05:00+00:00"), p)
    append_snapshot(make_snapshot(
        event_id="MLB:1", league="MLB", entity_type="game", entity_id="1",
        source="TEST", payload={"starter": "C"},
        prediction_cutoff="2026-09-12T10:00:00+00:00",
        available_at=None,
        retrieved_at="2026-09-12T09:30:00+00:00",
        status="UNVERIFIABLE"), p)
    rows = replay(p, cutoff="2026-09-12T10:00:00+00:00")
    assert len(rows) == 1
    assert rows[0]["payload_hash"]
    assert rows[0]["available_at"] == "2026-09-12T09:00:00+00:00"


def test_replay_filters_league(tmp_path: Path):
    p = tmp_path / "snapshots.jsonl"
    for league in ("MLB", "NPB"):
        append_snapshot(make_snapshot(
            event_id=f"{league}:1", league=league, entity_type="game", entity_id="1",
            source="TEST", payload={"league": league},
            prediction_cutoff="2026-09-12T10:00:00+00:00",
            available_at="2026-09-12T09:00:00+00:00",
            retrieved_at="2026-09-12T09:01:00+00:00"), p)
    rows = replay(p, cutoff="2026-09-12T10:00:00+00:00", league="NPB")
    assert len(rows) == 1
    assert rows[0]["league"] == "NPB"
