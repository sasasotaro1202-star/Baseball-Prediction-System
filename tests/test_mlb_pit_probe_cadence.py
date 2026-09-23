import json
from datetime import datetime, timezone

import data.pit_acquisition as pit


def test_mlb_probe_cadence_reads_latest_successful_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "source_snapshots.jsonl"
    rows = [
        {"league": "MLB", "entity_type": "game_content", "entity_id": "123",
         "retrieved_at": "2026-09-23T12:00:00+00:00", "status": "KNOWN"},
        {"league": "MLB", "entity_type": "game_content", "entity_id": "123",
         "retrieved_at": "2026-09-23T12:20:00+00:00", "status": "KNOWN"},
        {"league": "MLB", "entity_type": "game_feed_timestamps", "entity_id": "123",
         "retrieved_at": "2026-09-23T11:00:00+00:00", "status": "KNOWN"},
        {"league": "MLB", "entity_type": "game_content", "entity_id": "999",
         "retrieved_at": "2026-09-23T12:20:00+00:00", "status": "UNAVAILABLE"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(pit, "SNAPSHOT_LOG", path)

    latest = pit._load_last_probe_times()
    assert latest[("game_content", "123")] == datetime(2026, 9, 23, 12, 20, tzinfo=timezone.utc)
    assert latest[("game_feed_timestamps", "123")] == datetime(2026, 9, 23, 11, 0, tzinfo=timezone.utc)
    assert ("game_content", "999") not in latest

    assert not pit._probe_due(
        latest[("game_content", "123")],
        datetime(2026, 9, 23, 12, 45, tzinfo=timezone.utc),
    )
    assert pit._probe_due(
        latest[("game_content", "123")],
        datetime(2026, 9, 23, 13, 20, tzinfo=timezone.utc),
    )
    assert pit._probe_due(None, datetime(2026, 9, 23, 12, 45, tzinfo=timezone.utc))


def test_mlb_game_start_requires_timezone():
    assert pit._mlb_game_start({"gameDate": "2026-09-23T13:00:00Z"}) == datetime(2026, 9, 23, 13, tzinfo=timezone.utc)
    assert pit._mlb_game_start({"gameDate": "2026-09-23T13:00:00"}) is None
