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
    assert pit.PROBE_TIMEOUT_SECONDS >= 3
    assert pit.PROBE_RETRIES >= 1
    assert pit.PROBE_MAX_GAMES <= 16


def test_mlb_game_start_requires_timezone():
    assert pit._mlb_game_start({"gameDate": "2026-09-23T13:00:00Z"}) == datetime(2026, 9, 23, 13, tzinfo=timezone.utc)
    assert pit._mlb_game_start({"gameDate": "2026-09-23T13:00:00"}) is None


def test_mlb_acquisition_persists_first_observed_starter_boundary(tmp_path, monkeypatch):
    event_log = tmp_path / "event_observations.jsonl"
    availability_log = tmp_path / "availability_observations.jsonl"
    snapshot_log = tmp_path / "source_snapshots.jsonl"
    history = {
        "game_id": "123",
        "league": "MLB",
        "home_starter_id": "456",
        "home_starter": "Pitcher A",
        "away_starter_id": "789",
        "away_starter": "Pitcher B",
        "source": pit.MLB_SCHEDULE_URL,
        "observed_at": "2026-09-23T09:30:00+00:00",
        "prediction_cutoff": "2026-09-23T09:30:00+00:00",
        "payload_hash": "history",
    }
    event_log.write_text(json.dumps(history) + "\n", encoding="utf-8")

    payload = {
        "dates": [{
            "games": [{
                "gamePk": 123,
                "gameDate": "2026-09-23T13:00:00Z",
                "teams": {
                    "home": {
                        "team": {"name": "Home"},
                        "probablePitcher": {"id": 456, "fullName": "Pitcher A"},
                    },
                    "away": {
                        "team": {"name": "Away"},
                        "probablePitcher": {"id": 789, "fullName": "Pitcher B"},
                    },
                },
            }]
        }]
    }
    monkeypatch.setattr(pit, "EVENT_LOG", event_log)
    monkeypatch.setattr(pit, "AVAILABILITY_LOG", availability_log)
    monkeypatch.setattr(pit, "SNAPSHOT_LOG", snapshot_log)
    monkeypatch.setattr(
        pit,
        "get_json",
        lambda *args, **kwargs: (payload, "2026-09-23T10:00:00+00:00"),
    )

    assert pit.acquire_mlb() == 1
    rows = [json.loads(line) for line in event_log.read_text(encoding="utf-8").splitlines()]
    current = rows[-1]
    assert current["home_starter_first_observed_at"] == "2026-09-23T09:30:00+00:00"
    assert current["away_starter_first_observed_at"] == "2026-09-23T09:30:00+00:00"
    assert current["home_starter_availability_evidence"] == "OFFICIAL_FIRST_OBSERVED"
    assert current["away_starter_availability_evidence"] == "OFFICIAL_FIRST_OBSERVED"
    assert current["home_starter_announced_at"] is None
    assert current["away_starter_announced_at"] is None
