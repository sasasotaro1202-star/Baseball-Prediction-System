import json

from research.closed_loop_governance import calibration_stage, pit_revision_stage


def test_calibration_stage_accepts_current_per_league_schema(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "leagues": {
                    "NPB": {"temperature": 0.925},
                    "MLB": {"temperature": 3.0},
                },
            }
        ),
        encoding="utf-8",
    )

    stage = calibration_stage(path)

    assert stage.status == "READY"
    assert stage.blockers == ()


def test_calibration_stage_rejects_invalid_per_league_temperature(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "leagues": {
                    "NPB": {"temperature": 0.0},
                    "MLB": {"temperature": 3.0},
                },
            }
        ),
        encoding="utf-8",
    )

    stage = calibration_stage(path)

    assert stage.status == "BLOCKED"
    assert stage.blockers == ("invalid_calibration:ValueError",)


def test_pit_revision_stage_blocks_historical_backfill(tmp_path):
    path = tmp_path / "source_snapshots.jsonl"
    rows = [
        {
            "league": "NPB",
            "entity_type": "game",
            "entity_id": "G1",
            "event_id": "G1",
            "source": "source-a",
            "payload_hash": "hash-v1",
            "available_at": "2026-09-10T12:00:00+00:00",
            "retrieved_at": "2026-09-10T12:01:00+00:00",
        },
        {
            "league": "NPB",
            "entity_type": "game",
            "entity_id": "G1",
            "event_id": "G1",
            "source": "source-a",
            "payload_hash": "hash-v2",
            "available_at": "2026-09-09T12:00:00+00:00",
            "retrieved_at": "2026-09-12T12:01:00+00:00",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    stage = pit_revision_stage(path)

    assert stage.status == "BLOCKED"
    assert stage.blockers == ("pit_backfill_detected:1",)
