import json
from pathlib import Path

import pandas as pd

from research.oos_pit_join import attach_pit_evidence


def test_exact_snapshot_is_joined_without_inference(tmp_path: Path):
    av = tmp_path / "availability.jsonl"
    snap = tmp_path / "snapshots.jsonl"
    av.write_text(json.dumps({
        "game_id": "G1",
        "event_id": "MLB:G1",
        "prediction_cutoff": "2026-09-10T10:00:00Z",
        "observed_at": "2026-09-10T10:00:00Z",
        "status": "OBSERVED_UNVERIFIABLE_ANNOUNCEMENT_TIME",
    }) + "\n", encoding="utf-8")
    snap.write_text(json.dumps({
        "entity_id": "MLB:G1",
        "event_id": "MLB:G1",
        "available_at": "2026-09-10T09:59:00Z",
        "prediction_cutoff": "2026-09-10T10:00:00Z",
        "status": "KNOWN",
    }) + "\n", encoding="utf-8")

    oos = pd.DataFrame([{"game_id": "G1", "datetime": "2026-09-10T12:00:00Z"}])
    out, report = attach_pit_evidence(oos, availability_path=av, snapshots_path=snap)

    assert report["status"] == "PIT_COMPLETE"
    assert out.loc[0, "pit_join_status"] == "PIT_VERIFIED"
    assert out.loc[0, "available_at"] < out.loc[0, "prediction_time"]


def test_missing_game_is_unresolved(tmp_path: Path):
    av = tmp_path / "availability.jsonl"
    snap = tmp_path / "snapshots.jsonl"
    av.write_text(json.dumps({
        "game_id": "OTHER",
        "event_id": "MLB:OTHER",
        "prediction_cutoff": "2026-09-10T10:00:00Z",
        "observed_at": "2026-09-10T10:00:00Z",
    }) + "\n", encoding="utf-8")
    snap.write_text("", encoding="utf-8")

    oos = pd.DataFrame([{"game_id": "G1", "datetime": "2026-09-10T12:00:00Z"}])
    out, report = attach_pit_evidence(oos, availability_path=av, snapshots_path=snap)

    assert report["status"] == "PIT_PARTIAL"
    assert report["matched_rows"] == 0
    assert out.loc[0, "pit_join_status"] == "UNRESOLVED"
    assert pd.isna(out.loc[0, "prediction_time"])


def test_snapshot_after_cutoff_is_rejected(tmp_path: Path):
    av = tmp_path / "availability.jsonl"
    snap = tmp_path / "snapshots.jsonl"
    av.write_text(json.dumps({
        "game_id": "G1",
        "event_id": "MLB:G1",
        "prediction_cutoff": "2026-09-10T10:00:00Z",
        "observed_at": "2026-09-10T10:00:00Z",
    }) + "\n", encoding="utf-8")
    snap.write_text(json.dumps({
        "entity_id": "MLB:G1",
        "event_id": "MLB:G1",
        "available_at": "2026-09-10T11:00:00Z",
        "prediction_cutoff": "2026-09-10T10:00:00Z",
        "status": "KNOWN",
    }) + "\n", encoding="utf-8")

    oos = pd.DataFrame([{"game_id": "G1", "datetime": "2026-09-10T12:00:00Z"}])
    out, report = attach_pit_evidence(oos, availability_path=av, snapshots_path=snap)

    assert report["matched_rows"] == 0
    assert out.loc[0, "pit_join_status"] == "UNRESOLVED"
