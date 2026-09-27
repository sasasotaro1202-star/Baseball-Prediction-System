import json
from pathlib import Path

import pandas as pd

from research.oos_pit_join import attach_pit_evidence


def _oos(game_id: str = "G1", prediction_time: str | None = "2026-09-10T10:00:00Z") -> pd.DataFrame:
    row = {"game_id": game_id, "datetime": "2026-09-10T12:00:00Z"}
    if prediction_time is not None:
        row["prediction_time"] = prediction_time
    return pd.DataFrame([row])


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

    out, report = attach_pit_evidence(_oos(), availability_path=av, snapshots_path=snap)

    assert report["status"] == "PIT_COMPLETE"
    assert out.loc[0, "pit_join_status"] == "PIT_VERIFIED"
    assert out.loc[0, "available_at"] < out.loc[0, "prediction_time"]


def test_missing_prediction_time_is_fail_closed(tmp_path: Path):
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
        "available_at": "2026-09-10T09:59:00Z",
        "status": "KNOWN",
    }) + "\n", encoding="utf-8")

    out, report = attach_pit_evidence(_oos(prediction_time=None), availability_path=av, snapshots_path=snap)

    assert report["status"] == "NO_EXPLICIT_PREDICTION_TIME"
    assert report["matched_rows"] == 0
    assert out.loc[0, "pit_join_status"] == "UNRESOLVED"


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

    out, report = attach_pit_evidence(_oos(), availability_path=av, snapshots_path=snap)

    assert report["status"] == "PIT_PARTIAL"
    assert report["matched_rows"] == 0
    assert out.loc[0, "pit_join_status"] == "UNRESOLVED"
    assert pd.isna(out.loc[0, "available_at"])


def test_snapshot_after_prediction_time_is_rejected(tmp_path: Path):
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

    out, report = attach_pit_evidence(_oos(), availability_path=av, snapshots_path=snap)

    assert report["matched_rows"] == 0
    assert out.loc[0, "pit_join_status"] == "UNRESOLVED"


def test_latest_snapshot_after_cutoff_does_not_hide_earlier_safe_snapshot(tmp_path: Path):
    av = tmp_path / "availability.jsonl"
    snap = tmp_path / "snapshots.jsonl"
    av.write_text(json.dumps({
        "game_id": "G2",
        "event_id": "MLB:G2",
        "prediction_cutoff": "2026-09-10T10:00:00Z",
        "observed_at": "2026-09-10T10:00:00Z",
    }) + "\n", encoding="utf-8")
    snap.write_text(
        "\n".join([
            json.dumps({
                "entity_id": "MLB:G2",
                "event_id": "MLB:G2",
                "available_at": "2026-09-10T09:00:00Z",
                "status": "KNOWN",
            }),
            json.dumps({
                "entity_id": "MLB:G2",
                "event_id": "MLB:G2",
                "available_at": "2026-09-10T11:00:00Z",
                "status": "KNOWN",
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    out, report = attach_pit_evidence(_oos("G2"), availability_path=av, snapshots_path=snap)

    assert report["status"] == "PIT_COMPLETE"
    assert out.loc[0, "pit_join_status"] == "PIT_VERIFIED"
    assert out.loc[0, "available_at"] == pd.Timestamp("2026-09-10T09:00:00Z")

def test_unavailable_pit_ledger_row_cannot_be_used_as_evidence(tmp_path: Path):
    av = tmp_path / "availability.jsonl"
    snap = tmp_path / "snapshots.jsonl"
    av.write_text(json.dumps({
        "game_id": "G1",
        "event_id": "MLB:G1",
        "prediction_cutoff": "2026-09-10T10:00:00Z",
        "observed_at": "2026-09-10T10:00:00Z",
        "status": "UNAVAILABLE",
    }) + "\n", encoding="utf-8")
    snap.write_text(json.dumps({
        "entity_id": "MLB:G1",
        "event_id": "MLB:G1",
        "available_at": "2026-09-10T09:59:00Z",
        "prediction_cutoff": "2026-09-10T10:00:00Z",
        "status": "KNOWN",
    }) + "\n", encoding="utf-8")

    out, report = attach_pit_evidence(_oos(), availability_path=av, snapshots_path=snap)

    assert report["status"] == "PIT_PARTIAL"
    assert report["matched_rows"] == 0
    assert out.loc[0, "pit_join_status"] == "UNRESOLVED"
    assert out.loc[0, "pit_join_reason"] == "no_exact_pit_record_at_prediction_time"
