import json
from pathlib import Path

import pytest

from research.experience_integrity_audit import audit_prediction_directory


def _row(**overrides):
    row = {
        "game_id": "NPB-TEST-1",
        "datetime_jst": "2026-10-01T18:00:00+09:00",
        "prediction_cutoff_utc": "2026-10-01T08:30:00+00:00",
        "prediction_generated_at": "2026-10-01T08:31:00+00:00",
        "starter_evidence_observed_at_utc": "2026-10-01T08:20:00+00:00",
        "starter_evidence_status": "official_announced",
        "prediction_deadline_utc": "2026-10-01T08:30:00+00:00",
        "preferred_prediction_cutoff_utc": "2026-10-01T08:30:00+00:00",
        "lead_minutes_at_generation": 29.0,
        "pit_status": "PASS",
        "home_win_pct": 45.0,
        "draw_pct": 5.0,
        "away_win_pct": 50.0,
        "low_pct": 60.0,
        "high_pct": 40.0,
        "prediction_id": "prediction-1",
    }
    row.update(overrides)
    return row


def _write(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_experience_integrity_audit_passes_valid_archive(tmp_path):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    _write(pred_dir / "2026-10-01.jsonl", [_row()])
    result = audit_prediction_directory(pred_dir)
    assert result["status"] == "PASS"
    assert result["prediction_rows"] == 1
    assert result["pit_pass_rows"] == 1
    assert result["unique_prediction_ids"] == 1


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"prediction_cutoff_utc": "2026-10-01T18:00:00+00:00"}, "cutoff is not pregame"),
        ({"prediction_generated_at": "2026-10-01T18:01:00+00:00"}, "generation timing"),
        ({"starter_evidence_observed_at_utc": "2026-10-01T08:31:01+00:00"}, "observed after cutoff"),
        ({"pit_status": "FAIL"}, "not PIT PASS"),
        ({"home_win_pct": 46.0}, "win probability sum"),
        ({"high_pct": 41.0}, "Low/High probability sum"),
    ],
)
def test_experience_integrity_audit_fails_closed(tmp_path, overrides, message):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    _write(pred_dir / "2026-10-01.jsonl", [_row(**overrides)])
    with pytest.raises(ValueError, match=message):
        audit_prediction_directory(pred_dir)


def test_experience_integrity_audit_rejects_duplicate_prediction_ids(tmp_path):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    _write(
        pred_dir / "2026-10-01.jsonl",
        [_row(), _row(game_id="NPB-TEST-2")],
    )
    with pytest.raises(ValueError, match="duplicate prediction_id"):
        audit_prediction_directory(pred_dir)


def test_legacy_scheduled_cutoff_is_quarantined_not_failed(tmp_path):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    row = _row(
        prediction_generated_at="2026-10-01T03:52:00+00:00",
        prediction_cutoff_utc="2026-10-01T08:30:00+00:00",
    )
    # Deliberately remove modern timing markers so this row exercises the
    # pre-v18 scheduled-cutoff quarantine path.
    for field in (
        "lead_minutes_at_generation",
        "prediction_deadline_utc",
        "preferred_prediction_cutoff_utc",
    ):
        row.pop(field, None)
    _write(pred_dir / "2026-10-01.jsonl", [row])
    result = audit_prediction_directory(pred_dir)
    assert result["status"] == "PASS_WITH_LEGACY_QUARANTINE"
    assert result["prediction_rows"] == 0
    assert result["pit_pass_rows"] == 0
    assert result["legacy_quarantined_rows"] == 1
    assert result["unique_prediction_ids"] == 1


def test_experience_integrity_audit_checks_starter_provenance(tmp_path):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    row = _row(starter_evidence_status="unverified")
    _write(pred_dir / "2026-10-01.jsonl", [row])
    with pytest.raises(ValueError, match="unsupported starter evidence status"):
        audit_prediction_directory(pred_dir)

    row = _row(
        prediction_id="prediction-2",
        starter_source="https://example.com/not-npb",
    )
    _write(pred_dir / "2026-10-01.jsonl", [row])
    with pytest.raises(ValueError, match="non-official starter source"):
        audit_prediction_directory(pred_dir)


def test_audit_includes_generation_and_source_head_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    result = audit.audit_prediction_directory(tmp_path / "missing")
    assert result["status"] == "NO_PREDICTIONS"
    assert "generated_at_utc" in result
    assert result["source_head"] == "abc123"
