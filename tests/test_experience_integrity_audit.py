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
    _write(pred_dir / "2026-10-01.jsonl", [row])
    result = audit_prediction_directory(pred_dir)
    assert result["status"] == "PASS_WITH_LEGACY_QUARANTINE"
    assert result["prediction_rows"] == 0
    assert result["pit_pass_rows"] == 0
    assert result["legacy_quarantined_rows"] == 1
    assert result["unique_prediction_ids"] == 1
