from __future__ import annotations

from datetime import datetime, timedelta, timezone

from monitoring.production_runtime_health import (
    LONG_RUNNING_MINUTES,
    STALE_RISK_MINUTES,
    build_health,
    classify_job,
    classify_run,
)


def test_classify_run_states() -> None:
    now = datetime(2026, 10, 1, 6, 0, tzinfo=timezone.utc)
    assert classify_run("in_progress", None, 10) == "RUNNING"
    assert classify_run("in_progress", None, LONG_RUNNING_MINUTES) == "LONG_RUNNING"
    assert classify_run("in_progress", None, STALE_RISK_MINUTES) == "STALE_RISK"
    assert classify_run("completed", "success", 10) == "SUCCESS"
    assert classify_run("completed", "failure", 10) == "FAILURE"
    assert classify_run("completed", "cancelled", 10) == "CANCELLED"
    assert classify_run("queued", None, None) == "QUEUED"
    assert now.tzinfo is not None


def test_classify_job_states() -> None:
    assert classify_job({"status": "in_progress", "conclusion": None}) == "RUNNING"
    assert classify_job({"status": "completed", "conclusion": "success"}) == "SUCCESS"
    assert classify_job({"status": "completed", "conclusion": "failure"}) == "FAILURE"
    assert classify_job({"status": "queued", "conclusion": None}) == "QUEUED"


def test_build_health_is_monitoring_only_and_exposes_operational_warnings() -> None:
    now = datetime(2026, 10, 1, 8, 30, tzinfo=timezone.utc)
    run = {
        "id": 123,
        "name": "NPB Production Prediction",
        "status": "in_progress",
        "conclusion": None,
        "head_sha": "abc",
        "created_at": "2026-10-01T06:30:00Z",
        "run_started_at": "2026-10-01T06:30:00Z",
        "updated_at": "2026-10-01T08:20:00Z",
        "html_url": "https://github.com/example/run/123",
    }
    jobs = [
        {"name": "PIT and syntax gate", "status": "completed", "conclusion": "success"},
        {"name": "Run NPB production prediction", "status": "in_progress", "conclusion": None},
    ]
    artifacts = [
        {"name": "npb-production-output-2026-10-01", "expired": False},
        {"name": "unrelated", "expired": False},
    ]

    report = build_health(
        run=run,
        jobs=jobs,
        artifacts=artifacts,
        current_main_sha="different",
        now=now,
    )

    assert report["status"] == "PASS"
    assert report["health_state"] == "STALE_RISK"
    assert report["monitoring_only"] is True
    assert report["promotion_gate"] is False
    assert report["production_run"]["head_sha_matches_current_main"] is False
    assert report["jobs"]["active"] == ["Run NPB production prediction"]
    assert report["artifacts"]["production_output_present"] is True
    assert "main_sha_mismatch" in report["warnings"]


def test_build_health_success_with_no_job_failures() -> None:
    run = {
        "id": 456,
        "name": "NPB Production Prediction",
        "status": "completed",
        "conclusion": "success",
        "head_sha": "abc",
        "created_at": "2026-10-01T05:00:00Z",
        "run_started_at": "2026-10-01T05:01:00Z",
        "updated_at": "2026-10-01T05:20:00Z",
        "html_url": "https://github.com/example/run/456",
    }
    report = build_health(
        run=run,
        jobs=[{"name": "all", "status": "completed", "conclusion": "success"}],
        artifacts=[{"name": "npb-production-output-2026-10-01", "expired": False}],
        current_main_sha="abc",
        now=datetime(2026, 10, 1, 5, 30, tzinfo=timezone.utc),
    )
    assert report["health_state"] == "SUCCESS"
    assert report["warnings"] == []


def test_build_health_marks_missing_output_artifact() -> None:
    run = {
        "id": 789,
        "name": "NPB Production Prediction",
        "status": "completed",
        "conclusion": "success",
        "head_sha": "abc",
        "created_at": "2026-10-01T05:00:00Z",
        "run_started_at": "2026-10-01T05:01:00Z",
        "updated_at": "2026-10-01T05:20:00Z",
    }
    report = build_health(
        run=run,
        jobs=[],
        artifacts=[],
        current_main_sha="abc",
        now=datetime(2026, 10, 1, 5, 30, tzinfo=timezone.utc),
    )
    assert report["health_state"] == "SUCCESS"
    assert report["artifacts"]["production_output_present"] is False
    assert "production_output_artifact_missing" in report["warnings"]
