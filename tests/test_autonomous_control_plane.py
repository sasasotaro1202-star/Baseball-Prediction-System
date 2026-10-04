from datetime import datetime, timezone, timedelta

from research.autonomous_control_plane import Target, decide


def test_active_run_is_not_restarted():
    now = datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc)
    runs = [{
        "path": ".github/workflows/x.yml",
        "head_branch": "main",
        "status": "in_progress",
        "conclusion": None,
        "created_at": "2026-10-04T08:00:00Z",
        "id": 1,
    }]
    result = decide(Target(".github/workflows/x.yml", 1), runs, now)
    assert result["decision"] == "NOOP"
    assert result["reason"] == "active_in_progress_run"


def test_stale_success_dispatches():
    now = datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc)
    runs = [{
        "path": ".github/workflows/x.yml",
        "head_branch": "main",
        "status": "completed",
        "conclusion": "success",
        "created_at": "2026-10-04T07:00:00Z",
        "event": "schedule",
        "id": 1,
    }]
    result = decide(Target(".github/workflows/x.yml", 2), runs, now)
    assert result["decision"] == "DISPATCH"


def test_deterministic_failure_is_held():
    now = datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc)
    runs = [{
        "path": ".github/workflows/x.yml",
        "head_branch": "main",
        "status": "completed",
        "conclusion": "failure",
        "created_at": "2026-10-04T07:00:00Z",
        "event": "schedule",
        "id": 1,
    }]
    result = decide(Target(".github/workflows/x.yml", 2), runs, now)
    assert result["decision"] == "HOLD"
    assert result["reason"] == "deterministic_failure_is_authoritative"


def test_dispatch_cap_holds_stale_success():
    now = datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc)
    attempts = []
    for hour in (1, 5):
        attempts.append({
            "path": ".github/workflows/x.yml",
            "head_branch": "main",
            "status": "completed",
            "conclusion": "success",
            "created_at": f"2026-10-04T0{hour}:00:00Z",
            "event": "workflow_dispatch",
            "id": hour,
        })
    result = decide(Target(".github/workflows/x.yml", 2), attempts, now)
    assert result["decision"] == "HOLD"
    assert result["reason"] == "dispatch_cap_reached"


def test_scheduler_stuck_pending_is_recovered():
    now = datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc)
    runs = [{
        "path": ".github/workflows/x.yml",
        "head_branch": "main",
        "status": "pending",
        "conclusion": None,
        "created_at": "2026-10-04T08:00:00Z",
        "id": 7,
        "head_sha": "current",
    }]
    result = decide(Target(".github/workflows/x.yml", 2, pending_recover_minutes=30), runs, now, current_sha="current")
    assert result["decision"] == "RECOVER"
    assert result["reason"] == "scheduler_stuck_pending"


def test_superseded_pending_is_recovered():
    now = datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc)
    runs = [{
        "path": ".github/workflows/x.yml",
        "head_branch": "main",
        "status": "pending",
        "conclusion": None,
        "created_at": "2026-10-04T09:59:00Z",
        "id": 8,
        "head_sha": "old",
    }]
    result = decide(Target(".github/workflows/x.yml", 2), runs, now, current_sha="current")
    assert result["decision"] == "RECOVER"
    assert result["reason"] == "queued_run_on_superseded_sha"


def test_stale_in_progress_is_recovered():
    now = datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc)
    runs = [{
        "path": ".github/workflows/x.yml",
        "head_branch": "main",
        "status": "in_progress",
        "conclusion": None,
        "created_at": "2026-10-04T06:00:00Z",
        "id": 9,
        "head_sha": "current",
    }]
    result = decide(Target(".github/workflows/x.yml", 2, max_runtime_hours=3), runs, now, current_sha="current")
    assert result["decision"] == "RECOVER"
    assert result["reason"] == "stale_in_progress_run"


def test_expected_skipped_is_healthy():
    now = datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc)
    runs = [{
        "path": ".github/workflows/x.yml",
        "head_branch": "main",
        "status": "completed",
        "conclusion": "skipped",
        "created_at": "2026-10-03T00:00:00Z",
        "id": 10,
    }]
    result = decide(Target(".github/workflows/x.yml", 2, skip_is_healthy=True), runs, now)
    assert result["decision"] == "NOOP"
    assert result["reason"] == "expected_skipped_state"
