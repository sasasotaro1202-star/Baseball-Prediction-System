from research import autonomous_control_plane as control_plane
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
        "job_count": 1,
    }]
    result = decide(Target(".github/workflows/x.yml", 2), runs, now)
    assert result["decision"] == "HOLD"
    assert result["reason"] == "deterministic_failure_or_unverifiable_startup_state"


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


def test_list_runs_uses_workflow_scoped_history(monkeypatch):
    from research import autonomous_control_plane as control_plane

    calls = []

    def fake_gh(args):
        calls.append(list(args))
        workflow = args[args.index("--workflow") + 1]
        return (
            '[{"databaseId": 1, "status": "completed", '
            '"conclusion": "success", "createdAt": "2026-10-05T00:00:00Z", '
            '"updatedAt": "2026-10-05T00:01:00Z", "headSha": "current", '
            '"headBranch": "main", "event": "schedule", "path": "'
            + workflow
            + '"}]'
        )

    monkeypatch.setattr(control_plane, "_gh", fake_gh)
    rows = control_plane.list_runs("owner/repo")

    assert len(calls) == len(control_plane.TARGETS)
    assert {call[call.index("--workflow") + 1] for call in calls} == {
        target.workflow for target in control_plane.TARGETS
    }
    assert len(rows) == len(control_plane.TARGETS)
    assert {row["path"] for row in rows} == {
        target.workflow for target in control_plane.TARGETS
    }


def test_game_script_lab_is_under_control_plane_supervision():
    from research.autonomous_control_plane import TARGETS

    targets = {target.workflow: target for target in TARGETS}
    target = targets[".github/workflows/baseball_game_script_lab.yml"]
    assert target.heavy is True
    assert target.max_age_hours <= 8.0
    assert target.pending_recover_minutes <= 60
    assert target.max_runtime_hours <= 5.5


def test_critical_ci_targets_are_under_control_plane_supervision():
    from research.autonomous_control_plane import TARGETS
    workflows = {target.workflow for target in TARGETS}
    assert ".github/workflows/baseball_regression_tests.yml" in workflows
    assert ".github/workflows/baseball_v44_compatibility.yml" in workflows
    assert ".github/workflows/npb_game_state_research.yml" in workflows

def test_zero_job_failure_is_recoverable_startup_failure():
    from research.autonomous_control_plane import Target
    now = control_plane.datetime.now(control_plane.timezone.utc)
    target = Target("wf.yml", 12.0, heavy=True, pending_recover_minutes=60, max_runtime_hours=3.0)
    runs = [{
        "id": 123,
        "status": "completed",
        "conclusion": "failure",
        "created_at": (now - control_plane.timedelta(minutes=20)).isoformat(),
        "updated_at": now.isoformat(),
        "head_sha": "sha",
        "head_branch": "main",
        "event": "workflow_dispatch",
        "path": "wf.yml",
        "job_count": 0,
    }]
    result = control_plane.decide(target, runs, now, cap=2, current_sha="sha")
    assert result["decision"] == "DISPATCH"
    assert result["reason"] == "startup_failure_no_jobs"


def test_failure_with_jobs_remains_fail_closed():
    from research.autonomous_control_plane import Target
    now = control_plane.datetime.now(control_plane.timezone.utc)
    target = Target("wf.yml", 12.0, heavy=False, pending_recover_minutes=60, max_runtime_hours=3.0)
    runs = [{
        "id": 124,
        "status": "completed",
        "conclusion": "failure",
        "created_at": (now - control_plane.timedelta(minutes=20)).isoformat(),
        "updated_at": now.isoformat(),
        "head_sha": "sha",
        "head_branch": "main",
        "event": "workflow_dispatch",
        "path": "wf.yml",
        "job_count": 1,
    }]
    result = control_plane.decide(target, runs, now, cap=2, current_sha="sha")
    assert result["decision"] == "HOLD"
    assert result["failure_job_count"] == 1


def test_failure_with_unknown_job_count_remains_fail_closed():
    from research.autonomous_control_plane import Target
    now = control_plane.datetime.now(control_plane.timezone.utc)
    target = Target("wf.yml", 12.0, heavy=False, pending_recover_minutes=60, max_runtime_hours=3.0)
    runs = [{
        "id": 125,
        "status": "completed",
        "conclusion": "failure",
        "created_at": (now - control_plane.timedelta(minutes=20)).isoformat(),
        "updated_at": now.isoformat(),
        "head_sha": "sha",
        "head_branch": "main",
        "event": "workflow_dispatch",
        "path": "wf.yml",
    }]
    result = control_plane.decide(target, runs, now, cap=2, current_sha="sha")
    assert result["decision"] == "HOLD"


def test_transient_github_failure_is_retryable():
    assert control_plane._is_transient_gh_failure("HTTP 502 Bad Gateway")
    assert control_plane._is_transient_gh_failure("rate limit exceeded")
    assert control_plane._is_transient_gh_failure("connection reset by peer")
    assert not control_plane._is_transient_gh_failure("HTTP 404 Not Found")


def test_cancel_run_polls_only_target_run(monkeypatch):
    calls = []

    def fake_gh(args):
        calls.append(list(args))
        if args[:3] == ["run", "cancel", "99"]:
            return ""
        if args[:3] == ["run", "view", "99"] and "--json" in args:
            return "completed"
        raise AssertionError(args)

    monkeypatch.setattr(control_plane, "_gh", fake_gh)
    control_plane.cancel_run("owner/repo", 99)
    assert len(calls) == 2
    assert calls[1][:4] == ["run", "view", "99", "--repo"]


def test_dispatch_verification_is_workflow_scoped(monkeypatch):
    calls = []

    def fake_gh(args):
        calls.append(list(args))
        if args[:2] == ["workflow", "run"]:
            return ""
        if args[:3] == ["run", "list", "--repo"]:
            return '[{"databaseId": 7, "status": "queued", "conclusion": null, "createdAt": "2026-10-05T10:00:01Z", "headBranch": "main", "event": "workflow_dispatch", "path": ".github/workflows/x.yml"}]'
        raise AssertionError(args)

    monkeypatch.setattr(control_plane, "_gh", fake_gh)
    run_id = control_plane.dispatch_and_verify(
        "owner/repo",
        Target(".github/workflows/x.yml", 2),
        int(datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc).timestamp()),
    )
    assert run_id == 7
    list_calls = [call for call in calls if call[:2] == ["run", "list"]]
    assert len(list_calls) == 1
    assert ".github/workflows/x.yml" in list_calls[0]


def test_control_plane_workflow_preserves_runtime_failure_evidence():
    workflow = (
        __import__("pathlib").Path(".github/workflows/baseball_autonomous_control_plane.yml")
        .read_text(encoding="utf-8")
    )
    assert "control_plane_runtime.log" in workflow
    assert "runtime_failure.json" in workflow
    assert 'PIPESTATUS[0]' in workflow
    assert 'failure_class": "AUTONOMOUS_CONTROL_PLANE_RUNTIME"' in workflow
