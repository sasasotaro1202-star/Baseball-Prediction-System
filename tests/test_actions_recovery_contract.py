from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "baseball_24h_supervisor.yml"


def test_recovery_defines_clock_before_zero_job_cooldown():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "now_epoch=" in workflow
    assert "pregame_recovery_age_minutes" in workflow
    assert "latest_failure_job_count=" in workflow


def test_recovery_defines_bounded_gh_cli_retry():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "gh_retry() {" in workflow
    assert "for attempt in 1 2 3 4" in workflow
    assert "gh_retry workflow run" in workflow


def test_recovery_marks_unverified_zero_job_redispatch_as_failure():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "PRE_GAME_ZERO_JOB_REDISPATCHED" in workflow
    assert "PRE_GAME_ZERO_JOB_DISPATCH_FAILED" in workflow or "Pregame recovery dispatch was accepted but no active run was observed." in workflow


def test_zero_job_path_defines_retry_helper_before_first_use():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert workflow.index("gh_retry() {") < workflow.index("gh_retry workflow run")
    assert "latest_failure_job_count=" in workflow
    assert "pregame_recovery_attempts_24h" in workflow


def test_recovery_only_operates_on_main_branch_events():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "TARGET_WORKFLOW: baseball_closed_loop.yml" in workflow
    assert "CONTROL_PLANE_WORKFLOW: baseball_forever_autopilot.yml" in workflow
    assert "branch main" in workflow

