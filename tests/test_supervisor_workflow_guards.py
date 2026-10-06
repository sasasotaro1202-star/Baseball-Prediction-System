from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / ".github" / "workflows" / "baseball_24h_supervisor.yml"
CANDIDATE = ROOT / ".github" / "workflows" / "baseball_candidate_oos.yml"


def _text(path: Path) -> str:
    assert path.is_file(), f"missing workflow: {path}"
    return path.read_text(encoding="utf-8")


def test_candidate_recovery_is_reachable_before_canonical_guard() -> None:
    text = _text(SUPERVISOR)
    candidate_marker = "# Candidate OOS has a separate concurrency group"
    canonical_guard = "Active canonical closed-loop runs after Candidate OOS recovery"
    restart_logic = "# Automatic restart is limited"

    assert text.count(candidate_marker) == 1
    assert text.count(canonical_guard) == 1
    assert text.index(candidate_marker) < text.index(canonical_guard) < text.index(restart_logic)


def test_supervisor_keeps_scheduler_recovery_fail_closed() -> None:
    text = _text(SUPERVISOR)

    stale_candidate_guard = next(
        line for line in text.splitlines()
        if "cage_minutes" in line and "-ge 30" in line
    )
    assert '[ "${cage_minutes}" -ge 30 ]' in stale_candidate_guard
    assert 'cstatus}" = "in_progress"' not in text
    assert 'updated_age_minutes}" -ge 150' in text
    assert 'Candidate OOS stale-run recovery dispatch verified.' in text
    assert 'Candidate OOS dispatch completed but no active run was observed.' in text


def test_supervisor_preserves_action_permissions_and_non_destructive_concurrency() -> None:
    text = _text(SUPERVISOR)

    assert "permissions:\n  actions: write\n  contents: read" in text
    assert "cancel-in-progress: true" in text


def test_candidate_oos_never_cancels_an_in_progress_validation() -> None:
    text = _text(CANDIDATE)

    assert "group: baseball-candidate-oos" in text
    assert "cancel-in-progress: false" in text
    assert "matrix:\n        league: [NPB, MLB]" in text
    assert "timeout-minutes: 260" in text

def test_supervisor_monitors_scheduled_pregame_runs_and_recovers_missed_schedule() -> None:
    text = _text(SUPERVISOR)

    assert 'actions/runs?branch=main&per_page=100' in text
    assert 'event: .event' in text
    assert 'event=push&branch=main' not in text
    assert 'latest successful pregame run is stale' in text
    assert 'pregame_latest_age_minutes}" -ge 15' in text
    assert 'Pregame missed-schedule daily cap reached' in text
    assert 'pregame_recovery_attempts_24h}" -ge 3' in text


def test_autonomous_control_plane_avoids_workflow_file_push_startup_trigger() -> None:
    text = _text(ROOT / ".github" / "workflows" / "baseball_forever_autopilot.yml")

    trigger = text.split("permissions:", 1)[0]
    assert "push:" not in trigger
    assert 'cron: "*/5 * * * *"' in trigger
    assert "workflow_dispatch: {}" in trigger
    assert "workflow_run:" not in trigger


def test_keeper_recovers_control_plane_before_24h_failover() -> None:
    text = _text(ROOT / ".github" / "workflows" / "baseball_24h_research_keeper_canonical.yml")

    assert "First recovery tier: restore the canonical control plane itself." in text
    assert "control_dispatch_attempts_24h" in text
    assert '[ "$control_dispatch_attempts_24h" -lt 2 ]' in text
    assert '[ "$control_dispatch_cooldown_minutes" -ge 15 ]' in text
    assert "CONTROL_PLANE_RECOVERED" in text
    assert "CONTROL_PLANE_RECOVERY_UNVERIFIED" in text
    assert "SUPERVISOR_WORKFLOW: baseball_24h_supervisor.yml" in text
    assert "Primary supervisor is healthy/active; keeper remains passive" in text
    assert text.index("Primary supervisor is healthy/active") < text.index("First recovery tier: restore the canonical control plane itself.")
    assert text.index("CONTROL_PLANE_RECOVERED") < text.index("24h-autopilot failover mode")


def test_24h_keeper_is_main_scoped_bounded_failover_only() -> None:
    text = _text(ROOT / ".github" / "workflows" / "baseball_24h_research_keeper_canonical.yml")

    assert '--branch main' in text
    assert 'baseball_forever_autopilot.yml' in text
    assert 'Control plane is absent/stale; entering bounded 24h-autopilot failover mode.' in text
    assert '24h autopilot already active; failover exits.' in text
    assert 'latest_failure_job_count=' in text
    assert 'DAILY_FAILOVER_CAP' in text
    assert 'latest successful 24h cycle is at least 24h old' in text
    assert 'consecutive_failure_streak' not in text

def test_supervisor_recovery_bootstraps_current_main_and_reruns_current_run():
    text = SUPERVISOR.read_text(encoding="utf-8")
    assert 'CONTROL_PLANE_WORKFLOW: baseball_forever_autopilot.yml' in text
    assert 'gh_retry workflow run "$CONTROL_PLANE_WORKFLOW" --repo "$GH_REPO" --ref main' in text
    assert 'gh_retry run rerun "$control_plane_latest_id" --repo "$GH_REPO"' in text
    assert 'control_plane_latest_sha' in text
    assert 'current-main forever heartbeat' in text
    assert 'run_attempt' in text
    assert 'current_main_sha' in text
