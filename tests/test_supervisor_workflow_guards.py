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


def test_supervisor_is_a_small_secondary_watchdog() -> None:
    text = _text(SUPERVISOR)
    assert len(text.splitlines()) <= 140
    assert "CONTROL_PLANE_WORKFLOW=baseball_autonomous_control_plane.yml" in text
    assert "gh_retry workflow run baseball_autonomous_control_plane.yml" in text
    assert "cancel-in-progress: true" in text


def test_supervisor_uses_bounded_recovery() -> None:
    text = _text(SUPERVISOR)
    assert "dispatches_24h" in text
    assert "dispatch_age" in text
    assert '"$dispatches_24h" -ge 4' in text
    assert '"$dispatch_age" -lt 15' in text
    assert "FAILED_REDISPATCH" in text
