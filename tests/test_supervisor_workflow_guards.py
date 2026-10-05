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


def test_supervisor_shell_block_is_valid_bash() -> None:
    import subprocess
    text = _text(SUPERVISOR)
    lines = text.splitlines()
    start_line = next((i for i, line in enumerate(lines) if line == "        run: |"), None)
    assert start_line is not None, "supervisor run block not found"
    end_line = next((i for i in range(start_line + 1, len(lines)) if lines[i].startswith("      - name: ")), len(lines))
    block_lines = [line[10:] if len(line) >= 10 else "" for line in lines[start_line + 1:end_line]]
    block = "\n".join(block_lines)
    result = subprocess.run(["bash", "-n"], input=block + "\n", text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr

def test_supervisor_watchdogs_autonomous_control_plane() -> None:
    text = _text(SUPERVISOR)
    assert "CONTROL_PLANE_WORKFLOW=baseball_autonomous_control_plane.yml" in text
    assert 'gh_retry run list --repo "${GH_REPO}" --workflow "${CONTROL_PLANE_WORKFLOW}"' in text
    assert 'gh_retry workflow run "${CONTROL_PLANE_WORKFLOW}" --repo "${GH_REPO}" --ref main' in text
    assert "control_dispatches_24h" in text
    assert "control_dispatch_age_minutes" in text
    assert "Control Plane terminal state: FAILED_REDISPATCH" in text
