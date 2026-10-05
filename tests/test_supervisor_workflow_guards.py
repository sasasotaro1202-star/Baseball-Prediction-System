from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / ".github" / "workflows" / "baseball_24h_supervisor.yml"
CANDIDATE = ROOT / ".github" / "workflows" / "baseball_candidate_oos.yml"


def _text(path: Path) -> str:
    assert path.is_file(), f"missing workflow: {path}"
    return path.read_text(encoding="utf-8")


def test_supervisor_is_small_and_single_purpose():
    text = _text(SUPERVISOR)
    assert len(text.splitlines()) <= 140
    assert "CONTROL_PLANE_WORKFLOW=baseball_autonomous_control_plane.yml" in text
    assert "gh_retry workflow run baseball_autonomous_control_plane.yml" in text
    assert "cancel-in-progress: true" in text


def test_supervisor_runs_every_15_minutes_and_has_action_write():
    text = _text(SUPERVISOR)
    assert "cron: '*/15 * * * *'" in text
    assert "permissions:\n  actions: write\n  contents: read" in text
    assert "workflow_dispatch:" in text


def test_supervisor_uses_bounded_recovery_and_fail_closed_verification():
    text = _text(SUPERVISOR)
    assert "dispatches_24h" in text
    assert '"$dispatches_24h" -ge 4' in text
    assert "dispatch_age" in text
    assert '"$dispatch_age" -lt 15' in text
    assert "FAILED_REDISPATCH" in text
    assert 'exit 1' in text


def test_candidate_oos_never_cancels_an_in_progress_validation():
    text = _text(CANDIDATE)
    assert "group: baseball-candidate-oos" in text
    assert "cancel-in-progress: false" in text
    assert "matrix:\n        league: [NPB, MLB]" in text
    assert "timeout-minutes: 260" in text
