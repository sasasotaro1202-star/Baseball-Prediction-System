from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/baseball_forever_autopilot.yml"


def test_forever_autopilot_has_independent_bootstrap_heartbeat():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "name: Baseball Forever Autopilot" in text
    assert 'cron: "*/5 * * * *"' in text
    assert "workflow_dispatch: {}" in text
    assert "push:" in text
    assert 'paths:\n      - ".github/workflows/baseball_forever_autopilot.yml"' in text
    assert "actions: write" in text
    assert "contents: read" in text
    assert "baseball_autonomous_control_plane_canonical.yml" in text
    assert "gh workflow run" in text
    assert "bootstrap_daily_cap_reached" in text
    assert "no_control_plane_history" in text
    assert "control_plane_heartbeat_stale" in text
    assert "active_control_plane_on_superseded_sha" in text
    assert "control_plane_scheduler_stuck" in text
    assert "control_plane_runtime_stale" in text
    assert "gh run cancel" in text
    assert "python -m research.autonomous_control_plane" in text
    assert "--max-dispatches-per-cycle" in text
    assert "actions/upload-artifact@" in text


def test_forever_autopilot_has_bounded_retry_and_current_main_gate():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "gh_retry() {" in text
    assert '[ "${current_sha}" != "${EVENT_SHA}" ]' in text
    assert 'EVENT_SHA: ${{ github.sha }}' in text
    assert "actions/setup-python@" in text
    assert "pytest>=8.3,<9" in text


def test_forever_autopilot_never_masks_failures():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "continue-on-error: true" not in text
    assert "|| true" not in text
    assert "set +e" in text
    assert 'if [ "${rc}" -ne 0 ]; then' in text
    assert 'exit "${rc}"' in text


def test_forever_autopilot_is_bounded():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "timeout-minutes: 8" in text
    assert "group: baseball-forever-autopilot" in text
    assert "cancel-in-progress: true" in text
    assert "cap=2" in text
    assert "cap=1" in text
