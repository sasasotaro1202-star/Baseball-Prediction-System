from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/baseball_forever_autopilot.yml"


def test_forever_autopilot_exists_and_is_scheduled():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "name: Baseball Forever Autopilot" in text
    assert 'cron: "*/5 * * * *"' in text
    assert "workflow_dispatch: {}" in text


def test_forever_autopilot_has_narrow_bootstrap_trigger():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "push:" in text
    assert "branches: [main]" in text
    assert 'paths:\n      - ".github/workflows/baseball_forever_autopilot.yml"' in text


def test_forever_autopilot_uses_minimal_permissions_and_current_sha_gate():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "actions: write" in text
    assert "contents: read" in text
    assert "EVENT_SHA: ${{ github.sha }}" in text
    assert "git/ref/heads/main" in text
    assert '[ "$current_sha" != "$EVENT_SHA" ]' in text


def test_forever_autopilot_delegates_to_active_control_plane():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "baseball_autonomous_control_plane.yml" in text
    assert "workflow run" in text
    assert "--ref main" in text
    assert "--max-dispatches-per-cycle" not in text
    assert "DAILY_DISPATCH_CAP" in text


def test_forever_autopilot_recovers_only_bounded_states():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "scheduler_stuck_queued" in text
    assert "stale_in_progress_run" in text
    assert "startup_failure_no_jobs" in text
    assert "deterministic_failure_or_unverifiable_startup_state" in text
    assert "daily_dispatch_cap_reached" in text
    assert "gh run cancel" not in text
    assert "gh_retry run cancel" in text


def test_forever_autopilot_preserves_evidence_and_never_promotes():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "results/forever_autopilot/heartbeat.json" in text
    assert "actions/upload-artifact@" in text
    assert "auto_promotion:false" in text
    assert "fail_closed:true" in text
    assert "continue-on-error: true" not in text
    assert "|| true" not in text


def test_all_actions_are_pinned():
    text = WORKFLOW.read_text(encoding="utf-8")
    for line in text.splitlines():
        if re.match(r"^\\s*-\\s*uses:\\s+", line):
            ref = line.rsplit("@", 1)[-1].strip()
            assert re.fullmatch(r"[0-9a-fA-F]{40}", ref), line


def test_forever_autopilot_executes_control_plane_locally():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "actions/checkout@" in text
    assert "research.autonomous_control_plane" in text
    assert "--max-dispatches-per-cycle 2" in text
    assert "CONTROL_WORKFLOW" not in text
    assert "gh workflow run" not in text
    assert "timeout-minutes: 15" in text
