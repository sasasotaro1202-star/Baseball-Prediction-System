from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/baseball_forever_autopilot.yml"


def test_forever_autopilot_exists_and_is_scheduled():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "name: Baseball Forever Autopilot" in text
    assert 'cron: "*/5 * * * *"' in text
    assert "workflow_dispatch: {}" in text


def test_forever_autopilot_has_no_self_push_or_workflow_run_trigger():
    text = WORKFLOW.read_text(encoding="utf-8")
    trigger = text.split("permissions:", 1)[0]
    assert "push:" not in trigger
    assert "workflow_run:" not in trigger
    assert "schedule:" in trigger
    assert "workflow_dispatch: {}" in trigger


def test_forever_autopilot_uses_minimal_permissions_and_current_sha_gate():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "actions: write" in text
    assert "contents: read" in text
    assert "EVENT_SHA: ${{ github.sha }}" in text
    assert "git/ref/heads/main" in text
    assert '[ "$current_sha" != "$EVENT_SHA" ]' in text


def test_forever_autopilot_executes_bounded_control_plane_locally():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "research.autonomous_control_plane" in text
    assert "--max-dispatches-per-cycle 2" in text
    assert "timeout-minutes: 15" in text
    assert "CONTROL_WORKFLOW" not in text
    assert "gh workflow run" not in text


def test_forever_autopilot_preserves_failure_evidence_and_never_promotes():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "results/forever_autopilot/heartbeat.json" in text
    assert "results/control_plane/forever_runtime_failure.json" in text
    assert "actions/upload-artifact@" in text
    assert "auto_promotion:false" in text
    assert "fail_closed:true" in text
    assert "continue-on-error: true" not in text
    assert "|| true" not in text


def test_forever_autopilot_request_recovery_is_bounded_and_reuses_existing_run():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'gh_retry run rerun "${request_latest_id}" --repo "$GH_REPO"' in text
    assert 'gh_retry workflow run "$REQUEST_WORKFLOW"' not in text
    assert "request_rerun_attempt" in text
    assert "REQUEST_DAILY_DISPATCH_CAP" in text
    assert "REQUEST_FAILURE_COOLDOWN_MINUTES" in text
    assert "current_main_sha" in text


def test_forever_autopilot_all_actions_are_pinned():
    text = WORKFLOW.read_text(encoding="utf-8")
    refs = re.findall(r"uses:\s*(actions/[^@\s]+)@([^\s#]+)", text)
    assert refs
    for action, ref in refs:
        assert re.fullmatch(r"[0-9a-fA-F]{40}", ref), f"{action} is not pinned: {ref}"
