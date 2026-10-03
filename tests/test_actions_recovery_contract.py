from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "baseball_actions_recovery.yml"


def test_recovery_defines_clock_before_zero_job_cooldown():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert 'now_epoch="$(date +%s)"' in workflow
    assert 'prior_age_minutes=$(( (now_epoch - prior_created_epoch) / 60 ))' in workflow


def test_recovery_defines_bounded_gh_cli_retry():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "gh_retry() {" in workflow
    assert 'if gh "$@"; then' in workflow
    assert 'if [ "${attempt}" -ge 3 ]; then' in workflow
