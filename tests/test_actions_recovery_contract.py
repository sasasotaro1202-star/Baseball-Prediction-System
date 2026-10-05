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

def test_recovery_marks_unverified_zero_job_redispatch_as_failure():
    workflow = (ROOT / ".github" / "workflows" / "baseball_actions_recovery.yml").read_text(encoding="utf-8")
    marker = "Recovery terminal state: FAILED_PREGAME_ZERO_JOB_DISPATCH"
    marker_pos = workflow.index(marker)
    assert "exit 1" in workflow[marker_pos:marker_pos + 300]

def test_zero_job_path_defines_retry_helper_before_first_use():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    definition = workflow.index("gh_retry() {")
    first_use = workflow.index("gh_retry run list")
    assert definition < first_use
    assert "job_count=\"$(gh_retry run view" in workflow
    assert "prior_job_count=\"$(gh_retry run view" in workflow


def test_recovery_only_operates_on_main_branch_events():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "github.event.workflow_run.head_branch == 'main'" in workflow
    assert "github.event.workflow_run.conclusion == 'failure'" in workflow
