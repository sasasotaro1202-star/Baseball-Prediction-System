from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WATCHDOG = ROOT / ".github" / "workflows" / "baseball_candidate_oos_watchdog.yml"

def _text() -> str:
    assert WATCHDOG.is_file()
    return WATCHDOG.read_text(encoding="utf-8")

def test_watchdog_recovers_only_stale_same_sha_in_progress_runs() -> None:
    text = _text()
    assert "stale_in_progress=0" in text
    assert '[[ "${status}" == "in_progress" ]]' in text
    assert '[[ "${sha}" == "${current_sha}" ]]' in text
    assert '[[ "${age_minutes}" -ge 360 ]]' in text
    assert 'gh run cancel "${id}" --repo "${GH_REPO}"' in text

def test_watchdog_cancellation_failure_is_fail_closed() -> None:
    text = _text()
    start = text.index('if ! gh run cancel "${id}" --repo "${GH_REPO}"; then')
    end = text.index("fi", start)
    section = text[start:end]
    assert "Failed to cancel" in section
    assert "exit 1" in section

def test_watchdog_preserves_active_run_protection_and_recovery_dispatch() -> None:
    text = _text()
    assert 'if [[ "${in_progress_count}" -gt 0 ]]; then' in text
    assert 'if [[ "${stale_in_progress}" -eq 1 ]]; then' in text
    assert 'gh workflow run "${CANDIDATE_WORKFLOW}" --repo "${GH_REPO}" --ref main' in text

def test_watchdog_has_no_failure_masking_shortcuts() -> None:
    text = _text()
    assert "|| true" not in text
    assert "continue-on-error" not in text


def test_watchdog_does_not_supersede_queued_runs_for_safe_continuity_changes() -> None:
    text = _text()
    assert "is_non_runtime_only_change()" in text
    assert "tests/*" in text
    assert "docs/*" in text
    assert "PROJECT_INSTRUCTIONS.md" in text
    assert ".github/workflows/baseball_manual_prediction.yml" in text
    assert "tests/*" in text
    assert 'Keeping queued candidate run' in text
    assert 'Cancelling superseded queued candidate run' in text


def test_watchdog_does_not_dispatch_duplicate_when_current_sha_is_queued() -> None:
    text = _text()
    assert 'current_pending_count=' in text
    assert 'Candidate OOS already has current-main queued/pending work; no duplicate dispatch.' in text
def test_watchdog_cancels_evidence_stale_superseded_in_progress_runs_immediately() -> None:
    text = _text()
    assert 'if [[ "${sha}" != "${current_sha}" ]]' in text
    assert 'is_non_runtime_only_change "${sha}"' in text
    assert 'Cancelling evidence-stale superseded in-progress candidate run' in text
    assert 'elif [[ "${age_minutes}" -ge 360 ]]' in text


def test_watchdog_keeps_superseded_in_progress_for_continuity_only_changes() -> None:
    text = _text()
    assert 'Keeping superseded in-progress candidate run' in text
    assert 'main moved only through non-runtime files.' in text
def test_watchdog_refreshes_run_state_after_stale_in_progress_cancellation() -> None:
    text = _text()
    start = text.index('if [[ "${stale_in_progress}" -eq 1 ]]; then')
    end = text.index('in_progress_count=', start)
    section = text[start:end]
    assert 'runs="$(gh run list' in section
    assert '--json databaseId,status,conclusion,createdAt,headSha,number' in section
def test_watchdog_is_triggered_by_candidate_evidence_changes() -> None:
    text = _text()
    assert '  push:' in text
    for expected in (
        'research/npb_candidate_replay.py',
        'research/mlb_candidate_replay.py',
        'research/adoption_gate.py',
        'research/validation_pipeline.py',
        'baseball_backtest.py',
        '.github/workflows/baseball_candidate_oos.yml',
    ):
        assert expected in text
