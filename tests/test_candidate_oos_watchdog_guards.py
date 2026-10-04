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
    start = text.index('echo "Cancelling stale same-SHA in-progress candidate run')
    end = text.index("done < <(jq -c", start)
    section = text[start:end]
    assert "Failed to cancel stale in-progress candidate run" in section
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


def test_watchdog_does_not_supersede_queued_runs_for_non_runtime_changes() -> None:
    text = _text()
    assert "is_non_runtime_only_change()" in text
    assert 'data/experience/*|tests/*)' in text
    assert 'Keeping queued candidate run' in text
    assert 'Cancelling superseded queued candidate run' in text


def test_watchdog_does_not_dispatch_duplicate_when_current_sha_is_queued() -> None:
    text = _text()
    assert 'current_pending_count=' in text
    assert 'Candidate OOS already has current-main queued/pending work; no duplicate dispatch.' in text
