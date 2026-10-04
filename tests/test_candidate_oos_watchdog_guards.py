def test_watchdog_has_no_failure_masking_shortcuts() -> None:
    text = _text()
    assert "|| true" not in text
    assert "continue-on-error" not in text


def test_watchdog_does_not_supersede_queued_runs_for_non_runtime_changes() -> None:
    text = _text()
    assert "is_non_runtime_only_change()" in text
    assert 'data/experience/*' in text
    assert 'tests/*' in text
    assert 'Keeping queued candidate run' in text
    assert 'Cancelling superseded queued candidate run' in text


def test_watchdog_does_not_dispatch_duplicate_when_current_sha_is_queued() -> None:
    text = _text()
    assert 'current_pending_count=' in text