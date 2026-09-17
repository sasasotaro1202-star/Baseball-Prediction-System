from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLOSED_LOOP = ROOT / ".github" / "workflows" / "baseball_closed_loop.yml"
RECOVERY = ROOT / ".github" / "workflows" / "baseball_actions_recovery.yml"


def test_closed_loop_keeps_safe_sequential_execution_and_quality_gates():
    text = CLOSED_LOOP.read_text(encoding="utf-8")

    assert "group: baseball-closed-loop" in text
    assert "queue: max" in text
    assert "cancel-in-progress: false" in text

    # PIT must be acquired and validated before any OOS research is allowed.
    pit_pos = text.index("- name: Acquire current PIT observations")
    validate_pos = text.index("- name: Validate PIT before research")
    oos_pos = text.index("- name: Run chronological Baseball OOS research")
    assert pit_pos < validate_pos < oos_pos
    assert "if stage.status != 'READY'" in text

    # Research and lifecycle gates must remain fail-closed.
    assert "obj.get('overall_status') != 'SUCCESS'" in text
    assert "obj.get('status') != 'READY'" in text


def test_recovery_is_bounded_and_only_retries_transient_steps():
    text = RECOVERY.read_text(encoding="utf-8")

    assert 'if [ "${RUN_ATTEMPT}" -ge 3 ]; then' in text
    assert 'gh run rerun "${RUN_ID}" --failed' in text
    assert "for recovery_attempt in 1 2 3; do" in text
    assert "No retryable transient failure detected; preserving the failure for diagnosis." in text

    # Keep data/model quality failures out of automatic reruns.
    assert "Install research dependencies" in text
    assert "Acquire current PIT observations" in text
    assert "Run PIT acquisition" in text
    assert "Run chronological Baseball OOS research" not in text
    assert "Enforce lifecycle completion" not in text
