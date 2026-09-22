from research.real_data_validation import RealDataRun


def test_execution_failure_is_distinguished_from_completed_hold():
    failed = RealDataRun(
        "NPB", "HOLD", "error", None, None, None,
        ("RuntimeError: boom",), {}, "FAILED"
    )
    assert failed.execution_status == "FAILED"
    assert failed.stage == "error"


def test_completed_hold_is_still_an_executed_gate_decision():
    held = RealDataRun(
        "NPB", "HOLD", "locked_holdout_evaluated", 1000, 700, 200,
        ("starter_pit_evidence_not_verified",), {}, "EXECUTED"
    )
    assert held.execution_status == "EXECUTED"
    assert held.stage != "error"
