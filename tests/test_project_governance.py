from pathlib import Path

from research.project_governance import (
    source_contract_errors,
    runtime_policy_errors,
    workflow_contract_errors,
)


def test_source_contract_accepts_all_85_sections():
    text = "\n".join(f"{i}. Section {i}" for i in range(1, 86))
    text += "\n" + "\n".join(
        [
            "available_at <= prediction_cutoff",
            "retrieved_at ≠ published_at ≠ available_at",
            "HOME\nDRAW\nAWAY",
            "HOME\nAWAY",
            "LOW = total runs <= 6",
            "HIGH = total runs >= 7",
            "random split禁止",
            "PIT violations = 0",
            "NO-FAKE-SUCCESS",
            "Future Generalization",
            "Case-Level Correctness",
            "Calibration",
            "Uncertainty Quality",
            "Safe Degradation > False Prediction",
        ]
    )
    assert source_contract_errors(text) == []


def test_source_contract_rejects_missing_section_and_invariant():
    text = "\n".join(f"{i}. Section {i}" for i in range(1, 85))
    errors = source_contract_errors(text)
    assert any("project_source_sections_invalid" in x for x in errors)
    assert any("project_source_required_text_missing" in x for x in errors)


def test_runtime_policy_is_fail_closed_and_non_promoting():
    payload = {
        "policy": {
            "fail_closed": True,
            "auto_promotion": False,
            "research_fallback": False,
            "runtime_identity_is_recorded": True,
        },
        "runtimes": {},
    }
    assert runtime_policy_errors(payload) == []


def test_runtime_policy_rejects_auto_promotion():
    payload = {
        "policy": {
            "fail_closed": True,
            "auto_promotion": True,
            "research_fallback": False,
            "runtime_identity_is_recorded": True,
        },
        "runtimes": {},
    }
    assert "production_runtime_auto_promotion_must_be_false" in runtime_policy_errors(payload)


def test_workflow_contract_rejects_failure_masking_and_unpinned_actions(tmp_path: Path):
    bad = """
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    continue-on-error: true
    steps:
      - uses: actions/checkout@v4
      - run: false || true
"""
    errors = workflow_contract_errors(bad, tmp_path / "bad.yml")
    assert any("workflow_failure_masking" in x for x in errors)
    assert any("workflow_failure_masking_or_true" in x for x in errors)
    assert any("workflow_unpinned_action" in x for x in errors)


def test_source_file_is_present():
    source = Path("PROJECT_SOURCE.md")
    assert source.is_file()
    assert source.stat().st_size > 0
