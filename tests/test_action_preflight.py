from pathlib import Path

from research.action_preflight import _workflow_reliability_errors


GOOD = """name: test
on:
  workflow_dispatch:
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262
"""


def test_workflow_reliability_accepts_bounded_explicit_workflow():
    assert _workflow_reliability_errors(GOOD, Path(".github/workflows/good.yml")) == []


def test_workflow_reliability_rejects_unpinned_action():
    text = GOOD.replace(
        "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
        "actions/checkout@v4",
    )
    errors = _workflow_reliability_errors(text, Path("bad.yml"))
    assert any("immutable full SHA" in error for error in errors)


def test_workflow_reliability_rejects_failure_masking():
    text = GOOD.replace(
        "    timeout-minutes: 5",
        "    timeout-minutes: 5\n    continue-on-error: true",
    ).replace(
        "      - uses:",
        "      - run: false || true\n      - uses:",
    )
    errors = _workflow_reliability_errors(text, Path("bad.yml"))
    assert any("continue-on-error" in error for error in errors)
    assert any("|| true" in error for error in errors)


def test_workflow_reliability_requires_timeout_per_job():
    text = GOOD.replace("    timeout-minutes: 5\n", "")
    errors = _workflow_reliability_errors(text, Path("bad.yml"))
    assert any("timeout-minutes" in error for error in errors)


def test_workflow_reliability_requires_top_level_permissions():
    text = GOOD.replace("permissions:\n  contents: read\n", "")
    errors = _workflow_reliability_errors(text, Path("bad.yml"))
    assert any("explicit top-level permissions" in error for error in errors)
