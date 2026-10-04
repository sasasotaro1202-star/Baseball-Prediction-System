from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CHECKOUT_SHA = "11bd71901bbe5b1630ceea73d27597364c9af683"


def test_pregame_context_workflow_uses_resolvable_checkout_pin():
    workflow = (ROOT / ".github" / "workflows" / "baseball_pregame_context.yml").read_text(encoding="utf-8")
    assert f"actions/checkout@{EXPECTED_CHECKOUT_SHA}" in workflow
    assert "actions/checkout@3d3c42a5aac5ba805825da76410c181273ba90b1" not in workflow
