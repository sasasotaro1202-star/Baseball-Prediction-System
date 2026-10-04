from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "baseball_24h_research_autopilot.yml"


def test_24h_autopilot_targeted_test_paths_are_discoverable_and_present():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    paths = sorted(set(re.findall(r"tests/[A-Za-z0-9_.-]+\.py", workflow)))
    assert paths, "24h autopilot must expose concrete test paths"
    missing = [path for path in paths if not (ROOT / path).is_file()]
    assert missing == []


def test_24h_autopilot_frontier_lane_declares_requests_dependency():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "python -m pip install --disable-pip-version-check --retries 10 --timeout 120 pandas pytest requests" in workflow

def test_24h_autopilot_dependent_research_waves_fail_closed():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "needs['wave2-frontier'].result == 'success'" in workflow
    assert "needs['wave3-candidate-oos'].result == 'success'" in workflow
    assert "closeout:" in workflow
    assert "needs: [wave1-core-oos, wave2-frontier, wave3-candidate-oos, wave4-meta-research]" in workflow

RELIABILITY_WORKFLOW = ROOT / ".github" / "workflows" / "reliability_preflight.yml"


def test_reliability_preflight_concurrency_is_ref_scoped():
    workflow = RELIABILITY_WORKFLOW.read_text(encoding="utf-8")
    assert "group: reliability-preflight-${{ github.event.pull_request.number || github.ref }}" in workflow


def test_24h_autopilot_has_current_main_staleness_guard():
    workflow = (ROOT / ".github" / "workflows" / "baseball_24h_research_autopilot.yml").read_text(encoding="utf-8")
    assert workflow.count("name: Verify main has not advanced") == 4
    assert "current_main=\"$(git rev-parse origin/main)\"" in workflow
    assert "test \"$current_main\" = \"$EXPECTED_REF\"" in workflow
    assert "'stale_run':stale_run" in workflow
