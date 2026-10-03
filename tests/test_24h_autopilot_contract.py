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
