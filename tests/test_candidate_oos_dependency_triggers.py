from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "baseball_candidate_oos.yml"


def test_candidate_oos_retriggers_for_candidate_governance_dependencies():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "research/individually_calibrated_ensemble.py" in workflow
    assert "research/adoption_gate.py" in workflow


def test_candidate_oos_dependency_paths_are_under_main_push_filter():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    paths = workflow.split("paths:", 1)[1].split("schedule:", 1)[0]
    assert "research/individually_calibrated_ensemble.py" in paths
    assert "research/adoption_gate.py" in paths
    assert "baseball_backtest.py" in paths
    assert "evaluation/**" in paths
