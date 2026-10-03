from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_manual_prediction_workflow_is_on_demand_and_not_pregame_slot_bound():
    workflow = (ROOT / ".github" / "workflows" / "baseball_manual_prediction.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert 'options:\n          - NPB\n          - MLB' in workflow
    assert "prediction.current_production" in workflow
    assert "--pregame-only" not in workflow
    assert "prediction_target_lead_minutes" in workflow
    assert "BLOCKED_NO_CURRENT_PRODUCTION_RUNTIME" in workflow


def test_chat_dispatch_accepts_only_exact_manual_league_commands():
    workflow = (ROOT / ".github" / "workflows" / "baseball_chat_async_dispatch.yml").read_text(encoding="utf-8")
    assert '"/baseball-predict NPB")' in workflow
    assert '"/baseball-predict MLB")' in workflow
    assert "baseball_manual_prediction.yml" in workflow
    assert "inputs" in workflow
