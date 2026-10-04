from pathlib import Path

def test_candidate_oos_auto_controller_contract():
    workflow = Path(".github/workflows/baseball_candidate_oos_auto_controller.yml").read_text(encoding="utf-8")
    assert "cron: '*/15 * * * *'" in workflow
    assert "actions: write" in workflow
    assert "CANDIDATE_WORKFLOW: baseball_candidate_oos.yml" in workflow
    assert "candidate_active_count" in workflow
    assert "current_terminal" in workflow
    assert "compare/$latest_completed_sha...$current_main" in workflow
    assert "evidence_affecting=0" in workflow
    assert "data/experience/*|tests/*|docs/*|README*|PROJECT_INSTRUCTIONS.md|results/*|.github/workflows/*" in workflow
    assert 'gh_retry workflow run "$CANDIDATE_WORKFLOW" --repo "$GH_REPO" --ref main' in workflow
    assert "cancel-in-progress: true" in workflow
