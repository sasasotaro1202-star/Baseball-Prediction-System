from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rapid_pregame_cli_is_research_only():
    source = (ROOT / "production_npb.py").read_text(encoding="utf-8")
    assert '"--rapid-pregame"' in source
    assert 'if args.rapid_pregame and not args.research_shadow' in source
    assert "15.0 if args.rapid_pregame" in source
    assert "180.0 if args.rapid_pregame" in source


def test_rapid_pregame_workflow_has_bounded_fast_profile_and_pit_gates():
    workflow = (ROOT / ".github" / "workflows" / "npb_rapid_pregame.yml").read_text(
        encoding="utf-8"
    )
    assert "BASEBALL_PREGAME_FAST: '1'" in workflow
    assert "BASEBALL_SCORE_FAST_VALIDATION: '1'" in workflow
    assert "BASEBALL_FAST_SCORE_MODEL_POOL_NPB: 'Poisson,HistPoisson'" in workflow
    assert "BASEBALL_TIME_BUDGET_SEC: '1500'" in workflow
    assert "BASEBALL_HARD_CAP_SEC: '1800'" in workflow
    assert "--research-shadow" in workflow
    assert "--rapid-pregame" in workflow
    assert 'obj.get("pit_status") != "PASS"' in workflow
    assert 'p.get("pit_status") != "PASS"' in workflow
    assert "cache: pip" in workflow
