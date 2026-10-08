from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_oos_fails_closed_on_stale_main_snapshot():
    workflow = (ROOT / '.github' / 'workflows' / 'baseball_candidate_oos.yml').read_text(encoding='utf-8')
    assert 'Verify main snapshot is evidence-current' in workflow
    assert 'Verify candidate OOS snapshot remains evidence-current' in workflow
    assert 'EXPECTED_REF' in workflow
    assert 'git rev-parse origin/main' in workflow
    assert 'current_main' in workflow
    assert 'git fetch --no-tags --prune origin main' in workflow

def test_candidate_oos_watchdog_monitors_all_candidate_dependencies():
    workflow = (ROOT / '.github' / 'workflows' / 'baseball_candidate_oos.yml').read_text(encoding='utf-8')
    for path in (
        'research/real_data_validation.py',
        'research/validation_pipeline.py',
        'research/candidates.py',
        'research/regime_router.py',
        'research/correlated_score.py',
        'research/competition_taxonomy.py',
        'research/competition_strategy.py',
        'research/hierarchical_result_model.py',
        'research/individually_calibrated_ensemble.py',
        'research/adoption_gate.py',
        'baseball_backtest.py',
        'core/**',
        'evaluation/**',
    ):
        assert path in workflow

def test_candidate_oos_allows_only_non_runtime_main_updates():
    workflow = (ROOT / '.github' / 'workflows' / 'baseball_candidate_oos.yml').read_text(encoding='utf-8')
    assert 'mapfile -t changed_files' in workflow
    pit_pattern = 'tests/*|.github/workflows/baseball_candidate_oos_watchdog.yml)'
    assert workflow.count(pit_pattern) == 2
    assert 'data/pit/**' not in workflow
    start_check = workflow.split('name: Verify main snapshot is evidence-current', 1)[1].split('name: Set up Python', 1)[0]
    end_check = workflow.split('name: Verify candidate OOS snapshot remains evidence-current', 1)[1].split('name: Verify lifecycle decision is explicit', 1)[0]
    assert pit_pattern in start_check
    assert pit_pattern in end_check
    assert 'Evidence-affecting main update detected' in workflow
    assert 'refusing mixed-snapshot evidence' in workflow


def test_candidate_oos_runtime_does_not_depend_on_experience_archive():
    source_paths = (
        ROOT / "research" / "npb_candidate_replay.py",
        ROOT / "research" / "mlb_candidate_replay.py",
        ROOT / "research" / "real_data_validation.py",
        ROOT / "research" / "validation_pipeline.py",
        ROOT / "research" / "candidates.py",
        ROOT / "research" / "individually_calibrated_ensemble.py",
        ROOT / "research" / "adoption_gate.py",
        ROOT / "research" / "regime_router.py",
        ROOT / "research" / "correlated_score.py",
        ROOT / "research" / "competition_taxonomy.py",
        ROOT / "research" / "competition_strategy.py",
        ROOT / "research" / "hierarchical_result_model.py",
        ROOT / "baseball_backtest.py",
    )
    forbidden = ("data/experience", "experience_learning", "experience_ledger", "experience_rollup")
    for path in source_paths:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path