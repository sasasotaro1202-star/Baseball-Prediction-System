from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_oos_fails_closed_on_stale_main_snapshot():
    workflow = (ROOT / '.github' / 'workflows' / 'baseball_candidate_oos.yml').read_text(encoding='utf-8')
    assert 'name: Verify main snapshot is current' in workflow
    assert 'name: Verify main snapshot is still current' in workflow
    assert workflow.count('test "$current_main" = "$EXPECTED_REF"') >= 2
    assert 'git fetch --no-tags --prune origin main' in workflow

def test_candidate_oos_watchdog_monitors_all_candidate_dependencies():
    workflow = (ROOT / '.github' / 'workflows' / 'baseball_candidate_oos_watchdog.yml').read_text(encoding='utf-8')
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
