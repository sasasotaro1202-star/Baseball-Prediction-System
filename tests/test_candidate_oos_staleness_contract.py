from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_oos_fails_closed_on_stale_main_snapshot():
    workflow = (ROOT / '.github' / 'workflows' / 'baseball_candidate_oos.yml').read_text(encoding='utf-8')
    assert 'name: Verify main snapshot is current' in workflow
    assert 'name: Verify main snapshot is still current' in workflow
    assert workflow.count('test "$current_main" = "$EXPECTED_REF"') >= 2
    assert 'git fetch --no-tags --prune origin main' in workflow