from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_prediction_experience_archive_has_dependency_and_schedule_fallback():
    text = (ROOT / ".github/workflows/npb_prediction_experience_archive.yml").read_text(encoding="utf-8")
    assert 'pip install --disable-pip-version-check' in text
    assert 'schedule:' in text
    assert 'gh run list --repo "${GITHUB_REPOSITORY}" --workflow "NPB Production Prediction" --status success' in text
    assert 'echo "run_id=${run_id}" >> "$GITHUB_OUTPUT"' in text
    assert 'RUN_ID: ${{ steps.download.outputs.run_id }}' in text


def test_postgame_experience_runs_multiple_daily_reconciliations_and_rollup():
    text = (ROOT / ".github/workflows/npb_postgame_experience.yml").read_text(encoding="utf-8")
    assert '- cron: "30 1,4,7,10,13,16,19,22 * * *"' in text
    assert 'python -m research.experience_ledger --reconcile' in text
    assert 'python -m research.experience_rollup' in text
    assert 'data/experience/experience_case_summary.json' in text
    assert 'data/experience/experience_training_index.json' in text
