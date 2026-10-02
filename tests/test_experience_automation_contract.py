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



def test_pregame_automation_is_five_minute_60m_pit_gated_and_archives_experience():
    text = (ROOT / ".github/workflows/baseball_30m_pregame_auto.yml").read_text(encoding="utf-8")
    assert 'cron: "*/5 * * * *"' in text
    assert "python -m prediction.current_production --league NPB --date \"$date\" --data-dir data --pregame-only" in text
    assert "prediction generated at/after first pitch" in text
    assert "starter evidence observed after prediction information cutoff" in text
    assert "--min-lead-minutes 50" in text
    assert "--preferred-lead-minutes 60" in text
    assert "--scan-ahead-minutes 70" in text
    assert "--prediction-source AUTO_60M" in text
    assert "automatic target is approximately 60m before first pitch." in text
    assert "python -m research.experience_ledger --archive" in text
    assert 'git add data/experience/predictions/' in text


def test_manual_current_production_entrypoint_remains_independent_of_60m_scheduler():
    text = (ROOT / "prediction/current_production.py").read_text(encoding="utf-8")
    assert "pregame_only" in text
    assert "BLOCKED_STALE_TARGET_DATE" in text
    assert "--pregame-only" in text
    # The live current-production entry point itself has no 60m scheduler gate.
    assert "preferred-lead-minutes" not in text
