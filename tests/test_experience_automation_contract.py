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
    workflow = (ROOT / ".github/workflows/baseball_60m_pregame_auto.yml").read_text(encoding="utf-8")
    script = (ROOT / "scripts/pregame_auto.sh").read_text(encoding="utf-8")
    assert 'cron: "*/5 * * * *"' in workflow
    assert "run: bash scripts/pregame_auto.sh" in workflow
    assert "workflow_dispatch:" in workflow
    assert "python -m prediction.current_production --league NPB --date \"$date\" --data-dir data --pregame-only" in script
    assert "prediction generated at/after first pitch" in script
    assert "starter evidence observed after prediction information cutoff" in script
    assert "--min-lead-minutes 50" in script
    assert "--preferred-lead-minutes 60" in script
    assert "--scan-ahead-minutes 60" in script
    assert "--prediction-source AUTO_60M" in script
    assert "automatic target is approximately 60m before first pitch." in script
    assert "python -m research.experience_ledger --archive" in script
    assert script.count('for experience_path in data/experience/predictions data/experience/research_shadow; do') == 2
    assert script.count('if [ -d "$experience_path" ]; then') == 2
    assert script.count('git add "$experience_path"') == 2
    assert "\nrun: |\n" not in "\n" + script + "\n"
    assert not any(line.strip().startswith(("run:", "uses:", "with:", "steps:", "jobs:", "permissions:")) for line in script.splitlines())


def test_manual_current_production_entrypoint_remains_independent_of_60m_scheduler():
    text = (ROOT / "prediction/current_production.py").read_text(encoding="utf-8")
    assert "pregame_only" in text
    assert "BLOCKED_STALE_TARGET_DATE" in text
    assert "--pregame-only" in text
    # The live current-production entry point itself has no 60m scheduler gate.
    assert "preferred-lead-minutes" not in text


def test_user_prediction_shadow_experience_archive_is_automatic_and_fail_closed():
    workflow = (ROOT / ".github/workflows/npb_user_prediction_experience_archive.yml").read_text(encoding="utf-8")
    script = (ROOT / "scripts/archive_user_prediction_experience.py").read_text(encoding="utf-8")
    assert "Baseball User Prediction Request" in workflow
    assert 'prediction_requests/results/**' in workflow
    assert "python scripts/archive_user_prediction_experience.py" in workflow
    assert "data/experience/research_shadow/predictions" in workflow
    assert "fail_closed" in script
    assert '"UNKNOWN_IDENTITY"' in script
    assert "RESEARCH_SHADOW_EXECUTED" in script
    assert "production_eligibility" in script
