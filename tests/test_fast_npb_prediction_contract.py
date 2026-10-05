from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fast_npb_wrapper_defers_only_observation_context():
    source = (ROOT / "scripts/npb_fast_prediction.py").read_text(encoding="utf-8")
    predictor = (ROOT / "production_npb.py").read_text(encoding="utf-8")
    backtest = (ROOT / "baseball_backtest.py").read_text(encoding="utf-8")

    assert "collect_npb_player_context" in source
    assert "collect_npb_roster_context" in source
    assert "collect_npb_team_player_context" in source
    assert "collect_npb_pregame_context" in source
    assert '"status": "DEFERRED_FAST_MODE"' in source
    assert "FAST_SHADOW" in source
    assert '"BASEBALL_CATBOOST_ITERATIONS", "100"' in source
    # Diagnostics from the canonical predictor must not corrupt the user-facing
    # machine-readable JSON stdout protocol.
    assert "redirect_stdout" in source
    assert "StringIO()" in source
    assert "sys.stderr.write(captured)" in source
    assert "BASEBALL_FAST_MODEL_POOL_NPB" in source
    assert "BASEBALL_FAST_SCORE_MODEL_POOL_NPB" in source
    assert "BASEBALL_SCORE_REGRESSION_MAX_ITER" in source
    assert "BASEBALL_FAST_SCORE_MODEL_POOL" in backtest

    # Fast mode must call the canonical predictor rather than create a second
    # probability implementation.
    assert "predictor.predict(" in source
    assert "production_npb" in source

    # The canonical production predictor remains responsible for the PIT/model path.
    assert "official_starters(target_date)" in predictor
    assert "BaseballBacktest(Path(data_dir))" in predictor
    assert "fit_ensemble(X,y," in predictor
    assert '"NPB")' in predictor


def test_fast_lane_does_not_enable_production():
    policy = json.loads(
        (ROOT / "config/prediction_request_policy.json").read_text(encoding="utf-8")
    )
    command = policy["validated_research_shadow_commands"]["NPB"]
    assert command[0:2] == ["python", "scripts/npb_fast_prediction.py"]
    assert "--research-shadow" in command
    assert "--minimum-lead-minutes" in command
    assert "npb-production" not in " ".join(command)


def test_user_prediction_workflow_has_bounded_cache():
    workflow = (
        ROOT / ".github/workflows/baseball_manual_prediction.yml"
    ).read_text(encoding="utf-8")
    assert "cache: pip" in workflow
    assert "actions/cache@0400d5f644dc74513175e3cd8d07132dd4860809" in workflow
    assert "path: data/*_pbp.csv" in workflow
    # Cache identity follows the published PBP release fingerprint so revised
    # source data cannot silently reuse an older raw-data cache.
    assert "git/ref/tags/pbp" in workflow
    assert "id: pbp_ref" in workflow
    assert "steps.pbp_ref.outputs.sha" in workflow
    assert "hashFiles('requirements.txt', 'data/npb_pbp_adapter.py')" in workflow
    assert "steps.pbp_ref.outputs.sha" in workflow
    assert "Do not restore a different published PBP release" in workflow
    assert "gh release download pbp" in workflow
    assert '--pattern "${year}-${mm}_pbp.csv"' in workflow
    # The manual workflow must consume the wrapper's atomic JSON artifact rather
    # than parsing stdout, which may contain model diagnostics.
    assert 'source_output="results/npb_shadow_${target}.json"' in workflow
    assert 'cp "${source_output}" "${output}"' in workflow
    assert "npb_fast_stderr.log" in workflow


def test_fast_lane_has_recovery_fallbacks():
    source = (ROOT / "scripts/npb_fast_prediction.py").read_text(encoding="utf-8")
    assert "_run_daily_research_fallback" in source
    assert "_run_direct_runrate_emergency_fallback" in source
    assert "BLOCKED_STARTERS" in source
    assert '"UNVERIFIABLE"' in source
    assert '"RESEARCH_NPB_EMERGENCY_DIRECT_RUNRATE"' in source
