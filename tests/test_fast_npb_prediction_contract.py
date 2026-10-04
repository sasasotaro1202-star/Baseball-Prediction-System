from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fast_npb_wrapper_defers_only_observation_context():
    source = (ROOT / "scripts/npb_fast_prediction.py").read_text(encoding="utf-8")
    predictor = (ROOT / "production_npb.py").read_text(encoding="utf-8")

    assert "collect_npb_player_context" in source
    assert "collect_npb_roster_context" in source
    assert "collect_npb_team_player_context" in source
    assert "collect_npb_pregame_context" in source
    assert '"status": "DEFERRED_FAST_MODE"' in source
    assert "FAST_SHADOW" in source
    assert '"BASEBALL_CATBOOST_ITERATIONS", "100"' in source

    # Fast mode must call the canonical predictor rather than create a second
    # probability implementation.
    assert "predictor.predict(" in source
    assert "production_npb" in source

    # The canonical production predictor remains responsible for the PIT/model path.
    assert "official_starters(target_date)" in predictor
    assert "BaseballBacktest(Path(data_dir))" in predictor
    assert "fit_ensemble(X,y,"NPB")" in predictor


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
        ROOT / ".github/workflows/baseball-user-prediction-request.yml"
    ).read_text(encoding="utf-8")
    assert "cache: pip" in workflow
    assert "actions/cache@0400d5f644dc74513175e3cd8d07132dd4860809" in workflow
    assert "path: data/*_pbp.csv" in workflow
    assert "git/ref/tags/pbp" in workflow
