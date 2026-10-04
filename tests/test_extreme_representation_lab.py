from __future__ import annotations
import json
import numpy as np
import pandas as pd
import pytest
from research.extreme_representation_lab import (
    REPRESENTATIONS, SEED_FAMILY_PATTERNS, transform_representation, _safe_ratio_bases, MODEL_PROFILES,
)
from scripts.verify_extreme_representation_lab import validate

def test_representation_catalog_is_unique():
    assert len(REPRESENTATIONS) == 8
    assert len(set(REPRESENTATIONS)) == 8
    assert len(SEED_FAMILY_PATTERNS) == 16
    assert MODEL_PROFILES == ("BALANCED", "ROBUST", "SMOOTH", "DEEP", "LOCAL", "REGULARIZED")

def test_representation_transforms_are_deterministic_and_finite():
    frame=pd.DataFrame({
        "home_adv":[1.,1.],"expected_env":[6.,7.],
        "h_gf_10":[3.,4.],"a_gf_10":[2.,3.],"d_gf_10":[1.,1.],
        "h_bat_avg_10":[.25,.27],"a_bat_avg_10":[.23,.25],"d_bat_avg_10":[.02,.02],
        "h_era":[3.,4.],"a_era":[4.,3.],"d_era":[-1.,1.],
        "starter_x_quality_proxy":[.5,-.2],
    })
    for mode in REPRESENTATIONS:
        x1,m1=transform_representation(frame,mode)
        x2,m2=transform_representation(frame,mode)
        assert list(x1.columns)==list(x2.columns)
        assert m1["feature_schema_hash"]==m2["feature_schema_hash"]
        assert np.isfinite(x1.to_numpy()).all()

def test_ratio_bases_are_name_gated():
    assert "gf_10" in _safe_ratio_bases(["h_gf_10","a_gf_10","h_elo","a_elo"])
    assert "elo" not in _safe_ratio_bases(["h_elo","a_elo"])

def _artifact():
    return {
      "status":"RESEARCH_ONLY","research_contract_id":"extreme-representation-v1","decision":"NO_AUTO_ADOPTION",
      "git_commit_sha":"0123456789abcdef0123456789abcdef01234567",
      "locked_holdout_rows":10,
      "stage_counts":{"stage_a_requested":128,"stage_a_successful":128,"stage_a_failed":0,"stage_b_requested":180,"stage_b_successful":180,"stage_b_failed":0,"stage_c_requested":24,"stage_c_successful":24,"stage_c_failed":0},
      "research_gate":{"holdout_locked_before_selection":True,"no_auto_adoption":True,"unexpected_execution_failures":0},
      "representation_catalog":{"modes":list(REPRESENTATIONS)},
      "winner":{"candidate_id":"x"},
      "locked_holdout":{"winner_only":True,"metrics":{"LogLoss":1.0,"Brier":.5,"Accuracy":.5,"ECE":.1,"rows":10}}
    }

def test_verifier_accepts_complete_artifact(tmp_path):
    p=tmp_path/"x.json"; p.write_text(json.dumps(_artifact()),encoding="utf-8")
    assert validate(str(p))["winner"]=="x"

def test_verifier_rejects_execution_failure(tmp_path):
    x=_artifact(); x["research_gate"]["unexpected_execution_failures"]=1
    p=tmp_path/"x.json"; p.write_text(json.dumps(x),encoding="utf-8")
    with pytest.raises(RuntimeError,match="unexpected execution failures"): validate(str(p))
