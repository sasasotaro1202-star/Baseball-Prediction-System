from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from research.score_distribution_pattern_lab import (
    MEAN_SHRINKS,
    MODEL_MIXES,
    SHARED_SCALES,
    _mix_lambdas,
    _variant_id,
)
from scripts.verify_score_distribution_pattern_lab import validate


def test_score_variant_catalog_is_broad_and_deterministic():
    assert len(MODEL_MIXES) == 7
    assert len(SHARED_SCALES) == 6
    assert len(MEAN_SHRINKS) == 5
    ids = [
        _variant_id(mix, shared, shrink)
        for mix, _ in MODEL_MIXES
        for shared in SHARED_SCALES
        for shrink in MEAN_SHRINKS
    ]
    assert len(ids) == 210
    assert len(set(ids)) == 210


def test_mix_lambdas_is_deterministic_and_normalized():
    block = {
        "home_lambdas": np.array([[3.0, 4.0], [2.0, 5.0], [4.0, 3.0]]),
        "away_lambdas": np.array([[2.0, 3.0], [3.0, 2.0], [2.5, 2.5]]),
        "losses": np.array([1.0, 2.0, 4.0]),
    }
    h1, a1 = _mix_lambdas(block, "TOP3_INVLOSS")
    h2, a2 = _mix_lambdas(block, "TOP3_INVLOSS")
    assert np.allclose(h1, h2)
    assert np.allclose(a1, a2)
    hu, au = _mix_lambdas(block, "TOP3_UNIFORM")
    assert np.allclose(hu, np.mean(block["home_lambdas"], axis=0))
    assert np.allclose(au, np.mean(block["away_lambdas"], axis=0))


def _artifact():
    return {
        "schema_version": "baseball-score-distribution-pattern-lab-v1",
        "status": "RESEARCH_ONLY",
        "decision": "NO_AUTO_ADOPTION",
        "git_commit_sha": "0123456789abcdef0123456789abcdef01234567",
        "variant_catalog": {
            "model_mixes": [x[0] for x in MODEL_MIXES],
            "shared_scales": list(SHARED_SCALES),
            "mean_shrinks": list(MEAN_SHRINKS),
            "variant_count": 210,
        },
        "development_top": [{"variant_id": "x"}],
        "folds": {
            "screen": {"start_row": 500, "end_row": 600},
            "confirm": {"start_row": 600, "end_row": 700},
            "deep": {"start_row": 700, "end_row": 800},
        },
        "winner": {"variant_id": "x"},
        "locked_holdout_rows": 100,
        "locked_holdout": {
            "winner_only": True,
            "variant_id": "x",
            "metrics": {
                "ScoreMAE": 1.0,
                "HighLogLoss": 0.6,
                "HighBrier": 0.2,
                "HighAccuracy": 0.6,
                "Top4HitRate": 0.1,
                "ExactScoreRows": 98,
                "rows": 100,
            },
        },
        "research_gate": {
            "holdout_locked_before_selection": True,
            "no_auto_adoption": True,
            "unexpected_execution_failures": 0,
        },
    }


def test_verifier_accepts_complete_artifact(tmp_path):
    p = tmp_path / "score.json"
    p.write_text(json.dumps(_artifact()), encoding="utf-8")
    out = validate(str(p))
    assert out["winner"] == "x"


def test_verifier_rejects_holdout_winner_mismatch(tmp_path):
    obj = _artifact()
    obj["locked_holdout"]["variant_id"] = "other"
    p = tmp_path / "score.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(RuntimeError, match="holdout variant differs"):
        validate(str(p))
