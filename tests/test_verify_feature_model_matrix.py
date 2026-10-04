from __future__ import annotations

import json

import pytest

from scripts.verify_feature_model_matrix import validate
from research.feature_set_variants import SCREENING_VARIANTS


def _payload() -> dict:
    return {
        "status": "RESEARCH_ONLY",
        "selection_basis": "Development OOS only",
        "decision": "NO_AUTO_ADOPTION",
        "holdout_locked_before_selection": True,
        "matrix_size_requested": len(SCREENING_VARIANTS) * 5,
        "matrix_size_executed": len(SCREENING_VARIANTS) * 5,
        "variants_requested": list(SCREENING_VARIANTS),
        "half_lives_requested": [600, 900, 1800, 3600, 7200],
        "model_pools_requested": ["LINEAR_TREE"],
        "successful_configs": 150,
        "failed_configs": 15,
        "blocked_pit_context_configs": 15,
        "execution_failed_configs": 0,
        "locked_holdout_rows": 200,
        "winner": {"config_id": "BASELINE_TEAM_STATE|hl=1800|pool=LINEAR_TREE"},
        "locked_holdout": {
            "winner_only": True,
            "metrics": {"LogLoss": 1.0, "Brier": 0.6, "Accuracy": 0.55, "ECE": 0.04, "rows": 200},
        },
    }


def test_validate_accepts_reconciled_research_matrix(tmp_path):
    p = tmp_path / "matrix.json"
    p.write_text(json.dumps(_payload()), encoding="utf-8")
    result = validate(str(p))
    assert result["matrix_requested"] == len(SCREENING_VARIANTS) * 5
    assert result["execution_failed"] == 0


def test_validate_rejects_incomplete_matrix(tmp_path):
    obj = _payload()
    obj["matrix_size_executed"] -= 1
    p = tmp_path / "matrix.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(RuntimeError, match="execution incomplete"):
        validate(str(p))


def test_validate_rejects_execution_failure(tmp_path):
    obj = _payload()
    obj["execution_failed_configs"] = 1
    obj["failed_configs"] = obj["blocked_pit_context_configs"] + obj["execution_failed_configs"]
    p = tmp_path / "matrix.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(RuntimeError, match="failed execution"):
        validate(str(p))
