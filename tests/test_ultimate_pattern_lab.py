from __future__ import annotations

import json

import pytest

from research.ultimate_pattern_lab import (
    CORE_HORIZONS,
    HALF_LIVES,
    OPTIONAL_FAMILIES,
    STAGE_POOLS,
    _family_pattern_catalog,
    _pattern_id,
    _staged_folds,
    select_pattern,
)
from scripts.verify_ultimate_pattern_lab import validate


def test_family_catalog_is_exhaustive_and_deterministic():
    a = _family_pattern_catalog()
    b = _family_pattern_catalog()
    assert len(a) == 256
    assert a == b
    assert len({_pattern_id(x) for x in a}) == 256


def test_stage_contracts_have_expected_breadth():
    assert len(OPTIONAL_FAMILIES) == 8
    assert CORE_HORIZONS == ("ALL", "SHORT", "LONG")
    assert HALF_LIVES == (600, 900, 1800, 3600, 7200)
    assert STAGE_POOLS == ("LINEAR_TREE", "BROAD_TREE", "DIVERSE")


def test_staged_folds_are_ordered_and_disjoint():
    folds = _staged_folds(1000)
    assert folds["screen"][1] <= folds["confirm"][0]
    assert folds["confirm"][1] <= folds["deep"][0]
    assert folds["deep"][1] <= 1000


def test_select_pattern_fails_closed_for_missing_family():
    class Tiny:
        columns = ["home_adv"]

        def __getitem__(self, key):
            raise AssertionError("selection should fail before frame slicing")

    with pytest.raises(ValueError, match="requires unavailable feature families"):
        select_pattern(Tiny(), frozenset({"weather"}), horizon="ALL")


def _artifact():
    return {
        "status": "RESEARCH_ONLY",
        "git_commit_sha": "0123456789abcdef0123456789abcdef01234567",
        "decision": "NO_AUTO_ADOPTION",
        "locked_holdout_rows": 100,
        "stage_counts": {
            "stage_a_requested": 256,
            "stage_a_successful": 256,
            "stage_a_failed": 0,
            "stage_b_requested": 360,
            "stage_b_successful": 360,
            "stage_b_failed": 0,
            "stage_c_requested": 4,
            "stage_c_successful": 4,
            "stage_c_failed": 0,
        },
        "research_gate": {
            "holdout_locked_before_selection": True,
            "no_auto_adoption": True,
            "unexpected_execution_failures": 0,
        },
        "feature_family_catalog": {
            "optional_families": list(OPTIONAL_FAMILIES),
            "catalog_size": 256,
        },
        "folds": {
            "screen": {"start_row": 500, "end_row": 600},
            "confirm": {"start_row": 600, "end_row": 700},
            "deep": {"start_row": 700, "end_row": 800},
        },
        "winner": {"candidate_id": "x"},
        "locked_holdout": {
            "winner_only": True,
            "metrics": {"LogLoss": 1.0, "Brier": 0.5, "Accuracy": 0.5, "ECE": 0.1, "rows": 100},
        },
    }


def test_verifier_accepts_reconciled_artifact(tmp_path):
    p = tmp_path / "artifact.json"
    p.write_text(json.dumps(_artifact()), encoding="utf-8")
    out = validate(str(p))
    assert out["winner"] == "x"


def test_verifier_rejects_execution_failure(tmp_path):
    obj = _artifact()
    obj["research_gate"]["unexpected_execution_failures"] = 1
    p = tmp_path / "artifact.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected execution failures"):
        validate(str(p))
