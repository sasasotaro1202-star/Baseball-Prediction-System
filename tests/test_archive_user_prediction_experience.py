import json

import pytest

import research.shadow_experience as shadow
from scripts.archive_user_prediction_experience import archive_result_file


def _outer(classification="classified"):
    row = {
        "game_id": "NPB-2026-10-03-1",
        "datetime_jst": "2026-10-03T18:00:00+09:00",
        "prediction_cutoff_utc": "2026-10-03T08:55:00+00:00",
        "prediction_generated_at": "2026-10-03T08:56:00+00:00",
        "starter_evidence_observed_at_utc": "2026-10-03T08:54:00+00:00",
        "starter_evidence_status": "official_announced",
        "pit_status": "PASS",
        "home": "東京ヤクルトスワローズ",
        "away": "読売ジャイアンツ",
        "competition": "npb_regular" if classification == "classified" else "npb_unknown",
        "competition_stage": "regular_season" if classification == "classified" else "unknown",
        "season_type": "regular_season" if classification == "classified" else "unknown",
        "game_class": "official" if classification == "classified" else "unknown",
        "competition_key": "NPB:npb_regular:regular_season" if classification == "classified" else "NPB:npb_unknown:unknown",
        "competition_classification_status": classification,
        "home_starter": "A",
        "away_starter": "B",
        "home_win_pct": 55.0,
        "draw_pct": 5.0,
        "away_win_pct": 40.0,
        "low_pct": 60.0,
        "high_pct": 40.0,
        "lambda_home": 3.0,
        "lambda_away": 2.0,
        "top4_exact_scores": [{"score": "3-2", "prob_pct": 10.0}],
        "model": "TestCurrentMethod",
    }
    return {
        "schema_version": "baseball-user-prediction-result-v1",
        "request_id": "req-test",
        "competition_id": "NPB",
        "generation_lane": "VALIDATED_RESEARCH_SHADOW",
        "prediction_output": {
            "execution_status": "RESEARCH_SHADOW_EXECUTED",
            "scope": "RESEARCH_SHADOW",
            "production_eligibility": False,
            "target_date": "2026-10-03",
            "schema_version": "npb-production-v1",
            "feature_set_id": "feature-contract-v1:PIT_SAFE_CONTEXT_ACTIVE:testhash",
            "feature_schema_hash": "testhash",
            "feature_set_variant": "FULL_VALIDATED_ENSEMBLE",
            "feature_context_mode": "PIT_SAFE_CONTEXT_ACTIVE",
            "feature_manifest_version": "feature-contract-v1",
            "git_commit": "test-commit-vnext",
            "predictions": [row],
        },
    }


def test_archives_valid_user_result_and_persists_request_id(tmp_path, monkeypatch):
    root = tmp_path / "research_shadow"
    monkeypatch.setattr(shadow, "SHADOW_ROOT", root)
    monkeypatch.setattr(shadow, "PRED_DIR", root / "predictions")
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(_outer(), ensure_ascii=False), encoding="utf-8")

    result = archive_result_file(result_path, run_id="run-1")
    assert result["status"] == "ARCHIVED"
    assert result["archived"] == 1
    row = json.loads((root / "predictions" / "2026-10-03.jsonl").read_text().splitlines()[0])
    assert row["request_id"] == "req-test"
    assert row["source_run_id"] == "run-1"
    assert row["production_eligible"] is False


def test_unknown_identity_is_quarantined_and_never_archived(tmp_path, monkeypatch):
    root = tmp_path / "research_shadow"
    monkeypatch.setattr(shadow, "SHADOW_ROOT", root)
    monkeypatch.setattr(shadow, "PRED_DIR", root / "predictions")
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(_outer("unknown"), ensure_ascii=False),
        encoding="utf-8",
    )

    result = archive_result_file(result_path, run_id="run-1")
    assert result["status"] == "QUARANTINED"
    assert result["reason"] == "UNKNOWN_IDENTITY"
    assert not (root / "predictions").exists()
