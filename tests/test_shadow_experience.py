import json
from pathlib import Path

import pandas as pd
import pytest

import research.shadow_experience as shadow


@pytest.fixture(autouse=True)
def _current_experience_epoch(monkeypatch):
    monkeypatch.setattr(
        shadow,
        "_experience_epoch",
        lambda: pd.Timestamp("2026-10-01T00:00:00+00:00"),
    )


def _prediction():
    return {
        "execution_status": "RESEARCH_SHADOW_EXECUTED",
        "target_date": "2026-10-03",
        "schema_version": "npb-production-v1",
        "feature_set_id": "feature-contract-v1:PIT_SAFE_CONTEXT_ACTIVE:testhash",
        "feature_schema_hash": "testhash",
        "feature_set_variant": "FULL_VALIDATED_ENSEMBLE",
        "feature_context_mode": "PIT_SAFE_CONTEXT_ACTIVE",
        "git_commit": "test-commit-vnext",
        "predictions": [
            {
                "game_id": "NPB-2026-10-03-1",
                "datetime_jst": "2026-10-03T18:00:00+09:00",
                "prediction_cutoff_utc": "2026-10-03T08:55:00+00:00",
                "prediction_generated_at": "2026-10-03T08:56:00+00:00",
                "starter_evidence_observed_at_utc": "2026-10-03T08:54:00+00:00",
                "starter_evidence_status": "official_announced",
                "pit_status": "PASS",
                "home": "東京ヤクルトスワローズ",
                "away": "読売ジャイアンツ",
                "competition": "npb_regular",
                "competition_stage": "regular_season",
                "season_type": "regular_season",
                "game_class": "official",
                "competition_key": "NPB:npb_regular:regular_season",
                "competition_classification_status": "classified",
                "home_starter": "A",
                "away_starter": "B",
                "home_win_pct": 55.0,
                "draw_pct": 5.0,
                "away_win_pct": 40.0,
                "low_pct": 60.0,
                "high_pct": 40.0,
                "lambda_home": 3.0,
                "lambda_away": 2.0,
                "top4_exact_scores": [
                    {"score": "3-2", "prob_pct": 10.0},
                    {"score": "2-2", "prob_pct": 9.0},
                    {"score": "3-1", "prob_pct": 8.0},
                    {"score": "2-1", "prob_pct": 7.0},
                ],
                "prediction_source": "RESEARCH_SHADOW_AUTO_60M",
                "prediction_schedule": "scheduled",
                "prediction_target_lead_minutes": 60.0,
                "model": "TestCurrentMethod",
                "method_signature": (
                    "npb-production-v1|feature-contract-v1:PIT_SAFE_CONTEXT_ACTIVE:testhash|"
                    "testhash|FULL_VALIDATED_ENSEMBLE|PIT_SAFE_CONTEXT_ACTIVE|models:TestCurrentMethod"
                ),
            }
        ],
    }


def test_archive_persists_method_identity(tmp_path, monkeypatch):
    root = tmp_path / "research_shadow"
    monkeypatch.setattr(shadow, "SHADOW_ROOT", root)
    monkeypatch.setattr(shadow, "PRED_DIR", root / "predictions")
    input_path = tmp_path / "shadow.json"
    input_path.write_text(json.dumps(_prediction(), ensure_ascii=False), encoding="utf-8")

    result = shadow.archive_shadow_output(input_path, run_id="r1")
    assert result["archived"] == 1
    row = json.loads((root / "predictions" / "2026-10-03.jsonl").read_text().splitlines()[0])
    assert row["method_signature"] == (
        "npb-production-v1|feature-contract-v1:PIT_SAFE_CONTEXT_ACTIVE:testhash|"
        "testhash|FULL_VALIDATED_ENSEMBLE|PIT_SAFE_CONTEXT_ACTIVE|models:TestCurrentMethod"
    )
    assert row["feature_schema_hash"] == "testhash"


def test_archive_shadow_output_is_separate_and_idempotent(tmp_path, monkeypatch):
    root = tmp_path / "research_shadow"
    monkeypatch.setattr(shadow, "SHADOW_ROOT", root)
    monkeypatch.setattr(shadow, "PRED_DIR", root / "predictions")
    input_path = tmp_path / "shadow.json"
    input_path.write_text(json.dumps(_prediction(), ensure_ascii=False), encoding="utf-8")

    first = shadow.archive_shadow_output(input_path, run_id="r1")
    second = shadow.archive_shadow_output(input_path, run_id="r1")

    assert first["archived"] == 1
    assert second["archived"] == 0
    path = root / "predictions" / "2026-10-03.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 1
    assert rows[0]["prediction_scope"] == "RESEARCH_SHADOW"
    assert rows[0]["production_eligible"] is False


def test_shadow_reconcile_preserves_earliest_experience_availability(tmp_path, monkeypatch):
    root = tmp_path / "research_shadow"
    pred_dir = root / "predictions"
    pred_dir.mkdir(parents=True)
    monkeypatch.setattr(shadow, "SHADOW_ROOT", root)
    monkeypatch.setattr(shadow, "PRED_DIR", pred_dir)
    monkeypatch.setattr(shadow, "LEDGER_PATH", root / "shadow_experience_ledger.csv")
    monkeypatch.setattr(shadow, "LEDGER_JSONL", root / "shadow_experience_ledger.jsonl")
    monkeypatch.setattr(shadow, "SUMMARY_PATH", root / "shadow_experience_summary.json")
    monkeypatch.setattr(shadow, "CURRENT_METHOD_SUMMARY_PATH", root / "current_method_performance.json")

    row = dict(_prediction()["predictions"][0])
    row["prediction_id"] = "availability-test"
    pred_dir.joinpath("2026-10-03.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\\n",
        encoding="utf-8",
    )
    root.joinpath("shadow_experience_ledger.jsonl").write_text(
        json.dumps({
            "game_id": row["game_id"],
            "experience_available_at_utc": "2026-10-03T12:00:00+00:00",
        }, ensure_ascii=False) + "\\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        shadow,
        "_load_cached_results",
        lambda dates: pd.DataFrame([{
            "date": "2026-10-03",
            "home": "東京ヤクルトスワローズ",
            "away": "読売ジャイアンツ",
            "home_score": 3,
            "away_score": 2,
            "source_url": "test://npb",
        }]),
    )

    summary = shadow.reconcile_shadow()
    assert summary["status"] == "UPDATED"
    ledger = pd.read_csv(root / "shadow_experience_ledger.csv")
    assert set(ledger["experience_available_at_utc"]) == {"2026-10-03T12:00:00+00:00"}
    retrieved = pd.to_datetime(ledger["official_result_retrieved_at_utc"], utc=True, errors="raise")
    available = pd.to_datetime(ledger["experience_available_at_utc"], utc=True, errors="raise")
    assert (retrieved > available).all()


def test_shadow_reconcile_uses_latest_snapshot_per_game(tmp_path, monkeypatch):
    root = tmp_path / "research_shadow"
    pred_dir = root / "predictions"
    pred_dir.mkdir(parents=True)
    monkeypatch.setattr(shadow, "SHADOW_ROOT", root)
    monkeypatch.setattr(shadow, "PRED_DIR", pred_dir)
    monkeypatch.setattr(shadow, "LEDGER_PATH", root / "shadow_experience_ledger.csv")
    monkeypatch.setattr(shadow, "LEDGER_JSONL", root / "shadow_experience_ledger.jsonl")
    monkeypatch.setattr(shadow, "SUMMARY_PATH", root / "shadow_experience_summary.json")
    monkeypatch.setattr(shadow, "CURRENT_METHOD_SUMMARY_PATH", root / "current_method_performance.json")

    early = _prediction()["predictions"][0]
    late = dict(early)
    late["prediction_id"] = "later"
    late["prediction_cutoff_utc"] = "2026-10-03T08:58:00+00:00"
    late["prediction_generated_at"] = "2026-10-03T08:59:00+00:00"
    payload = [
        early,
        late,
    ]
    pred_dir.joinpath("2026-10-03.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in payload),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        shadow,
        "_load_cached_results",
        lambda dates: pd.DataFrame([{
            "date": "2026-10-03",
            "home": "東京ヤクルトスワローズ",
            "away": "読売ジャイアンツ",
            "home_score": 3,
            "away_score": 2,
            "source_url": "test://npb",
        }]),
    )

    summary = shadow.reconcile_shadow()
    assert summary["status"] == "UPDATED"
    assert summary["matched_snapshots"] == 2
    assert summary["canonical_cases"] == 1
    assert summary["canonical_metrics"]["accuracy"] == 1.0
    assert summary["production_modified"] is False

    ledger = pd.read_csv(root / "shadow_experience_ledger.csv")
    assert set(["prediction_id", "game_id", "target", "experience_available_at_utc", "actual_outcome"]).issubset(ledger.columns)
    assert set(ledger["target"].astype(str)) == {"NPB"}
    assert ledger["experience_available_at_utc"].notna().all()
    pd.to_datetime(ledger["experience_available_at_utc"], utc=True, errors="raise")
    assert ledger["method_signature"].notna().all()



def test_shadow_reconcile_records_horizon_breakdown(tmp_path, monkeypatch):
    root = tmp_path / "research_shadow"
    pred_dir = root / "predictions"
    pred_dir.mkdir(parents=True)
    monkeypatch.setattr(shadow, "SHADOW_ROOT", root)
    monkeypatch.setattr(shadow, "PRED_DIR", pred_dir)
    monkeypatch.setattr(shadow, "LEDGER_PATH", root / "shadow_experience_ledger.csv")
    monkeypatch.setattr(shadow, "LEDGER_JSONL", root / "shadow_experience_ledger.jsonl")
    monkeypatch.setattr(shadow, "SUMMARY_PATH", root / "shadow_experience_summary.json")
    monkeypatch.setattr(shadow, "CURRENT_METHOD_SUMMARY_PATH", root / "current_method_performance.json")

    row = dict(_prediction()["predictions"][0])
    row["prediction_id"] = "horizon-test"
    row["prediction_cutoff_utc"] = "2026-10-03T04:00:00+00:00"
    row["prediction_generated_at"] = "2026-10-03T04:05:00+00:00"
    row["starter_evidence_observed_at_utc"] = "2026-10-03T03:59:00+00:00"
    pred_dir.joinpath("2026-10-03.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        shadow,
        "_load_cached_results",
        lambda dates: pd.DataFrame([{
            "date": "2026-10-03",
            "home": "東京ヤクルトスワローズ",
            "away": "読売ジャイアンツ",
            "home_score": 3,
            "away_score": 2,
            "source_url": "test://npb",
        }]),
    )

    summary = shadow.reconcile_shadow()
    assert summary["by_horizon"]["3_TO_6H"]["rows"] == 1
    assert summary["by_horizon"]["3_TO_6H"]["mean_actual_lead_minutes"] == 295.0



def test_shadow_reconcile_keeps_horizon_breakdown_case_level(tmp_path, monkeypatch):
    root = tmp_path / "research_shadow"
    pred_dir = root / "predictions"
    pred_dir.mkdir(parents=True)
    monkeypatch.setattr(shadow, "SHADOW_ROOT", root)
    monkeypatch.setattr(shadow, "PRED_DIR", pred_dir)
    monkeypatch.setattr(shadow, "LEDGER_PATH", root / "shadow_experience_ledger.csv")
    monkeypatch.setattr(shadow, "LEDGER_JSONL", root / "shadow_experience_ledger.jsonl")
    monkeypatch.setattr(shadow, "SUMMARY_PATH", root / "shadow_experience_summary.json")
    monkeypatch.setattr(shadow, "CURRENT_METHOD_SUMMARY_PATH", root / "current_method_performance.json")

    early = dict(_prediction()["predictions"][0])
    early["prediction_id"] = "early"
    early["prediction_cutoff_utc"] = "2026-10-03T04:00:00+00:00"
    early["prediction_generated_at"] = "2026-10-03T04:05:00+00:00"
    early["starter_evidence_observed_at_utc"] = "2026-10-03T03:59:00+00:00"

    late = dict(early)
    late["prediction_id"] = "late"
    late["prediction_cutoff_utc"] = "2026-10-03T08:24:00+00:00"
    late["prediction_generated_at"] = "2026-10-03T08:25:00+00:00"
    late["starter_evidence_observed_at_utc"] = "2026-10-03T08:23:00+00:00"

    pred_dir.joinpath("2026-10-03.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in (early, late)),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        shadow,
        "_load_cached_results",
        lambda dates: pd.DataFrame([{
            "date": "2026-10-03",
            "home": "東京ヤクルトスワローズ",
            "away": "読売ジャイアンツ",
            "home_score": 3,
            "away_score": 2,
            "source_url": "test://npb",
        }]),
    )

    summary = shadow.reconcile_shadow()
    assert summary["matched_snapshots"] == 2
    assert summary["canonical_cases"] == 1
    assert set(summary["by_horizon"]) == {"3_TO_6H", "30_TO_60M"}
    assert summary["by_horizon"]["3_TO_6H"]["rows"] == 1
    assert summary["by_horizon"]["30_TO_60M"]["rows"] == 1
    assert set(summary["canonical_by_horizon"]) == {"30_TO_60M"}
    assert summary["canonical_by_horizon"]["30_TO_60M"]["rows"] == 1


def test_shadow_reconcile_reports_current_method_separately(tmp_path, monkeypatch):
    root = tmp_path / "research_shadow"
    pred_dir = root / "predictions"
    pred_dir.mkdir(parents=True)
    monkeypatch.setattr(shadow, "SHADOW_ROOT", root)
    monkeypatch.setattr(shadow, "PRED_DIR", pred_dir)
    monkeypatch.setattr(shadow, "LEDGER_PATH", root / "shadow_experience_ledger.csv")
    monkeypatch.setattr(shadow, "LEDGER_JSONL", root / "shadow_experience_ledger.jsonl")
    monkeypatch.setattr(shadow, "SUMMARY_PATH", root / "shadow_experience_summary.json")
    monkeypatch.setattr(shadow, "CURRENT_METHOD_SUMMARY_PATH", root / "current_method_performance.json")

    base = _prediction()["predictions"][0]
    early = dict(base)
    early["prediction_id"] = "old-method"
    early["prediction_cutoff_utc"] = "2026-10-03T04:00:00+00:00"
    early["prediction_generated_at"] = "2026-10-03T04:01:00+00:00"
    early["method_signature"] = "old-method"
    early["feature_schema_hash"] = "oldhash"
    early["starter_evidence_observed_at_utc"] = "2026-10-03T03:59:00+00:00"

    late = dict(base)
    late["prediction_id"] = "current-method"
    late["prediction_cutoff_utc"] = "2026-10-03T08:00:00+00:00"
    late["prediction_generated_at"] = "2026-10-03T08:01:00+00:00"
    late["starter_evidence_observed_at_utc"] = "2026-10-03T07:59:00+00:00"
    late["method_signature"] = (
        "npb-production-v1|feature-contract-v1:PIT_SAFE_CONTEXT_ACTIVE:testhash|"
        "testhash|FULL_VALIDATED_ENSEMBLE|PIT_SAFE_CONTEXT_ACTIVE|models:TestCurrentMethod"
    )

    pred_dir.joinpath("2026-10-03.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in (early, late)),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        shadow,
        "_load_cached_results",
        lambda dates: pd.DataFrame([{
            "date": "2026-10-03",
            "home": "東京ヤクルトスワローズ",
            "away": "読売ジャイアンツ",
            "home_score": 3,
            "away_score": 2,
            "source_url": "test://npb",
        }]),
    )

    summary = shadow.reconcile_shadow()
    assert summary["status"] == "UPDATED"
    assert summary["current_method_signature"] == late["method_signature"]
    assert summary["current_method_performance"]["canonical_cases"] == 1
    current = json.loads((root / "current_method_performance.json").read_text(encoding="utf-8"))
    assert current["scope"] == "RESEARCH_SHADOW_CURRENT_METHOD"
    assert current["production_modified"] is False
    assert current["method_signature"] == summary["current_method_signature"]
    assert set(summary["by_method"]) == {"old-method", late["method_signature"]}


def test_shadow_reconcile_excludes_pre_epoch_predictions(tmp_path, monkeypatch):
    root = tmp_path / "research_shadow"
    pred_dir = root / "predictions"
    pred_dir.mkdir(parents=True)
    monkeypatch.setattr(shadow, "SHADOW_ROOT", root)
    monkeypatch.setattr(shadow, "PRED_DIR", pred_dir)
    monkeypatch.setattr(shadow, "LEDGER_PATH", root / "shadow_experience_ledger.csv")
    monkeypatch.setattr(shadow, "LEDGER_JSONL", root / "shadow_experience_ledger.jsonl")
    monkeypatch.setattr(shadow, "SUMMARY_PATH", root / "shadow_experience_summary.json")
    monkeypatch.setattr(shadow, "CURRENT_METHOD_SUMMARY_PATH", root / "current_method_performance.json")

    row = dict(_prediction()["predictions"][0])
    row["prediction_id"] = "pre-epoch"
    row["prediction_generated_at"] = "2026-09-30T08:56:00+00:00"
    row["prediction_cutoff_utc"] = "2026-09-30T08:55:00+00:00"
    row["starter_evidence_observed_at_utc"] = "2026-09-30T08:54:00+00:00"
    pred_dir.joinpath("2026-09-30.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = shadow.reconcile_shadow()
    assert summary["status"] == "NO_SHADOW_PREDICTIONS"
    assert summary["prediction_rows"] == 0
    assert not (root / "shadow_experience_ledger.csv").exists()


def test_archive_rejects_unknown_competition_identity(tmp_path, monkeypatch):
    root = tmp_path / "research_shadow"
    monkeypatch.setattr(shadow, "SHADOW_ROOT", root)
    monkeypatch.setattr(shadow, "PRED_DIR", root / "predictions")
    input_path = tmp_path / "shadow.json"
    payload = _prediction()
    payload["predictions"][0]["competition_classification_status"] = "unknown"
    payload["predictions"][0]["competition"] = "npb_unknown"
    payload["predictions"][0]["competition_stage"] = "unknown"
    payload["predictions"][0]["season_type"] = "unknown"
    payload["predictions"][0]["game_class"] = "unknown"
    payload["predictions"][0]["competition_key"] = "NPB:npb_unknown:unknown"
    input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="UNKNOWN|non-classified"):
        shadow.archive_shadow_output(input_path, run_id="r1")


def test_archive_does_not_create_empty_file_for_pre_epoch_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "research_shadow"
    monkeypatch.setattr(shadow, "SHADOW_ROOT", root)
    monkeypatch.setattr(shadow, "PRED_DIR", root / "predictions")
    input_path = tmp_path / "shadow.json"
    payload = _prediction()
    payload["predictions"][0]["prediction_generated_at"] = "2026-09-30T08:56:00+00:00"
    payload["predictions"][0]["prediction_cutoff_utc"] = "2026-09-30T08:55:00+00:00"
    payload["predictions"][0]["starter_evidence_observed_at_utc"] = "2026-09-30T08:54:00+00:00"
    input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    result = shadow.archive_shadow_output(input_path, run_id="r1")
    assert result == {"archived": 0, "skipped": 0, "updated": 0}
    assert not (root / "predictions").exists()
