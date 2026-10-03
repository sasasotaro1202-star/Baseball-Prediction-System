import json
from pathlib import Path

import pandas as pd

import research.shadow_experience as shadow


def _prediction():
    return {
        "execution_status": "RESEARCH_SHADOW_EXECUTED",
        "target_date": "2026-10-03",
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
            }
        ],
    }


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


def test_shadow_reconcile_uses_latest_snapshot_per_game(tmp_path, monkeypatch):
    root = tmp_path / "research_shadow"
    pred_dir = root / "predictions"
    pred_dir.mkdir(parents=True)
    monkeypatch.setattr(shadow, "SHADOW_ROOT", root)
    monkeypatch.setattr(shadow, "PRED_DIR", pred_dir)
    monkeypatch.setattr(shadow, "LEDGER_PATH", root / "shadow_experience_ledger.csv")
    monkeypatch.setattr(shadow, "LEDGER_JSONL", root / "shadow_experience_ledger.jsonl")
    monkeypatch.setattr(shadow, "SUMMARY_PATH", root / "shadow_experience_summary.json")

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



def test_shadow_reconcile_records_horizon_breakdown(tmp_path, monkeypatch):
    root = tmp_path / "research_shadow"
    pred_dir = root / "predictions"
    pred_dir.mkdir(parents=True)
    monkeypatch.setattr(shadow, "SHADOW_ROOT", root)
    monkeypatch.setattr(shadow, "PRED_DIR", pred_dir)
    monkeypatch.setattr(shadow, "LEDGER_PATH", root / "shadow_experience_ledger.csv")
    monkeypatch.setattr(shadow, "LEDGER_JSONL", root / "shadow_experience_ledger.jsonl")
    monkeypatch.setattr(shadow, "SUMMARY_PATH", root / "shadow_experience_summary.json")

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
