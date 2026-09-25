from pathlib import Path
import json

import pandas as pd
import numpy as np

import research.experience_ledger as exp


def _prediction(tmp: Path):
    p = {
        "execution_status": "EXECUTED",
        "target_date": "2026-09-26",
        "predictions": [{
            "game_id": "NPB-2026-09-26-1",
            "datetime_jst": "2026-09-26T14:00:00+09:00",
            "prediction_cutoff_utc": "2026-09-26T02:00:00+00:00",
            "prediction_generated_at": "2026-09-26T02:00:02+00:00",
            "home": "横浜DeNAベイスターズ",
            "away": "阪神タイガース",
            "home_starter": "尾形崇斗",
            "away_starter": "才木浩人",
            "regime": "s2_e2_m2",
            "score_regime": "s2_e2_m2",
            "model": "Production",
            "situation_tags": ["strength:neutral", "environment:high"],
            "home_win_pct": 60.0,
            "draw_pct": 5.0,
            "away_win_pct": 35.0,
            "low_pct": 70.0,
            "high_pct": 30.0,
            "lambda_home": 3.2,
            "lambda_away": 2.4,
            "shared_lambda": 0.0,
            "top4_exact_scores": [
                {"score": "3-2", "prob_pct": 7.0},
                {"score": "3-3", "prob_pct": 6.0},
                {"score": "4-2", "prob_pct": 5.0},
                {"score": "2-2", "prob_pct": 5.0},
            ],
            "classification_regime_model_weights": {
                "RandomForest": 0.35,
                "ExtraTrees": 0.30,
                "CatBoost": 0.20,
                "KNNAnalog": 0.10,
                "XGBoost": 0.05,
            },
            "score_regime_model_weights": {},
            "git_commit": "test",
        }],
    }
    path = tmp / "pred.json"
    path.write_text(json.dumps(p, ensure_ascii=False), encoding="utf-8")
    return path


def test_archive_preserves_each_prediction_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "EXPERIENCE", tmp_path / "experience")
    monkeypatch.setattr(exp, "PRED_DIR", tmp_path / "experience" / "predictions")
    monkeypatch.setattr(exp, "RESULT_DIR", tmp_path / "experience" / "official_results")
    result = exp.archive_production_output(_prediction(tmp_path), run_id="123")
    assert result["archived"] == 1
    path = tmp_path / "experience" / "predictions" / "2026-09-26.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    assert len(rows) == 1
    assert rows[0]["source_run_id"] == "123"
    assert rows[0]["prediction_id"]


def test_reconcile_computes_real_game_experience(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "EXPERIENCE", tmp_path / "experience")
    monkeypatch.setattr(exp, "PRED_DIR", tmp_path / "experience" / "predictions")
    monkeypatch.setattr(exp, "RESULT_DIR", tmp_path / "experience" / "official_results")
    monkeypatch.setattr(exp, "LEDGER_PATH", tmp_path / "experience" / "experience_ledger.csv")
    monkeypatch.setattr(exp, "LEDGER_JSONL", tmp_path / "experience" / "experience_ledger.jsonl")
    monkeypatch.setattr(exp, "SUMMARY_PATH", tmp_path / "experience" / "experience_summary.json")
    exp.archive_production_output(_prediction(tmp_path))

    exp._load_cached_results = lambda dates: pd.DataFrame([{
        "date": "2026-09-26",
        "home": "横浜DeNAベイスターズ",
        "away": "阪神タイガース",
        "home_score": 4,
        "away_score": 2,
        "source_url": "test://npb",
    }])

    summary = exp.reconcile()
    ledger = pd.read_csv(tmp_path / "experience" / "experience_ledger.csv")
    assert len(ledger) == 1
    assert ledger.loc[0, "actual_outcome"] == "HOME_WIN"
    assert int(ledger.loc[0, "outcome_correct"]) == 1
    assert int(ledger.loc[0, "low_high_correct"]) == 1
    assert int(ledger.loc[0, "top1_exact_hit"]) == 0
    assert int(ledger.loc[0, "top4_hit"]) == 1
    assert abs(float(ledger.loc[0, "classification_weight_RandomForest"]) - 0.35) < 1e-12
    assert summary["matched_rows_total"] == 1
    assert summary["outcome_accuracy"] == 1.0


def test_reconcile_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "EXPERIENCE", tmp_path / "experience")
    monkeypatch.setattr(exp, "PRED_DIR", tmp_path / "experience" / "predictions")
    monkeypatch.setattr(exp, "RESULT_DIR", tmp_path / "experience" / "official_results")
    monkeypatch.setattr(exp, "LEDGER_PATH", tmp_path / "experience" / "experience_ledger.csv")
    monkeypatch.setattr(exp, "LEDGER_JSONL", tmp_path / "experience" / "experience_ledger.jsonl")
    monkeypatch.setattr(exp, "SUMMARY_PATH", tmp_path / "experience" / "experience_summary.json")
    exp.archive_production_output(_prediction(tmp_path))
    exp._load_cached_results = lambda dates: pd.DataFrame([{
        "date": "2026-09-26",
        "home": "横浜DeNAベイスターズ",
        "away": "阪神タイガース",
        "home_score": 4,
        "away_score": 2,
        "source_url": "test://npb",
    }])
    exp.reconcile()
    exp.reconcile()
    ledger = pd.read_csv(tmp_path / "experience" / "experience_ledger.csv")
    assert len(ledger) == 1
