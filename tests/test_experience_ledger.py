from pathlib import Path
import json

import pandas as pd
import numpy as np
import pytest

import research.experience_ledger as exp


@pytest.fixture(autouse=True)
def _current_experience_epoch(monkeypatch):
    # Keep historical unit fixtures inside a deliberately old test epoch.
    monkeypatch.setattr(
        exp,
        "_experience_epoch",
        lambda: pd.Timestamp("2026-09-01T00:00:00+00:00"),
    )


def _prediction(tmp: Path):
    p = {
        "execution_status": "EXECUTED",
        "target_date": "2026-09-26",
        "predictions": [{
            "game_id": "NPB-2026-09-26-1",
            "datetime_jst": "2026-09-26T14:00:00+09:00",
            "prediction_cutoff_utc": "2026-09-26T02:00:00+00:00",
            "prediction_generated_at": "2026-09-26T02:00:02+00:00",
            "starter_evidence_observed_at_utc": "2026-09-26T01:59:30+00:00",
            "starter_evidence_status": "official_announced",
            "pit_status": "PASS",
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


def test_exact_score_metrics_are_unavailable_when_all_score_inputs_missing():
    from research.experience_ledger import _prediction_target_metrics

    frame = pd.DataFrame([{
        "home_win_pct": 60.0,
        "draw_pct": 5.0,
        "away_win_pct": 35.0,
        "actual_outcome": "HOME_WIN",
        "high_pct": 30.0,
        "low_pct": 70.0,
        "low_high_actual": 0,
        "top1_exact_hit": np.nan,
        "top4_hit": np.nan,
        "score_mae": np.nan,
    }])
    result = _prediction_target_metrics(frame)
    exact = result["exact_score"]
    assert exact["status"] == "UNAVAILABLE"
    assert exact["top1_evaluable_rows"] == 0
    assert exact["top4_evaluable_rows"] == 0
    assert exact["score_evaluable_rows"] == 0
    assert exact["top1_exact_hit_rate"] is None
    assert exact["top4_exact_hit_rate"] is None
    assert exact["score_mae"] is None


def test_reconcile_preserves_prediction_time_competition_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "EXPERIENCE", tmp_path / "experience")
    monkeypatch.setattr(exp, "PRED_DIR", tmp_path / "experience" / "predictions")
    monkeypatch.setattr(exp, "RESULT_DIR", tmp_path / "experience" / "official_results")
    monkeypatch.setattr(exp, "LEDGER_PATH", tmp_path / "experience" / "experience_ledger.csv")
    monkeypatch.setattr(exp, "LEDGER_JSONL", tmp_path / "experience" / "experience_ledger.jsonl")
    monkeypatch.setattr(exp, "SUMMARY_PATH", tmp_path / "experience" / "experience_summary.json")

    prediction = _prediction(tmp_path)
    payload = json.loads(prediction.read_text(encoding="utf-8"))
    payload["predictions"][0].update({
        "competition": "npb_regular",
        "competition_stage": "regular_season",
        "season_type": "regular_season",
        "game_class": "official",
        "competition_key": "NPB:npb_regular:regular_season",
        "competition_classification_status": "classified",
        "competition_metadata_source": "https://npb.jp/games/2026/schedule_09_detail.html",
        "competition_metadata_source_field": "npb_daily_schedule_heading",
        "competition_metadata_source_value": "公式戦",
        "game_type": "公式戦",
        "series_description": "",
    })
    prediction.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    exp.archive_production_output(prediction)
    monkeypatch.setattr(exp, "_load_cached_results", lambda dates, **kwargs: pd.DataFrame([{
        "date": "2026-09-26",
        "home": "横浜DeNAベイスターズ",
        "away": "阪神タイガース",
        "home_score": 4,
        "away_score": 2,
        "source_url": "test://npb",
    }]))

    exp.reconcile()
    ledger = pd.read_csv(tmp_path / "experience" / "experience_ledger.csv")
    row = ledger.loc[0]
    assert row["competition"] == "npb_regular"
    assert row["competition_stage"] == "regular_season"
    assert row["season_type"] == "regular_season"
    assert row["game_class"] == "official"
    assert row["competition_key"] == "NPB:npb_regular:regular_season"
    assert row["competition_classification_status"] == "classified"
    assert row["competition_metadata_source_field"] == "npb_daily_schedule_heading"
    assert row["competition_metadata_source_value"] == "公式戦"


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


def test_empty_result_cache_is_refreshed(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "RESULT_DIR", tmp_path / "experience" / "official_results")
    active_now = pd.Timestamp("2026-10-03T12:00:00+00:00")
    result_dir = tmp_path / "experience" / "official_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    cache = result_dir / "2026-10.csv"
    cache.write_text(
        "date,home,away,home_score,away_score,source_url\n",
        encoding="utf-8",
    )

    calls = []

    def fake_fetch(year, month):
        calls.append((year, month))
        return [{
            "date": "2026-10-02",
            "home": "東京ヤクルトスワローズ",
            "away": "読売ジャイアンツ",
            "home_score": 5,
            "away_score": 1,
            "source_url": "test://npb",
        }]

    monkeypatch.setattr(exp, "_fetch_month", fake_fetch)
    loaded = exp._load_cached_results(
        [pd.Timestamp("2026-10-02T18:00:00+09:00")],
        now=active_now,
    )

    assert calls == [(2026, 10)]
    assert len(loaded) == 1
    assert int(loaded.loc[0, "home_score"]) == 5
    assert len(pd.read_csv(cache)) == 1


def test_legacy_scheduled_cutoff_survives_dataframe_materialization(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "PRED_DIR", tmp_path / "experience" / "predictions")
    pred_dir = tmp_path / "experience" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    legacy = {
        "game_id": "NPB-2026-10-01-legacy",
        "datetime_jst": "2026-10-01T18:00:00+09:00",
        "prediction_cutoff_utc": "2026-10-01T08:30:00+00:00",
        "prediction_generated_at": "2026-10-01T03:52:26+00:00",
        "starter_evidence_observed_at_utc": "2026-10-01T03:49:31+00:00",
        "pit_status": "PASS",
        "home": "阪神タイガース",
        "away": "読売ジャイアンツ",
        "home_win_pct": 44.3064,
        "draw_pct": 2.6478,
        "away_win_pct": 53.0458,
        "low_pct": 52.4146,
        "high_pct": 47.5854,
        "lambda_home": 3.1,
        "lambda_away": 3.4,
    }
    modern = {
        **legacy,
        "game_id": "NPB-2026-10-01-modern",
        "prediction_cutoff_utc": "2026-10-01T03:45:00+00:00",
        "prediction_generated_at": "2026-10-01T03:52:26+00:00",
        "starter_evidence_observed_at_utc": "2026-10-01T03:44:31+00:00",
        "lead_minutes_at_generation": 852.0,
        "prediction_deadline_utc": "2026-10-01T08:30:00+00:00",
        "preferred_prediction_cutoff_utc": "2026-10-01T08:30:00+00:00",
    }
    (pred_dir / "2026-10-01.jsonl").write_text(
        json.dumps(legacy, ensure_ascii=False) + "\n"
        + json.dumps(modern, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    loaded = exp._load_predictions()
    assert set(loaded["game_id"]) == {"NPB-2026-10-01-modern"}

def test_current_result_cache_is_refreshed_even_when_nonempty(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "RESULT_DIR", tmp_path / "experience" / "official_results")
    active_now = pd.Timestamp("2026-10-03T12:00:00+00:00")
    result_dir = tmp_path / "experience" / "official_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    cache = result_dir / "2026-10.csv"
    cache.write_text(
        "date,home,away,home_score,away_score,source_url\n"
        "2026-10-01,東京ヤクルトスワローズ,阪神タイガース,1,2,test://old\n",
        encoding="utf-8",
    )

    calls = []
    def fake_fetch(year, month):
        calls.append((year, month))
        return [{
            "date": "2026-10-02",
            "home": "東京ヤクルトスワローズ",
            "away": "読売ジャイアンツ",
            "home_score": 5,
            "away_score": 1,
            "source_url": "test://new",
        }]

    monkeypatch.setattr(exp, "_fetch_month", fake_fetch)
    loaded = exp._load_cached_results(
        [pd.Timestamp("2026-10-02T18:00:00+09:00")],
        now=active_now,
    )
    assert calls == [(2026, 10)]
    assert len(loaded) == 1
    assert int(loaded.loc[0, "home_score"]) == 5
    assert loaded.loc[0, "source_url"] == "test://new"


def test_multiclass_ece_uses_confidence_vs_accuracy():
    probabilities = np.array([
        [0.60, 0.05, 0.35],
        [0.20, 0.65, 0.15],
    ])
    y_true = np.array([0, 1])
    assert exp._multiclass_ece(probabilities, y_true, bins=10) == 0.375


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


def test_reconcile_top_level_score_metrics_are_unavailable_when_inputs_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "EXPERIENCE", tmp_path / "experience")
    monkeypatch.setattr(exp, "PRED_DIR", tmp_path / "experience" / "predictions")
    monkeypatch.setattr(exp, "RESULT_DIR", tmp_path / "experience" / "official_results")
    monkeypatch.setattr(exp, "LEDGER_PATH", tmp_path / "experience" / "experience_ledger.csv")
    monkeypatch.setattr(exp, "LEDGER_JSONL", tmp_path / "experience" / "experience_ledger.jsonl")
    monkeypatch.setattr(exp, "SUMMARY_PATH", tmp_path / "experience" / "experience_summary.json")

    prediction_path = _prediction(tmp_path)
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    pred = payload["predictions"][0]
    pred["lambda_home"] = None
    pred["lambda_away"] = None
    pred["top4_exact_scores"] = []
    prediction_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    exp.archive_production_output(prediction_path)
    exp._load_cached_results = lambda dates: pd.DataFrame([{
        "date": "2026-09-26",
        "home": "横浜DeNAベイスターズ",
        "away": "阪神タイガース",
        "home_score": 4,
        "away_score": 2,
        "source_url": "test://npb",
    }])

    summary = exp.reconcile()
    assert summary["score_mae"] is None
    assert summary["top1_exact_hit_rate"] is None
    assert summary["top4_exact_hit_rate"] is None
    assert summary["score_evaluable_rows"] == 0
    assert summary["top1_evaluable_rows"] == 0
    assert summary["top4_evaluable_rows"] == 0



def test_reconcile_preserves_first_experience_availability_timestamp(tmp_path, monkeypatch):
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

    clock = iter([
        "2026-09-26T05:00:00+00:00",
        "2026-09-26T05:00:01+00:00",
        "2026-09-26T06:00:00+00:00",
        "2026-09-26T06:00:01+00:00",
    ])
    monkeypatch.setattr(exp, "_utc_now", lambda: next(clock))

    exp.reconcile()
    first = pd.read_csv(tmp_path / "experience" / "experience_ledger.csv")
    assert first.loc[0, "experience_available_at_utc"] == "2026-09-26T05:00:00+00:00"

    exp.reconcile()
    second = pd.read_csv(tmp_path / "experience" / "experience_ledger.csv")
    assert second.loc[0, "experience_available_at_utc"] == "2026-09-26T05:00:00+00:00"

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


def test_archive_and_reconcile_expose_target_level_metrics(tmp_path, monkeypatch):
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
    assert ledger.loc[0, "target"] == "NPB"
    assert ledger.loc[0, "competition_id"] == "NPB"
    assert set(summary["by_target"]) == {"NPB"}
    assert summary["by_target"]["NPB"]["rows"] == 1
    assert summary["by_target"]["NPB"]["accuracy"] == 1.0


def test_reconcile_reports_independent_prediction_target_metrics(tmp_path, monkeypatch):
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
    targets = summary["by_prediction_target"]
    assert set(targets) == {"win_3way", "low_high", "exact_score"}
    assert targets["win_3way"]["rows"] == 1
    assert targets["low_high"]["rows"] == 1
    assert targets["exact_score"]["rows"] == 1
    assert targets["win_3way"]["accuracy"] == 1.0
    assert abs(targets["win_3way"]["logloss"] + np.log(0.60)) < 1e-12
    # Multiclass Brier sums squared probability errors over all three outcome classes.
    assert abs(targets["win_3way"]["brier"] - 0.285) < 1e-12
    assert abs(targets["win_3way"]["ece"] - 0.40) < 1e-12
    assert abs(targets["low_high"]["logloss"] + np.log(0.70)) < 1e-12
    assert abs(targets["low_high"]["brier"] - 0.09) < 1e-12
    assert abs(targets["low_high"]["ece"] - 0.30) < 1e-12
    assert targets["win_3way"]["selective_0.60"]["n"] == 1
    assert targets["win_3way"]["selective_0.60"]["accuracy"] == 1.0
    assert targets["win_3way"]["selective_0.90"]["n"] == 0
    # The canonical ledger stores outcome probabilities on the normalized [0,1] scale.
    # Headline, target-level, and rolling ECE must therefore all use that same scale.
    assert abs(summary["ece"] - targets["win_3way"]["ece"]) < 1e-12
    assert abs(summary["by_target"]["NPB"]["ece"] - targets["win_3way"]["ece"]) < 1e-12
    assert abs(summary["rolling"]["30"]["ece"] - targets["win_3way"]["ece"]) < 1e-12


def test_prediction_target_metrics_accept_normalized_snapshot_scale():
    frame = pd.DataFrame([{
        "home_win_pct": 0.60,
        "draw_pct": 0.05,
        "away_win_pct": 0.35,
        "actual_outcome": "HOME_WIN",
        "high_pct": 0.30,
        "low_high_actual": 0,
        "top1_exact_hit": 0,
        "top4_hit": 1,
        "score_mae": 1.0,
    }])
    targets = exp._prediction_target_metrics(frame)
    assert abs(targets["win_3way"]["logloss"] + np.log(0.60)) < 1e-12
    assert abs(targets["win_3way"]["brier"] - 0.285) < 1e-12
    assert abs(targets["low_high"]["logloss"] + np.log(0.70)) < 1e-12
    assert abs(targets["low_high"]["brier"] - 0.09) < 1e-12
    assert abs(targets["win_3way"]["ece"] - 0.40) < 1e-12
    assert abs(targets["low_high"]["ece"] - 0.30) < 1e-12
    assert "logloss" in targets["win_3way"] and "ece" in targets["win_3way"]
    assert "logloss" in targets["low_high"] and "brier" in targets["low_high"] and "ece" in targets["low_high"]
    assert "top1_exact_hit_rate" in targets["exact_score"]
    assert "top4_exact_hit_rate" in targets["exact_score"]
    assert "score_mae" in targets["exact_score"]



def test_timing_30m_metrics_distinguishes_late_snapshot():
    frame = pd.DataFrame([
        {
            "datetime_jst": "2026-10-01T18:00:00+09:00",
            "prediction_generated_at": "2026-10-01T08:29:00+00:00",
            "prediction_cutoff_utc": "2026-10-01T08:30:00+00:00",
        },
        {
            "datetime_jst": "2026-10-01T18:00:00+09:00",
            "prediction_generated_at": "2026-10-01T08:31:00+00:00",
            "prediction_cutoff_utc": "2026-10-01T08:30:00+00:00",
        },
    ])
    timing = exp._timing_30m_metrics(frame)
    assert timing["status"] == "MEASURED"
    assert timing["eligible_rows"] == 2
    assert timing["on_time_rows"] == 1
    assert timing["late_rows"] == 1
    assert timing["compliance_rate"] == 0.5
    assert timing["scheduled_cutoff_below_30m_rows"] == 0



def test_load_predictions_rejects_invalid_pit_timing(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "PRED_DIR", tmp_path / "predictions")
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True)

    valid = json.loads(_prediction(tmp_path).read_text(encoding="utf-8"))["predictions"][0]

    bad_generated = dict(valid)
    bad_generated["prediction_id"] = "bad-generated"
    bad_generated["prediction_generated_at"] = "2026-09-26T05:00:00+00:00"

    bad_observed = dict(valid)
    bad_observed["prediction_id"] = "bad-observed"
    bad_observed["starter_evidence_observed_at_utc"] = "2026-09-26T02:00:30+00:00"

    path = pred_dir / "2026-09-26.jsonl"
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in (bad_generated, bad_observed)
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prediction snapshot"):
        exp._load_predictions()


def test_archive_rejects_non_normalized_probabilities(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "EXPERIENCE", tmp_path / "experience")
    monkeypatch.setattr(exp, "PRED_DIR", tmp_path / "experience" / "predictions")

    pred_path = _prediction(tmp_path)
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    payload["predictions"][0]["away_win_pct"] = 30.0
    pred_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="valid normalized distribution"):
        exp.archive_production_output(pred_path)

def test_archive_tracks_only_strictly_prior_prediction_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "EXPERIENCE", tmp_path / "experience")
    monkeypatch.setattr(exp, "PRED_DIR", tmp_path / "experience" / "predictions")
    monkeypatch.setattr(exp, "RESULT_DIR", tmp_path / "experience" / "official_results")

    base = json.loads(_prediction(tmp_path).read_text(encoding="utf-8"))
    first = base["predictions"][0]
    first["git_commit"] = "first"
    first["prediction_cutoff_utc"] = "2026-09-26T02:00:00+00:00"
    first["prediction_generated_at"] = "2026-09-26T02:00:02+00:00"
    first["starter_evidence_observed_at_utc"] = "2026-09-26T01:59:30+00:00"

    second = dict(first)
    second["git_commit"] = "second"
    second["prediction_cutoff_utc"] = "2026-09-26T02:30:00+00:00"
    second["prediction_generated_at"] = "2026-09-26T02:30:02+00:00"
    second["starter_evidence_observed_at_utc"] = "2026-09-26T02:29:30+00:00"
    second["home_win_pct"] = 50.0
    second["draw_pct"] = 10.0
    second["away_win_pct"] = 40.0

    same_cutoff = dict(second)
    same_cutoff["prediction_id"] = "ignored-input-id"
    same_cutoff["home_win_pct"] = 51.0
    same_cutoff["draw_pct"] = 9.0

    for i, row in enumerate((first,)):
        payload = {"execution_status": "EXECUTED", "target_date": "2026-09-26", "predictions": [row]}
        path = tmp_path / f"pred-{i}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        exp.archive_production_output(path)

    payload = {"execution_status": "EXECUTED", "target_date": "2026-09-26", "predictions": [second]}
    second_path = tmp_path / "pred-second.json"
    second_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    exp.archive_production_output(second_path)

    rows = [
        json.loads(line)
        for line in (tmp_path / "experience" / "predictions" / "2026-09-26.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_cutoff = {row["prediction_cutoff_utc"]: row for row in rows}
    revised = by_cutoff["2026-09-26T02:30:00+00:00"]
    assert revised["revision_status"] == "REVISED"
    assert revised["revision_outcome_changed"] is False
    assert revised["revision_previous_prediction_id"] == by_cutoff["2026-09-26T02:00:00+00:00"]["prediction_id"]
    assert revised["revision_l1_pct_points"] == pytest.approx(20.0)
    assert revised["revision_max_abs_pct_points"] == pytest.approx(10.0)


def test_revision_metadata_never_uses_future_snapshot():
    record = {
        "game_id": "g1",
        "prediction_cutoff_utc": "2026-09-26T03:00:00+00:00",
        "home_win_pct": 50.0,
        "draw_pct": 10.0,
        "away_win_pct": 40.0,
    }
    existing = {
        "future": {
            "game_id": "g1",
            "prediction_id": "future",
            "prediction_cutoff_utc": "2026-09-26T04:00:00+00:00",
            "home_win_pct": 10.0,
            "draw_pct": 10.0,
            "away_win_pct": 80.0,
        }
    }
    meta = exp._revision_metadata(record, existing)
    assert meta["revision_status"] == "INITIAL"
    assert meta["revision_previous_prediction_id"] is None


def test_revision_metrics_reports_probability_motion():
    frame = pd.DataFrame([
        {
            "revision_status": "INITIAL",
            "revision_l1_pct_points": np.nan,
            "revision_max_abs_pct_points": np.nan,
            "revision_outcome_changed": False,
        },
        {
            "revision_status": "REVISED",
            "revision_l1_pct_points": 20.0,
            "revision_max_abs_pct_points": 10.0,
            "revision_outcome_changed": True,
        },
        {
            "revision_status": "REVISED",
            "revision_l1_pct_points": 8.0,
            "revision_max_abs_pct_points": 4.0,
            "revision_outcome_changed": False,
        },
    ])
    metrics = exp._revision_metrics(frame)
    assert metrics["status"] == "MEASURED"
    assert metrics["eligible_rows"] == 3
    assert metrics["revised_rows"] == 2
    assert metrics["revision_rate"] == pytest.approx(2 / 3)
    assert metrics["outcome_reversal_rows"] == 1
    assert metrics["outcome_reversal_rate"] == pytest.approx(0.5)
    assert metrics["mean_l1_pct_points"] == pytest.approx(14.0)
    assert metrics["median_l1_pct_points"] == pytest.approx(14.0)
    assert metrics["maximum_l1_pct_points"] == pytest.approx(20.0)
    assert metrics["maximum_single_class_change_pct_points"] == pytest.approx(10.0)


def test_load_predictions_quarantines_legacy_scheduled_cutoff(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "PRED_DIR", tmp_path / "predictions")
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()

    valid = json.loads(_prediction(tmp_path).read_text(encoding="utf-8"))["predictions"][0]
    valid["prediction_generated_at"] = "2026-09-26T01:52:00+00:00"
    valid["prediction_cutoff_utc"] = "2026-09-26T02:00:00+00:00"

    path = pred_dir / "2026-09-26.jsonl"
    path.write_text(json.dumps(valid, ensure_ascii=False) + "\n", encoding="utf-8")

    loaded = exp._load_predictions()
    assert loaded.empty


def test_revision_metadata_ignores_quarantined_legacy_snapshot():
    record = {
        "game_id": "g1",
        "prediction_cutoff_utc": "2026-09-26T03:00:00+00:00",
        "home_win_pct": 50.0,
        "draw_pct": 10.0,
        "away_win_pct": 40.0,
    }
    existing = {
        "legacy": {
            "game_id": "g1",
            "prediction_id": "legacy",
            "prediction_cutoff_utc": "2026-09-26T02:00:00+00:00",
            "prediction_generated_at": "2026-09-26T01:52:00+00:00",
            "home_win_pct": 10.0,
            "draw_pct": 10.0,
            "away_win_pct": 80.0,
        }
    }
    meta = exp._revision_metadata(record, existing)
    assert meta["revision_status"] == "INITIAL"
    assert meta["revision_previous_prediction_id"] is None



def test_horizon_bucket_is_deterministic_and_outcome_free():
    assert exp._horizon_bucket(-1) == "UNKNOWN"
    assert exp._horizon_bucket("bad") == "UNKNOWN"
    assert exp._horizon_bucket(29.999) == "LT_30M"
    assert exp._horizon_bucket(30.0) == "30_TO_60M"
    assert exp._horizon_bucket(59.999) == "30_TO_60M"
    assert exp._horizon_bucket(60.0) == "1_TO_3H"
    assert exp._horizon_bucket(180.0) == "3_TO_6H"
    assert exp._horizon_bucket(360.0) == "GE_6H"


def test_reconcile_records_realized_prediction_horizon(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "EXPERIENCE", tmp_path / "experience")
    monkeypatch.setattr(exp, "PRED_DIR", tmp_path / "experience" / "predictions")
    monkeypatch.setattr(exp, "RESULT_DIR", tmp_path / "experience" / "official_results")
    monkeypatch.setattr(exp, "LEDGER_PATH", tmp_path / "experience" / "experience_ledger.csv")
    monkeypatch.setattr(exp, "LEDGER_JSONL", tmp_path / "experience" / "experience_ledger.jsonl")
    monkeypatch.setattr(exp, "SUMMARY_PATH", tmp_path / "experience" / "experience_summary.json")

    pred_path = _prediction(tmp_path)
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    payload["predictions"][0]["prediction_cutoff_utc"] = "2026-09-26T00:00:00+00:00"
    payload["predictions"][0]["prediction_generated_at"] = "2026-09-26T00:05:00+00:00"
    payload["predictions"][0]["starter_evidence_observed_at_utc"] = "2026-09-25T23:59:30+00:00"
    pred_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    exp.archive_production_output(pred_path)

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
    assert ledger.loc[0, "prediction_horizon"] == "3_TO_6H"
    assert summary["by_horizon"]["3_TO_6H"]["rows"] == 1
    assert summary["by_horizon"]["3_TO_6H"]["mean_actual_lead_minutes"] == pytest.approx(4 * 60 + 55)



def test_archive_quarantines_pre_epoch_prediction(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "EXPERIENCE", tmp_path / "experience")
    monkeypatch.setattr(exp, "PRED_DIR", tmp_path / "experience" / "predictions")

    payload = json.loads(_prediction(tmp_path).read_text(encoding="utf-8"))
    row = payload["predictions"][0]
    row["prediction_generated_at"] = "2026-08-31T23:59:00+00:00"
    row["prediction_cutoff_utc"] = "2026-08-31T23:58:00+00:00"
    row["starter_evidence_observed_at_utc"] = "2026-08-31T23:57:30+00:00"
    payload["target_date"] = "2026-08-31"
    path = tmp_path / "pre-epoch.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    result = exp.archive_production_output(path, run_id="pre-epoch-run")
    assert result == {"archived": 0, "skipped": 0, "updated": 0}
    assert not (tmp_path / "experience" / "predictions" / "2026-08-31.jsonl").exists()


def test_load_predictions_filters_pre_epoch_history(tmp_path, monkeypatch):
    pred_dir = tmp_path / "experience" / "predictions"
    pred_dir.mkdir(parents=True)
    row = json.loads(_prediction(tmp_path).read_text(encoding="utf-8"))["predictions"][0]
    row["prediction_generated_at"] = "2026-08-31T23:59:00+00:00"
    row["prediction_cutoff_utc"] = "2026-08-31T23:58:00+00:00"
    row["starter_evidence_observed_at_utc"] = "2026-08-31T23:57:30+00:00"
    pred_dir.joinpath("2026-08-31.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(exp, "PRED_DIR", pred_dir)

    loaded = exp._load_predictions()
    assert loaded.empty


def test_reconcile_empty_state_does_not_materialize_derived_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "EXPERIENCE", tmp_path / "experience")
    monkeypatch.setattr(exp, "PRED_DIR", tmp_path / "experience" / "predictions")
    monkeypatch.setattr(exp, "RESULT_DIR", tmp_path / "experience" / "official_results")
    monkeypatch.setattr(exp, "SUMMARY_PATH", tmp_path / "experience" / "experience_summary.json")

    summary = exp.reconcile()

    assert summary["status"] == "NO_PREDICTIONS"
    assert summary["matched_rows"] == 0
    assert not (tmp_path / "experience" / "experience_summary.json").exists()
