import json
from pathlib import Path

import pandas as pd
import pytest

import research.experience_rollup as roll
import research.experience_ledger as ledger


def _write_prediction(root: Path, *, cutoff: str, game_id: str = "NPB-2026-09-26-1"):
    p = root / "predictions"
    p.mkdir(parents=True, exist_ok=True)
    obj = {
        "game_id": game_id,
        "datetime_jst": "2026-09-26T14:00:00+09:00",
        "prediction_cutoff_utc": cutoff,
        "prediction_generated_at": cutoff,
        "starter_evidence_observed_at_utc": cutoff,
        "starter_evidence_status": "official_announced",
        "pit_status": "PASS",
        "source_run_id": "1",
        "home": "横浜DeNAベイスターズ",
        "away": "阪神タイガース",
        "home_starter": "尾形崇斗",
        "away_starter": "才木浩人",
        "regime": "s2_e2_m2",
        "score_regime": "global",
        "model": "Production",
        "situation_tags": ["regime:s2_e2_m2", "offense:balanced"],
        "home_win_pct": 60.0,
        "draw_pct": 5.0,
        "away_win_pct": 35.0,
        "low_pct": 70.0,
        "high_pct": 30.0,
        "lambda_home": 3.2,
        "lambda_away": 2.4,
        "shared_lambda": 0.0,
        "top4_exact_scores": [
            {"score": "3-2", "prob_pct": 8.0},
            {"score": "3-3", "prob_pct": 7.0},
            {"score": "4-2", "prob_pct": 6.0},
            {"score": "2-2", "prob_pct": 5.0},
        ],
        "classification_regime_model_weights": {
            "RandomForest": 0.30,
            "ExtraTrees": 0.25,
            "CatBoost": 0.25,
            "KNNAnalog": 0.10,
            "XGBoost": 0.10,
        },
    }
    (p / "2026-09-26.jsonl").write_text(
        json.dumps(obj, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def test_empty_result_cache_is_refreshed(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experience"
    monkeypatch.setattr(roll, "RESULT_DIR", exp_dir / "official_results")
    result_dir = exp_dir / "official_results"
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

    monkeypatch.setattr(roll, "_fetch_month", fake_fetch)
    loaded = roll._cache_results(
        [pd.Timestamp("2026-10-02T18:00:00+09:00")]
    )

    assert calls == [(2026, 10)]
    assert len(loaded) == 1
    assert int(loaded.loc[0, "away_score"]) == 1
    assert len(pd.read_csv(cache)) == 1


def test_current_result_cache_is_refreshed_even_when_nonempty(tmp_path, monkeypatch):
    monkeypatch.setattr(roll, "RESULT_DIR", tmp_path / "experience" / "official_results")
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

    monkeypatch.setattr(roll, "_fetch_month", fake_fetch)
    loaded = roll._cache_results(
        [pd.Timestamp("2026-10-02T18:00:00+09:00")]
    )
    assert calls == [(2026, 10)]
    assert len(loaded) == 1
    assert int(loaded.loc[0, "home_score"]) == 5
    assert loaded.loc[0, "source_url"] == "test://new"


def test_rollup_preserves_multiple_pregame_snapshots(tmp_path, monkeypatch):
    exp = tmp_path / "experience"
    monkeypatch.setattr(roll, "EXPERIENCE", exp)
    monkeypatch.setattr(roll, "PRED_DIR", exp / "predictions")
    monkeypatch.setattr(roll, "RESULT_DIR", exp / "official_results")
    monkeypatch.setattr(roll, "SNAPSHOT_PATH", exp / "snapshot_experience_ledger.csv")
    monkeypatch.setattr(roll, "SNAPSHOT_JSONL", exp / "snapshot_experience_ledger.jsonl")
    monkeypatch.setattr(roll, "CASE_SUMMARY_PATH", exp / "experience_case_summary.json")
    monkeypatch.setattr(roll, "TRAINING_INDEX_PATH", exp / "experience_training_index.json")

    pdir = exp / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, cutoff in enumerate(("2026-09-26T00:00:00+00:00", "2026-09-26T02:00:00+00:00")):
        obj = json.loads((tmp_path / "template.json").read_text()) if (tmp_path / "template.json").exists() else None
        if obj is None:
            _write_prediction(tmp_path / "stage", cutoff=cutoff)
            obj = json.loads((tmp_path / "stage" / "predictions" / "2026-09-26.jsonl").read_text())
            (tmp_path / "stage" / "predictions" / "2026-09-26.jsonl").unlink()
        rows.append(json.dumps(obj, ensure_ascii=False))
    (pdir / "2026-09-26.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

    monkeypatch.setattr(roll, "_cache_results", lambda dates: pd.DataFrame([{
        "date": "2026-09-26",
        "home": "横浜DeNAベイスターズ",
        "away": "阪神タイガース",
        "home_score": 4,
        "away_score": 2,
        "source_url": "test://npb",
    }]))

    result = roll.rollup()
    snap = pd.read_csv(exp / "snapshot_experience_ledger.csv")
    canonical = json.loads((exp / "experience_case_summary.json").read_text(encoding="utf-8"))
    assert result["matched_snapshot_rows"] == 2
    assert len(snap) == 2
    assert canonical["unique_games_with_results"] == 1
    assert canonical["experience_cases"]["rows"] == 1
    assert canonical["all_snapshot_experience"]["rows"] == 2
    assert result["timing_30m"]["eligible_rows"] == 2
    assert result["timing_30m"]["on_time_rows"] == 2
    assert result["timing_30m"]["compliance_rate"] == 1.0


def test_rollup_rejects_post_start_prediction(tmp_path, monkeypatch):
    exp = tmp_path / "experience"
    monkeypatch.setattr(roll, "EXPERIENCE", exp)
    monkeypatch.setattr(roll, "PRED_DIR", exp / "predictions")
    monkeypatch.setattr(roll, "RESULT_DIR", exp / "official_results")
    monkeypatch.setattr(roll, "CASE_SUMMARY_PATH", exp / "experience_case_summary.json")
    monkeypatch.setattr(roll, "TRAINING_INDEX_PATH", exp / "experience_training_index.json")
    pdir = exp / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    obj = json.loads((tmp_path / "post.json").read_text()) if (tmp_path / "post.json").exists() else None
    _write_prediction(tmp_path / "stage2", cutoff="2026-09-26T06:00:00+00:00")
    obj = json.loads((tmp_path / "stage2" / "predictions" / "2026-09-26.jsonl").read_text())
    (pdir / "2026-09-26.jsonl").write_text(json.dumps(obj, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="information cutoff is not pregame"):
        roll.rollup()


def test_rollup_low_high_threshold_uses_normalized_probability_scale():
    merged = pd.DataFrame(
        [
            {
                "game_id": "low",
                "datetime_jst": pd.Timestamp("2026-09-26T05:00:00+00:00"),
                "prediction_cutoff_utc": pd.Timestamp("2026-09-26T02:00:00+00:00"),
                "home_score": 4,
                "away_score": 2,
                "home_win_pct": 60.0,
                "draw_pct": 5.0,
                "away_win_pct": 35.0,
                "low_pct": 70.0,
                "high_pct": 30.0,
                "lambda_home": 3.2,
                "lambda_away": 2.4,
                "top4_exact_scores": [],
            },
            {
                "game_id": "high",
                "datetime_jst": pd.Timestamp("2026-09-26T06:00:00+00:00"),
                "prediction_cutoff_utc": pd.Timestamp("2026-09-26T03:00:00+00:00"),
                "home_score": 5,
                "away_score": 4,
                "home_win_pct": 60.0,
                "draw_pct": 5.0,
                "away_win_pct": 35.0,
                "low_pct": 30.0,
                "high_pct": 70.0,
                "lambda_home": 3.2,
                "lambda_away": 2.4,
                "top4_exact_scores": [],
            },
        ]
    )
    scored = roll._evaluate(merged)
    assert scored["high_probability"].tolist() == [0.3, 0.7]
    assert scored["low_high_predicted"].tolist() == [0, 1]
    assert scored["low_high_actual"].tolist() == [0, 1]
    assert scored["low_high_correct"].tolist() == [1, 1]


def test_rollup_accepts_four_decimal_serialization_rounding():
    merged = pd.DataFrame(
        [{
            "game_id": "rounded",
            "datetime_jst": pd.Timestamp("2026-09-26T14:00:00+09:00"),
            "prediction_cutoff_utc": pd.Timestamp("2026-09-26T02:00:00+00:00"),
            "home_score": 4,
            "away_score": 2,
            # Sum is 99.9999 because three four-decimal percentages were
            # independently rounded. This is serialization drift, not a
            # materially invalid probability vector.
            "home_win_pct": 33.3333,
            "draw_pct": 33.3333,
            "away_win_pct": 33.3333,
            "low_pct": 66.6666,
            "high_pct": 33.3333,
            "lambda_home": 3.2,
            "lambda_away": 2.4,
            "top4_exact_scores": [],
        }]
    )
    scored = roll._evaluate(merged)
    probs = scored[["home_win_pct", "draw_pct", "away_win_pct"]].iloc[0].to_numpy(float)
    low_high = scored[["low_probability", "high_probability"]].iloc[0].to_numpy(float)
    assert abs(float(probs.sum()) - 1.0) < 1e-12
    assert abs(float(low_high.sum()) - 1.0) < 1e-12


def test_rollup_rejects_invalid_probability_contract():
    merged = pd.DataFrame(
        [{
            "game_id": "invalid",
            "home_score": 4,
            "away_score": 2,
            "home_win_pct": 80.0,
            "draw_pct": 10.0,
            "away_win_pct": 30.0,
            "low_pct": 70.0,
            "high_pct": 30.0,
            "lambda_home": 3.2,
            "lambda_away": 2.4,
            "top4_exact_scores": [],
        }]
    )
    try:
        roll._evaluate(merged)
    except RuntimeError as exc:
        assert "invalid win probabilities" in str(exc)
    else:
        raise AssertionError("invalid probability contract must fail closed")


def test_rollup_rejects_invalid_low_high_contract():
    merged = pd.DataFrame(
        [{
            "game_id": "invalid-lh",
            "home_score": 4,
            "away_score": 2,
            "home_win_pct": 60.0,
            "draw_pct": 5.0,
            "away_win_pct": 35.0,
            "low_pct": 80.0,
            "high_pct": 30.0,
            "lambda_home": 3.2,
            "lambda_away": 2.4,
            "top4_exact_scores": [],
        }]
    )
    try:
        roll._evaluate(merged)
    except RuntimeError as exc:
        assert "invalid Low/High probabilities" in str(exc)
    else:
        raise AssertionError("invalid Low/High contract must fail closed")


def test_reconcile_persists_no_completed_results_status(tmp_path, monkeypatch):
    exp = tmp_path / "experience"
    monkeypatch.setattr(ledger, "EXPERIENCE", exp)
    monkeypatch.setattr(ledger, "SUMMARY_PATH", exp / "experience_summary.json")

    pred = pd.DataFrame(
        [{
            "game_id": "NPB-2026-09-26-1",
            "datetime_jst": pd.Timestamp("2026-09-26T14:00:00+09:00"),
            "prediction_cutoff_utc": pd.Timestamp("2026-09-26T02:00:00+00:00"),
        }]
    )
    monkeypatch.setattr(ledger, "_load_predictions", lambda: pred)
    monkeypatch.setattr(ledger, "_load_cached_results", lambda dates: pd.DataFrame())

    result = ledger.reconcile()
    persisted = json.loads((exp / "experience_summary.json").read_text(encoding="utf-8"))
    assert result["status"] == "NO_COMPLETED_RESULTS"
    assert persisted == result


def test_rollup_writes_training_index_when_results_are_unavailable(tmp_path, monkeypatch):
    exp = tmp_path / "experience"
    monkeypatch.setattr(roll, "EXPERIENCE", exp)
    monkeypatch.setattr(roll, "PRED_DIR", exp / "predictions")
    monkeypatch.setattr(roll, "RESULT_DIR", exp / "official_results")
    monkeypatch.setattr(roll, "CASE_SUMMARY_PATH", exp / "experience_case_summary.json")
    monkeypatch.setattr(roll, "TRAINING_INDEX_PATH", exp / "experience_training_index.json")

    _write_prediction(tmp_path, cutoff="2026-09-26T02:00:00+00:00")
    source = tmp_path / "predictions" / "2026-09-26.jsonl"
    (exp / "predictions").mkdir(parents=True, exist_ok=True)
    (exp / "predictions" / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(roll, "_cache_results", lambda dates: pd.DataFrame())

    result = roll.rollup()
    training = json.loads((exp / "experience_training_index.json").read_text(encoding="utf-8"))
    assert result["status"] == "NO_COMPLETED_RESULTS"
    assert training == result


def test_rollup_reports_independent_prediction_target_metrics(tmp_path, monkeypatch):
    exp = tmp_path / "experience"
    monkeypatch.setattr(roll, "EXPERIENCE", exp)
    monkeypatch.setattr(roll, "PRED_DIR", exp / "predictions")
    monkeypatch.setattr(roll, "RESULT_DIR", exp / "official_results")
    monkeypatch.setattr(roll, "SNAPSHOT_PATH", exp / "snapshot_experience_ledger.csv")
    monkeypatch.setattr(roll, "SNAPSHOT_JSONL", exp / "snapshot_experience_ledger.jsonl")
    monkeypatch.setattr(roll, "CASE_SUMMARY_PATH", exp / "experience_case_summary.json")
    monkeypatch.setattr(roll, "TRAINING_INDEX_PATH", exp / "experience_training_index.json")

    _write_prediction(tmp_path, cutoff="2026-09-26T02:00:00+00:00")
    source = tmp_path / "predictions" / "2026-09-26.jsonl"
    (exp / "predictions").mkdir(parents=True, exist_ok=True)
    (exp / "predictions" / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(roll, "_cache_results", lambda dates: pd.DataFrame([{
        "date": "2026-09-26",
        "home": "横浜DeNAベイスターズ",
        "away": "阪神タイガース",
        "home_score": 4,
        "away_score": 2,
        "source_url": "test://npb",
    }]))

    result = roll.rollup()
    training = json.loads((exp / "experience_training_index.json").read_text(encoding="utf-8"))
    targets = result["by_prediction_target"]
    assert set(targets) == {"win_3way", "low_high", "exact_score"}
    assert training["timing_30m"] == result["timing_30m"]
    assert training["revision_intelligence"] == result["revision_intelligence"]
    assert targets["win_3way"]["rows"] == 1
    assert targets["low_high"]["rows"] == 1
    assert targets["exact_score"]["rows"] == 1
    assert result["by_prediction_target_all_snapshots"]["win_3way"]["rows"] == 1


def test_rollup_quarantines_legacy_scheduled_cutoff(tmp_path, monkeypatch):
    exp = tmp_path / "experience"
    monkeypatch.setattr(roll, "EXPERIENCE", exp)
    monkeypatch.setattr(roll, "PRED_DIR", exp / "predictions")
    monkeypatch.setattr(roll, "RESULT_DIR", exp / "official_results")

    pdir = exp / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    legacy = {
        "game_id": "NPB-legacy-1",
        "datetime_jst": "2026-09-26T14:00:00+09:00",
        "prediction_cutoff_utc": "2026-09-26T02:00:00+00:00",
        "prediction_generated_at": "2026-09-26T01:52:00+00:00",
        "starter_evidence_observed_at_utc": "2026-09-26T01:52:00+00:00",
        "pit_status": "PASS",
        "prediction_id": "legacy-1",
        "home": "横浜DeNAベイスターズ",
        "away": "阪神タイガース",
        "home_win_pct": 60.0,
        "draw_pct": 5.0,
        "away_win_pct": 35.0,
        "low_pct": 70.0,
        "high_pct": 30.0,
        "lambda_home": 3.2,
        "lambda_away": 2.4,
        "top4_exact_scores": [],
    }
    (pdir / "2026-09-26.jsonl").write_text(
        json.dumps(legacy, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    result = roll.rollup()
    # A quarantined-only corpus is safely reduced to zero reusable snapshots.
    # The legacy row is not treated as a valid PIT prediction.
    assert result["status"] == "NO_PREGAME_PREDICTIONS"
    assert result["snapshot_rows"] == 0
