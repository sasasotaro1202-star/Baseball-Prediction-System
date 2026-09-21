import pandas as pd

from research.npb_draw_research import evaluate_top_draw_from_backtest


def test_draw_threshold_selection_is_chronological(tmp_path):
    dates = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    rows = []
    for i, dt in enumerate(dates):
        rows.append({
            "datetime": dt.isoformat(),
            "pred_draw": 0.05 + i * 0.01,
            "actual_home_score": 0,
            "actual_away_score": 0 if i == 9 else 1,
        })
    path = tmp_path / "npb_backtest_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    result = evaluate_top_draw_from_backtest(path)
    split = result["threshold_selection_split"]
    assert split["method"] == "chronological_jst_date_70_30"
    assert split["holdout_untouched_during_threshold_selection"] is True
    assert split["development_slates"] == 7
    assert split["evaluation_slates"] == 3
    assert result["fixed_threshold_holdout_evaluation"]["status"] == "PASS"
