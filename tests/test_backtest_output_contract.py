import pandas as pd

from research.backtest_output_contract import add_top_draw_selection, _repair_scores


def test_repair_uses_exact_top_four_and_total_run_boundary():
    df = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "datetime": "2026-09-01T09:00:00Z",
                "pred_home": 0.45,
                "pred_draw": 0.20,
                "pred_away": 0.35,
                "lambda_home": 6.0,
                "lambda_away": 6.0,
                "score1": "2-2",
                "score2": "2-1",
                "score3": "1-2",
                "score4": "その他",
                "low": 0.98,
                "high": 0.02,
            }
        ]
    )
    got = _repair_scores(df)
    assert len(got.loc[0, "score1"].split("-")) == 2
    assert got.loc[0, "score4"] != "その他"
    assert got.loc[0, "low"] < 0.98
    assert abs(float(got.loc[0, "low"]) + float(got.loc[0, "high"]) - 1.0) < 1e-12


def test_top_draw_selection_is_exactly_one_per_jst_date_and_not_winner_override():
    df = pd.DataFrame(
        [
            {"game_id": "g1", "datetime": "2026-09-01T09:00:00Z", "pred_home": 0.60, "pred_draw": 0.10, "pred_away": 0.30},
            {"game_id": "g2", "datetime": "2026-09-01T10:00:00Z", "pred_home": 0.31, "pred_draw": 0.22, "pred_away": 0.47},
            {"game_id": "g3", "datetime": "2026-09-01T11:00:00Z", "pred_home": 0.44, "pred_draw": 0.18, "pred_away": 0.38},
            {"game_id": "g4", "datetime": "2026-09-02T09:00:00Z", "pred_home": 0.40, "pred_draw": 0.26, "pred_away": 0.34},
        ]
    )
    got = add_top_draw_selection(df)
    selected = got[got["top_draw_selection"]]
    assert selected["game_id"].astype(str).tolist() == ["g2", "g4"]
    assert got.groupby("date_jst")["top_draw_selection"].sum().tolist() == [1, 1]
    assert float(selected.iloc[0]["pred_draw"]) == 0.22
    assert float(selected.iloc[0]["pred_away"]) == 0.47
