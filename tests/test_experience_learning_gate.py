import numpy as np
import pandas as pd

from research.experience_learning_gate import evaluate_locked_holdout


def _frame(n=10):
    cutoffs = pd.date_range("2026-09-01T00:00:00Z", periods=n, freq="h")
    available = cutoffs + pd.Timedelta(hours=2)
    rows=[]
    for i,(c,a) in enumerate(zip(cutoffs,available)):
        rows.append({
            "prediction_id":f"p{i}",
            "game_id":f"g{i}",
            "game_id":f"g{i}",
            "target":"NPB",
            "prediction_cutoff_utc":c,
            "experience_available_at_utc":a,
            "actual_outcome":"HOME_WIN" if i % 3 else "DRAW",
            "home_win_pct":55.0,
            "draw_pct":20.0,
            "away_win_pct":25.0,
            "regime":"normal",
            "score_regime":"balanced",
            "model":"production",
            "situation_tags":"[]",
        })
    return pd.DataFrame(rows)


def test_gate_contract_is_present_on_empty_input():
    result = evaluate_locked_holdout(pd.DataFrame())
    assert result["promotion_status"] == "HOLD"
    assert result["selection_contract"]["production_modified"] is False
    assert result["selection_contract"]["auto_promotion"] is False


def test_gate_holds_with_too_few_cases():
    result = evaluate_locked_holdout(
        _frame(10),
        min_train_cases=5,
        min_holdout_cases=5,
        bootstrap_min_cases=60,
    )
    assert result["status"] in {"INSUFFICIENT_POLICY", "INSUFFICIENT_CASES"}
    assert result["promotion_status"] == "HOLD"


def test_gate_blocks_when_holdout_is_too_small():
    result = evaluate_locked_holdout(
        _frame(10),
        min_train_cases=30,
        min_holdout_cases=30,
    )
    assert result["status"] == "INSUFFICIENT_CASES"
    assert result["promotion_status"] == "HOLD"


def test_gate_evaluates_locked_holdout_but_stays_conservative():
    result = evaluate_locked_holdout(
        _frame(150),
        min_train_cases=30,
        min_holdout_cases=30,
        bootstrap_min_cases=60,
    )
    assert result["status"] == "EVALUATED"
    assert result["selection_contract"]["holdout_outcomes_used_to_build_policy"] is False
    assert result["promotion_status"] == "HOLD"
    assert result["uncertainty"]["status"] == "UNAVAILABLE"


def test_gate_never_enables_auto_promotion():
    result = evaluate_locked_holdout(
        _frame(80),
        min_train_cases=30,
        min_holdout_cases=30,
        bootstrap_min_cases=60,
    )
    assert result["promotion_status"] in {"HOLD", "PROMOTION_CANDIDATE"}
    assert result["selection_contract"]["production_modified"] is False
    assert result["selection_contract"]["auto_promotion"] is False
