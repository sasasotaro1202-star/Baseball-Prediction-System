import numpy as np
import pandas as pd

from research.ultimate_v13_real_oos import run_real_oos_bridge


def _frame(with_pit=True, n=120):
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for i in range(n):
        pt = base + pd.Timedelta(hours=i)
        row = {
            "game_id": f"g{i}",
            "prediction_time": pt,
            "available_at": pt - pd.Timedelta(minutes=30),
            "pred_home": .65,
            "pred_away": .35,
            "actual_home_score": 1 if i % 3 else 0,
            "actual_away_score": 0 if i % 3 else 1,
        }
        if not with_pit:
            row.pop("available_at")
        rows.append(row)
    return pd.DataFrame(rows)


def test_real_oos_requires_pit():
    p = _frame(with_pit=False)
    path = "/tmp/ultimate_v13_no_pit.csv"
    p.to_csv(path, index=False)
    report = run_real_oos_bridge(path, league="MLB")
    assert report["status"] == "BLOCKED"
    assert "pit_not_verified" in report["blockers"]


def test_real_oos_executes_with_real_target_and_pit_contract():
    p = _frame()
    path = "/tmp/ultimate_v13_real_oos.csv"
    p.to_csv(path, index=False)
    report = run_real_oos_bridge(path, league="MLB", data_snapshot_id="snap")
    assert report["status"] == "EXECUTED"
    assert report["promotion_status"] == "HOLD"
    assert set(report["metrics"]) == {"Accuracy", "LogLoss", "Brier", "ECE"}
    assert report["pit"]["status"] == "PASS"


def test_multi_model_panel_never_fabricates():
    p = _frame(n=120)
    q = p.copy()
    q["model"] = "A"
    p["model"] = "A"
    q["model"] = "A"
    # Duplicate model panel intentionally remains structurally invalid through
    # duplicate game/model identities after concatenation.
    panel = pd.concat([p, q], ignore_index=True)
    path = "/tmp/ultimate_v13_bad_panel.csv"
    panel.to_csv(path, index=False)
    report = run_real_oos_bridge(path, league="MLB")
    assert report["model_panel"]["status"] == "BLOCKED"


def test_complete_model_panel_adds_error_and_future_failure_diagnostics():
    p = _frame(n=120)
    q = p.copy()
    p["model"] = "A"
    q["model"] = "B"
    q["pred_home"] = .55
    q["pred_away"] = .45
    panel = pd.concat([p, q], ignore_index=True).sort_values(
        ["prediction_time", "game_id", "model"], kind="mergesort"
    ).reset_index(drop=True)
    path = "/tmp/ultimate_v13_complete_panel.csv"
    panel.to_csv(path, index=False)
    report = run_real_oos_bridge(path, league="MLB")
    assert report["status"] == "EXECUTED"
    assert report["model_panel"]["status"] == "PASS"
    assert "error_correlation" in report["model_panel"]
    assert report["model_panel"]["future_failure"]["A"]
    assert report["model_panel"]["time_to_failure"]["A"] >= 1.0
