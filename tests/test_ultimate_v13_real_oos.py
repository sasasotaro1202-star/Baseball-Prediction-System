import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.future_generalization_v13 import predictability_series
from research.ultimate_v13_real_oos import run_real_oos_bridge


def _frame(with_pit=True, n=120):
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for i in range(n):
        pt = base + pd.Timedelta(hours=i)
        row = {
            "game_id": f"g{i}",
            "datetime": pt,
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


def _write_pit_ledgers(root: Path, frame: pd.DataFrame) -> tuple[Path, Path]:
    availability = root / "availability.jsonl"
    snapshots = root / "snapshots.jsonl"
    availability.write_text(
        "".join(
            json.dumps(
                {
                    "game_id": str(row["game_id"]),
                    "event_id": str(row["game_id"]),
                    "prediction_cutoff": pd.Timestamp(row["prediction_time"]).isoformat(),
                    "observed_at": pd.Timestamp(row["prediction_time"]).isoformat(),
                }
            )
            + "\n"
            for _, row in frame.iterrows()
        ),
        encoding="utf-8",
    )
    snapshots.write_text(
        "".join(
            json.dumps(
                {
                    "entity_id": str(row["game_id"]),
                    "status": "KNOWN",
                    "available_at": pd.Timestamp(row["available_at"]).isoformat(),
                }
            )
            + "\n"
            for _, row in frame.iterrows()
        ),
        encoding="utf-8",
    )
    return availability, snapshots


def test_real_oos_requires_pit(tmp_path):
    p = _frame(with_pit=False)
    path = tmp_path / "ultimate_v13_no_pit.csv"
    p.to_csv(path, index=False)
    report = run_real_oos_bridge(
        path,
        league="MLB",
        pit_availability_path=tmp_path / "empty_availability.jsonl",
        pit_snapshots_path=tmp_path / "empty_snapshots.jsonl",
    )
    assert report["status"] == "BLOCKED"
    assert "pit_not_verified" in report["blockers"]


def test_real_oos_executes_with_real_target_and_pit_contract(tmp_path):
    p = _frame()
    path = tmp_path / "ultimate_v13_real_oos.csv"
    p.to_csv(path, index=False)
    availability, snapshots = _write_pit_ledgers(tmp_path, p)
    report = run_real_oos_bridge(
        path,
        league="MLB",
        data_snapshot_id="snap",
        pit_availability_path=availability,
        pit_snapshots_path=snapshots,
    )
    assert report["status"] == "EXECUTED"
    assert report["promotion_status"] == "HOLD"
    assert set(report["metrics"]) == {"Accuracy", "LogLoss", "Brier", "ECE"}
    assert report["pit"]["status"] == "PASS"
    assert report["pit_join"]["coverage"] == 1.0


def test_multi_model_panel_never_fabricates(tmp_path):
    p = _frame(n=120)
    q = p.copy()
    q["model"] = "A"
    p["model"] = "A"
    panel = pd.concat([p, q], ignore_index=True)
    path = tmp_path / "ultimate_v13_bad_panel.csv"
    panel.to_csv(path, index=False)
    availability, snapshots = _write_pit_ledgers(tmp_path, p)
    report = run_real_oos_bridge(
        path,
        league="MLB",
        pit_availability_path=availability,
        pit_snapshots_path=snapshots,
    )
    assert report["model_panel"]["status"] == "BLOCKED"


def test_complete_model_panel_adds_error_and_future_failure_diagnostics(tmp_path):
    p = _frame(n=120)
    q = p.copy()
    p["model"] = "A"
    q["model"] = "B"
    q["pred_home"] = .55
    q["pred_away"] = .45
    panel = pd.concat([p, q], ignore_index=True).sort_values(
        ["prediction_time", "game_id", "model"], kind="mergesort"
    ).reset_index(drop=True)
    path = tmp_path / "ultimate_v13_complete_panel.csv"
    panel.to_csv(path, index=False)
    availability, snapshots = _write_pit_ledgers(tmp_path, p)
    report = run_real_oos_bridge(
        path,
        league="MLB",
        pit_availability_path=availability,
        pit_snapshots_path=snapshots,
    )
    assert report["status"] == "EXECUTED"
    assert report["model_panel"]["status"] == "PASS"
    assert "error_correlation" in report["model_panel"]
    assert report["model_panel"]["future_failure"]["A"]
    assert report["model_panel"]["time_to_failure"]["A"] >= 1.0
