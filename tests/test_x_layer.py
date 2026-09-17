from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from data.x_acquisition import MAX_LOOKBACK_DAYS, validate_window
from research.x_offline_eval import evaluate

ROOT = Path(__file__).resolve().parents[1]
X_WORKFLOW = ROOT / ".github" / "workflows" / "baseball_x_research.yml"


def test_x_window_rejects_historical_backfill_beyond_recent_search():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError):
        validate_window(now - timedelta(days=MAX_LOOKBACK_DAYS + 1), now, now)


def test_x_offline_eval_drops_pit_unsafe_rows(tmp_path: Path):
    cutoff = datetime(2026, 9, 17, tzinfo=timezone.utc)
    rows = []
    for i in range(200):
        available = cutoff - timedelta(minutes=10) if i < 199 else cutoff + timedelta(minutes=10)
        rows.append({
            "y_true": i % 2,
            "baseline_prob": 0.55 if i % 2 else 0.45,
            "candidate_prob": 0.60 if i % 2 else 0.40,
            "prediction_cutoff": cutoff.isoformat(),
            "feature_available_at": available.isoformat(),
            "season": 2026,
        })
    path = tmp_path / "x_eval.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = evaluate(path)
    assert result["rows_input"] == 200
    assert result["rows_pit_safe"] == 199
    assert result["rows_dropped_pit"] == 1
    assert result["promotion"] == "NOT_APPLICABLE"


def test_x_workflow_isolated_from_production_path():
    text = X_WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "X_BEARER_TOKEN" in text
    assert "baseball_closed_loop.yml" not in text
