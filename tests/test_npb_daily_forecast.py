from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from research.npb_daily_forecast import (
    EARLY_MAX_LEAD,
    EARLY_MIN_LEAD,
    _stable_prediction_id,
    _target_date,
)


def test_daily_forecast_window_covers_rapid_same_day_forecasts():
    assert 0.0 < EARLY_MIN_LEAD < 60.0
    assert EARLY_MAX_LEAD >= 180.0
    assert EARLY_MAX_LEAD > EARLY_MIN_LEAD


def test_prediction_id_is_deterministic_and_source_bound():
    a = _stable_prediction_id("NPB-2026-10-05-x", "2026-10-05T07:00:00+00:00")
    b = _stable_prediction_id("NPB-2026-10-05-x", "2026-10-05T07:00:00+00:00")
    c = _stable_prediction_id("NPB-2026-10-05-x", "2026-10-05T07:01:00+00:00")
    assert a == b
    assert a != c


def test_target_date_preserves_explicit_date():
    assert _target_date("2026-10-05") == "2026-10-05"


def test_target_date_uses_jst_for_default(monkeypatch):
    fixed = datetime(2026, 10, 4, 15, 30, tzinfo=timezone.utc)
    monkeypatch.setattr("research.npb_daily_forecast._now_utc", lambda: fixed)
    assert _target_date(None) == "2026-10-05"

def test_daily_workflow_has_fast_schedule_cache_and_pbp_staging():
    workflow = (
        (ROOT / ".github" / "workflows" / "npb_daily_early_research.yml")
        .read_text(encoding="utf-8")
    )
    assert '- cron: "*/5 * * * *"' in workflow
    assert 'branches: [main]' in workflow
    assert 'cache: pip' in workflow
    assert 'actions/cache/restore@1bd1e32a3bdc45362d1e726936510720a7c30a57' in workflow
    assert 'key: npb-game-script-pbp-v1-linux' in workflow
    assert 'gh release download pbp' in workflow
    assert 'data/pbp' in workflow
    assert '15-180 minute research window' in workflow


def test_loader_can_use_shared_pbp_cache():
    import pandas as pd
    from baseball_backtest import BaseballBacktest

    data_dir = ROOT / "tests" / "_tmp_shared_pbp_loader"
    cache_dir = data_dir / "pbp"
    data_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    try:
        pd.DataFrame([{"game_id": "cached-game", "date": "2026-01-01", "row_order": 1}]).to_csv(
            cache_dir / "2026-01_pbp.csv", index=False
        )
        pd.DataFrame([{"game_id": "fallback-game", "date": "2026-01-02", "row_order": 1}]).to_csv(
            data_dir / "2026-02_pbp.csv", index=False
        )
        bt = BaseballBacktest(data_dir)
        bt._normalize_npb_pbp = lambda frame: frame
        loaded = bt.load_npb_pbp()
        assert set(loaded["game_id"]) == {"cached-game"}
    finally:
        import shutil
        shutil.rmtree(data_dir, ignore_errors=True)
