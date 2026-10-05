from datetime import datetime, timezone

from research.npb_daily_forecast import (
    EARLY_MAX_LEAD,
    EARLY_MIN_LEAD,
    _stable_prediction_id,
    _target_date,
)


def test_daily_forecast_window_is_disjoint_from_existing_60m_lane():
    assert EARLY_MIN_LEAD > 60.0
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
