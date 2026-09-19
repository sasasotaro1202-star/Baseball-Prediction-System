import pandas as pd

from baseball_backtest import BaseballBacktest


def _row():
    return pd.Series(
        {
            "league": "MLB",
            "datetime": "2026-06-01T18:00:00+00:00",
            "home": "Home",
            "away": "Away",
            "home_score": 0,
            "away_score": 0,
            "home_starter": "",
            "away_starter": "",
            "home_lineup_json": '[{"player_id":"1","batting_order":1}]',
            "away_lineup_json": '[{"player_id":"2","batting_order":1}]',
            "weather_temp_c": 30.0,
            "weather_humidity_pct": 50.0,
            "weather_wind_kmh": 20.0,
            "weather_precip_mm": 0.0,
        }
    )


def test_lineup_and_weather_are_disabled_without_explicit_pit_context(monkeypatch):
    monkeypatch.delenv("PIT_SAFE_CONTEXT_DATA", raising=False)
    bt = BaseballBacktest()
    features = bt.match_features(_row())

    assert features["context_pit_safe"] == 0.0
    assert "weather_temp_c" not in features
    assert "h_lineup_avg" not in features


def test_lineup_and_weather_require_both_explicit_pre_cutoff_timestamps(monkeypatch):
    monkeypatch.setenv("PIT_SAFE_CONTEXT_DATA", "1")
    row = _row()
    row["prediction_cutoff"] = "2026-06-01T15:00:00+00:00"
    row["lineup_announced_at"] = "2026-06-01T13:00:00+00:00"
    row["weather_available_at"] = "2026-06-01T14:00:00+00:00"

    bt = BaseballBacktest()
    features = bt.match_features(row)

    assert features["context_pit_safe"] == 1.0
    assert features["weather_temp_c"] == 30.0
    assert "h_lineup_avg" in features


def test_context_pit_gate_rejects_post_cutoff_data(monkeypatch):
    monkeypatch.setenv("PIT_SAFE_CONTEXT_DATA", "1")
    row = _row()
    row["prediction_cutoff"] = "2026-06-01T15:00:00+00:00"
    row["lineup_announced_at"] = "2026-06-01T16:00:00+00:00"
    row["weather_available_at"] = "2026-06-01T14:00:00+00:00"

    bt = BaseballBacktest()
    features = bt.match_features(row)

    assert features["context_pit_safe"] == 0.0
    assert "weather_temp_c" not in features
    assert "h_lineup_avg" not in features
