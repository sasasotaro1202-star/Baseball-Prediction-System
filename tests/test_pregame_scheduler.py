from __future__ import annotations

from datetime import datetime, timezone

import prediction.pregame_scheduler as scheduler


def test_due_games_uses_current_production_runtime_and_30m_cutoff(monkeypatch):
    monkeypatch.setattr(
        scheduler,
        "load_runtimes",
        lambda: {
            "NPB": {
                "formal_adoption_status": "CURRENT_PRODUCTION",
                "entrypoint": "production_npb",
            },
            "MLB": {
                "formal_adoption_status": "BLOCKED_UNTIL_REGISTERED",
                "entrypoint": "",
            },
        },
    )
    monkeypatch.setattr(
        scheduler,
        "_schedule_for_date",
        lambda target_date: [
            {"home": "読売ジャイアンツ", "away": "阪神タイガース", "official_start_time": "18:00"}
        ],
    )
    monkeypatch.setattr(scheduler, "_archived_prediction_keys", lambda target_date: set())

    now = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
    result = scheduler.due_games(now_utc=now, min_lead_minutes=30.0, scan_ahead_minutes=60.0)

    assert result["status"] == "DUE"
    assert result["due_dates"] == ["2026-10-01"]
    row = result["due_games"][0]
    assert row["lead_minutes"] == 60.0
    assert row["prediction_cutoff_utc"] == "2026-10-01T09:30:00+00:00"
    assert row["status"] == "DUE"


def test_due_games_does_not_repeat_an_archived_30m_snapshot(monkeypatch):
    monkeypatch.setattr(
        scheduler,
        "load_runtimes",
        lambda: {
            "NPB": {
                "formal_adoption_status": "CURRENT_PRODUCTION",
                "entrypoint": "production_npb",
            }
        },
    )
    monkeypatch.setattr(
        scheduler,
        "_schedule_for_date",
        lambda target_date: [
            {"home": "読売ジャイアンツ", "away": "阪神タイガース", "official_start_time": "18:00"}
        ],
    )
    monkeypatch.setattr(
        scheduler,
        "_archived_prediction_keys",
        lambda target_date: {(
            "読売ジャイアンツ",
            "阪神タイガース",
            "2026-10-01T09:30:00+00:00",
        )},
    )

    now = datetime(2026, 10, 1, 8, 10, tzinfo=timezone.utc)
    result = scheduler.due_games(now_utc=now)

    assert result["status"] == "NO_DUE_GAMES"
    assert result["due_games"] == []


def test_schedule_parser_fails_closed_without_deterministic_game():
    monkeypatch = None
    parser = scheduler._ScheduleParser()
    parser.feed("<html><body><img alt='読売ジャイアンツ'></body></html>")
    assert parser.tokens == [("team", "読売ジャイアンツ")]


def test_non_production_runtime_is_reported_but_not_predicted(monkeypatch):
    monkeypatch.setattr(
        scheduler,
        "load_runtimes",
        lambda: {
            "NPB": {
                "formal_adoption_status": "CURRENT_PRODUCTION",
                "entrypoint": "production_npb",
            },
            "MLB": {
                "formal_adoption_status": "CURRENT_PRODUCTION",
                "entrypoint": "future_mlb_entrypoint",
            },
        },
    )
    monkeypatch.setattr(
        scheduler,
        "_schedule_for_date",
        lambda target_date: [
            {"home": "読売ジャイアンツ", "away": "阪神タイガース", "official_start_time": "18:00"}
        ],
    )
    monkeypatch.setattr(scheduler, "_archived_prediction_keys", lambda target_date: set())

    now = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
    result = scheduler.due_games(now_utc=now)

    assert "MLB" in {x["league"] for x in result["blocked_runtimes"]}
    assert all(x["league"] == "NPB" for x in result["due_games"])
