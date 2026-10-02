tests/test_pregame_scheduler.py

def test_due_games_exposes_npb_research_shadow_when_production_is_blocked(monkeypatch):
    _disable_external_research_discovery(monkeypatch)
    monkeypatch.setattr(
        scheduler,
        "load_runtimes",
        lambda: {
            "NPB": {
                "formal_adoption_status": "BLOCKED_UNTIL_ADOPTED",
                "entrypoint": "production_npb",
            }
        },
    )
    monkeypatch.setattr(
        scheduler,
        "_schedule_for_date",
        lambda target_date: [
            {
                "home": "読売ジャイアンツ",
                "away": "阪神タイガース",
                "official_start_time": "18:00",
            }
        ],
    )
    monkeypatch.setattr(
        scheduler,
        "_archived_prediction_sources",
        lambda target_date: set(),
    )
    now = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
    result = scheduler.due_games(
        now_utc=now,
        min_lead_minutes=50.0,
        preferred_lead_minutes=60.0,
        scan_ahead_minutes=60.0,
        prediction_source="AUTO_60M",
    )
    assert result["due_games"] == []
    assert len(result["research_shadow_due_games"]) == 1
    row = result["research_shadow_due_games"][0]
    assert row["prediction_source"] == "RESEARCH_SHADOW_AUTO_60M"
    assert row["prediction_eligibility"] == "RESEARCH_SHADOW_PIT_SAFE_STARTERS_REQUIRED"
    assert row["status"] == "RESEARCH_SHADOW_DUE"
    assert row["lead_minutes"] == 60.0
