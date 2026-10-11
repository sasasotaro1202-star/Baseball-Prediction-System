from __future__ import annotations

from datetime import datetime, timezone
import json

import prediction.pregame_scheduler as scheduler


def _disable_external_research_discovery(monkeypatch):
    monkeypatch.setattr(scheduler, "load_research_active_competitions", lambda: set())


def test_due_games_uses_current_production_runtime_and_requested_time_window(monkeypatch):
    _disable_external_research_discovery(monkeypatch)
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
    result = scheduler.due_games(
        now_utc=now, min_lead_minutes=0.0, preferred_lead_minutes=30.0, scan_ahead_minutes=60.0
    )

    assert result["status"] == "DUE"
    assert result["due_dates"] == ["2026-10-01"]
    row = result["due_games"][0]
    assert row["lead_minutes"] == 60.0
    assert row["prediction_cutoff_utc"] == "2026-10-01T08:00:00+00:00"
    assert row["preferred_prediction_cutoff_utc"] == "2026-10-01T08:30:00+00:00"
    assert row["preferred_30m_met"] is True
    assert row["status"] == "DUE"


def test_due_games_does_not_repeat_an_archived_pregame_snapshot(monkeypatch):
    _disable_external_research_discovery(monkeypatch)
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
            "2026-10-01T08:10:00+00:00",
        )},
    )

    now = datetime(2026, 10, 1, 8, 10, tzinfo=timezone.utc)
    result = scheduler.due_games(now_utc=now)

    assert result["status"] == "NO_DUE_GAMES"
    assert result["due_games"] == []


def test_schedule_parser_ignores_team_labels_before_explicit_schedule_heading():
    parser = scheduler._ScheduleParser()
    parser.feed("<html><body><img alt='読売ジャイアンツ'></body></html>")
    # A team label outside the explicit schedule section is not sufficient
    # evidence to create a prediction-time game candidate.
    assert parser.tokens == []


def test_non_production_runtime_is_reported_but_not_predicted(monkeypatch):
    _disable_external_research_discovery(monkeypatch)
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


def test_mlb_schedule_discovery_is_pit_blocked_and_jst_filtered(monkeypatch):
    payload = {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 123456,
                        "gameDate": "2026-10-01T09:00:00Z",
                        "teams": {
                            "home": {
                                "team": {"name": "Home Club"},
                                "probablePitcher": {"fullName": "Probable Home"},
                            },
                            "away": {
                                "team": {"name": "Away Club"},
                                "probablePitcher": {"fullName": "Probable Away"},
                            },
                        },
                    }
                ]
            }
        ]
    }
    monkeypatch.setattr(scheduler, "_fetch", lambda url: json.dumps(payload))
    rows = scheduler._mlb_schedule_for_date("2026-10-01")
    assert len(rows) == 1
    row = rows[0]
    assert row["game_id"] == "123456"
    assert row["official_start_time"] == "18:00"
    assert row["starter_evidence_status"] == "official_probable_only"
    assert row["pit_status"] == "NOT_ELIGIBLE_OFFICIAL_STARTER_REQUIRED"


def test_research_active_mlb_candidate_never_enters_production_due_set(monkeypatch):
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
    monkeypatch.setattr(scheduler, "load_research_active_competitions", lambda: {"MLB"})
    monkeypatch.setattr(
        scheduler,
        "_schedule_for_date",
        lambda target_date: [
            {"home": "読売ジャイアンツ", "away": "阪神タイガース", "official_start_time": "18:00"}
        ],
    )
    monkeypatch.setattr(
        scheduler,
        "_mlb_schedule_for_date",
        lambda target_date: [
            {
                "game_id": "123456",
                "home": "Home Club",
                "away": "Away Club",
                "official_start_time": "18:30",
                "scheduled_start_utc": "2026-10-01T09:30:00+00:00",
                "home_starter": "Probable Home",
                "away_starter": "Probable Away",
                "starter_evidence_status": "official_probable_only",
                "starter_source": "MLB Stats API schedule",
                "pit_status": "NOT_ELIGIBLE_OFFICIAL_STARTER_REQUIRED",
            }
        ],
    )
    monkeypatch.setattr(scheduler, "_archived_prediction_keys", lambda target_date: set())

    now = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    result = scheduler.due_games(
        now_utc=now,
        min_lead_minutes=0.0,
        preferred_lead_minutes=30.0,
        scan_ahead_minutes=60.0,
    )

    assert result["due_games"] == []
    assert result["due_dates"] == []
    assert result["status"] == "NO_DUE_GAMES"
    assert result["research_status"] == "RESEARCH_DUE"
    assert len(result["research_due_games"]) == 1
    candidate = result["research_due_games"][0]
    assert candidate["league"] == "MLB"
    assert candidate["prediction_eligibility"] == "RESEARCH_ONLY_BLOCKED_UNTIL_OFFICIAL_STARTERS"
    assert candidate["status"] == "RESEARCH_DUE"


def test_research_api_failure_does_not_block_production_scheduler(monkeypatch):
    _disable_external_research_discovery(monkeypatch)
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
    monkeypatch.setattr(scheduler, "_archived_prediction_keys", lambda target_date: set())
    monkeypatch.setattr(scheduler, "load_research_active_competitions", lambda: {"MLB"})
    def fail_mlb(_target_date):
        raise RuntimeError("temporary MLB schedule source failure")
    monkeypatch.setattr(scheduler, "_mlb_schedule_for_date", fail_mlb)

    now = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
    result = scheduler.due_games(
        now_utc=now,
        min_lead_minutes=0.0,
        preferred_lead_minutes=30.0,
        scan_ahead_minutes=60.0,
    )

    assert result["status"] == "DUE"
    assert len(result["due_games"]) == 1
    assert result["research_status"] == "RESEARCH_ERROR"
    assert result["research_due_games"] == []
    assert result["research_errors"][0]["league"] == "MLB"

def test_all_production_runtimes_blocked_does_not_disable_research_discovery(monkeypatch):
    monkeypatch.setattr(
        scheduler,
        "load_runtimes",
        lambda: {
            "NPB": {
                "formal_adoption_status": "BLOCKED_UNTIL_ADOPTED",
                "entrypoint": "production_npb",
            },
            "MLB": {
                "formal_adoption_status": "BLOCKED_UNTIL_REGISTERED",
                "entrypoint": "",
            },
        },
    )
    monkeypatch.setattr(scheduler, "load_research_active_competitions", lambda: {"MLB"})
    monkeypatch.setattr(
        scheduler,
        "_mlb_schedule_for_date",
        lambda target_date: [{
            "game_id": "123456",
            "home": "Home Club",
            "away": "Away Club",
            "official_start_time": "18:30",
            "scheduled_start_utc": "2026-10-01T09:30:00+00:00",
            "home_starter": "Probable Home",
            "away_starter": "Probable Away",
            "starter_evidence_status": "official_probable_only",
            "starter_source": "MLB Stats API schedule",
            "pit_status": "NOT_ELIGIBLE_OFFICIAL_STARTER_REQUIRED",
        }],
    )
    now = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    result = scheduler.due_games(
        now_utc=now,
        min_lead_minutes=0.0,
        preferred_lead_minutes=30.0,
        scan_ahead_minutes=60.0,
    )

    assert result["status"] == "NO_DUE_GAMES"
    assert result["due_games"] == []
    assert result["due_dates"] == []
    assert result["research_status"] == "RESEARCH_DUE"
    assert len(result["research_due_games"]) == 1
    assert {x["status"] for x in result["blocked_runtimes"]} == {"BLOCKED_NO_CURRENT_PRODUCTION_RUNTIME"}


def test_due_games_supports_a_60m_automatic_slot_without_repeating_it(monkeypatch):
    _disable_external_research_discovery(monkeypatch)
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
        lambda target_date: set(),
    )
    monkeypatch.setattr(
        scheduler,
        "_archived_prediction_sources",
        lambda target_date: set(),
    )

    # 60 minutes before 18:00 JST = 08:00 UTC.
    now = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
    result = scheduler.due_games(
        now_utc=now,
        min_lead_minutes=50.0,
        preferred_lead_minutes=60.0,
        scan_ahead_minutes=60.0,
        prediction_source="AUTO_60M",
    )

    assert result["status"] == "DUE"
    row = result["due_games"][0]
    assert row["lead_minutes"] == 60.0
    assert row["prediction_source"] == "AUTO_60M"
    assert row["preferred_prediction_cutoff_utc"] == "2026-10-01T08:00:00+00:00"

    monkeypatch.setattr(
        scheduler,
        "_archived_prediction_sources",
        lambda target_date: {("読売ジャイアンツ", "阪神タイガース", "AUTO_60M")},
    )
    repeated = scheduler.due_games(
        now_utc=now,
        min_lead_minutes=50.0,
        preferred_lead_minutes=60.0,
        scan_ahead_minutes=70.0,
        prediction_source="AUTO_60M",
    )
    assert repeated["due_games"] == []


def test_manual_or_other_source_is_not_blocked_by_automatic_slot(monkeypatch):
    _disable_external_research_discovery(monkeypatch)
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
        lambda target_date: set(),
    )
    monkeypatch.setattr(
        scheduler,
        "_archived_prediction_sources",
        lambda target_date: {("読売ジャイアンツ", "阪神タイガース", "AUTO_60M")},
    )
    now = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)

    manual = scheduler.due_games(
        now_utc=now,
        min_lead_minutes=0.0,
        preferred_lead_minutes=60.0,
        scan_ahead_minutes=60.0,
        prediction_source="MANUAL",
    )
    assert manual["status"] == "DUE"
    assert manual["due_games"][0]["prediction_source"] == "MANUAL"


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
    monkeypatch.setattr(scheduler, "_archived_prediction_sources", lambda target_date: set())
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
    assert row["prediction_source"] == "RESEARCH_SHADOW_TODAY"
    assert row["prediction_eligibility"] == "RESEARCH_SHADOW_PIT_SAFE_STARTERS_REQUIRED"
    assert row["status"] == "RESEARCH_SHADOW_DUE"
    assert row["lead_minutes"] == 60.0



def test_fetch_uses_stdlib_https(monkeypatch):
    seen = {}
    class _Response:
        status = 200
        def read(self):
            return "ok".encode("utf-8")
    class _Connection:
        def __init__(self, host, timeout):
            seen["host"] = host
            seen["timeout"] = timeout
        def request(self, method, path, headers):
            seen["method"] = method
            seen["path"] = path
            seen["headers"] = headers
        def getresponse(self):
            return _Response()
        def close(self):
            seen["closed"] = True
    monkeypatch.setattr(scheduler.http.client, "HTTPSConnection", _Connection)
    monkeypatch.setattr(scheduler.time, "sleep", lambda seconds: None)
    assert scheduler._fetch("https://example.test/path?a=1") == "ok"
    assert seen == {
        "host": "example.test",
        "timeout": 30,
        "method": "GET",
        "path": "/path?a=1",
        "headers": {"User-Agent": "Baseball-Prediction-System/pregame-scheduler"},
        "closed": True,
    }


def test_schedule_parser_accepts_visible_official_team_labels_and_japanese_spacing(monkeypatch):
    html = """
    <html><body>
      <nav><a>日本シリーズ</a><a>オールスター・ゲーム</a></nav>
      <h3>Regular Season (Schedules)</h3>
      <div class="game">
        <span>巨　人</span>
        <span>東京ドーム</span>
        <span>18:00</span>
        <span>DeNA</span>
      </div>
      <div class="game">
        <span>広島</span>
        <span>マツダ</span>
        <span>14:00</span>
        <span>阪神</span>
      </div>
    </body></html>
    """
    monkeypatch.setattr(scheduler, "_fetch", lambda url: html)
    rows = scheduler._schedule_for_date("2026-10-03")
    assert rows == [
        {
            "home": "読売ジャイアンツ",
            "away": "横浜DeNAベイスターズ",
            "official_start_time": "18:00",
        },
        {
            "home": "広島東洋カープ",
            "away": "阪神タイガース",
            "official_start_time": "14:00",
        },
    ]


def test_schedule_parser_ignores_hidden_team_labels_and_script_clocks():
    parser = scheduler._ScheduleParser()
    parser.feed(
        """
        <h3>Regular Season (Schedules)</h3>
        <script>巨人 99:99 DeNA</script>
        <div><span>巨人</span><span>18:00</span><span>DeNA</span></div>
        """
    )
    assert parser.tokens == [
        ("team", "読売ジャイアンツ"),
        ("time", "18:00"),
        ("team", "横浜DeNAベイスターズ"),
    ]


def test_schedule_parser_ignores_navigation_team_links_before_schedule_heading(monkeypatch):
    html = """
    <html><body>
      <nav>
        <span>巨人</span><span>18:01</span><span>DeNA</span>
        <span>日本シリーズ</span>
      </nav>
      <h3>Regular Season (Schedules)</h3>
      <div>
        <span>巨人</span><span>18:00</span><span>DeNA</span>
      </div>
    </body></html>
    """
    monkeypatch.setattr(scheduler, "_fetch", lambda url: html)
    rows = scheduler._schedule_for_date("2026-10-03")
    assert rows == [{
        "home": "読売ジャイアンツ",
        "away": "横浜DeNAベイスターズ",
        "official_start_time": "18:00",
    }]



def test_blocked_npb_shadow_can_discover_today_games_up_to_three_hours_ahead(monkeypatch):
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
        lambda target_date: [{
            "home": "読売ジャイアンツ",
            "away": "阪神タイガース",
            "official_start_time": "18:00",
        }],
    )
    monkeypatch.setattr(scheduler, "_archived_prediction_sources", lambda target_date: set())
    now = datetime(2026, 10, 1, 6, 0, tzinfo=timezone.utc)
    result = scheduler.due_games(
        now_utc=now,
        min_lead_minutes=50.0,
        preferred_lead_minutes=60.0,
        scan_ahead_minutes=180.0,
        prediction_source="AUTO_60M",
    )
    assert result["due_games"] == []
    assert len(result["research_shadow_due_games"]) == 1
    row = result["research_shadow_due_games"][0]
    assert row["lead_minutes"] == 120.0
    assert row["prediction_source"] == "RESEARCH_SHADOW_TODAY"
    assert row["preferred_60m_met"] is True
