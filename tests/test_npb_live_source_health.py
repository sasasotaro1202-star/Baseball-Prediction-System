from __future__ import annotations

from datetime import date

from scripts import verify_npb_live_sources as health


def test_select_pregame_probe_skips_empty_day_without_masking_source_failure(monkeypatch):
    calls = []

    def fake_collect(target_date: str):
        calls.append(target_date)
        if target_date == "2026-10-04":
            return {"status": "AVAILABLE", "games": [], "sources": []}
        return {"status": "AVAILABLE", "games": [{"game_id": "G-1"}], "sources": []}

    monkeypatch.setattr(health, "collect_npb_pregame_context", fake_collect)
    probe_date, context = health._select_pregame_probe(date(2026, 10, 4), max_future_days=2)
    assert calls == ["2026-10-04", "2026-10-05"]
    assert probe_date == "2026-10-05"
    assert context["games"][0]["game_id"] == "G-1"


def test_select_pregame_probe_fails_closed_on_source_failure(monkeypatch):
    def fake_collect(target_date: str):
        return {
            "status": "SOURCE_FAILED",
            "games": [],
            "sources": [],
            "error": "synthetic source failure",
        }

    monkeypatch.setattr(health, "collect_npb_pregame_context", fake_collect)
    try:
        health._select_pregame_probe(date(2026, 10, 4), max_future_days=2)
    except RuntimeError as exc:
        assert "synthetic source failure" in str(exc)
    else:
        raise AssertionError("SOURCE_FAILED was not propagated fail-closed")


def test_verify_game_team_stats_covers_all_participating_teams(monkeypatch):
    calls = []

    def fake_verify(team: str, season: int):
        calls.append((team, season))
        return {"team": team, "season": season, "sources": {}}

    monkeypatch.setattr(health, "_verify_team_stats", fake_verify)
    games = [
        {"home": "阪神タイガース", "away": "横浜DeNAベイスターズ"},
        {"home": "阪神タイガース", "away": "広島東洋カープ"},
    ]
    got = health._verify_game_team_stats(games, 2026)
    assert got["status"] == "AVAILABLE"
    assert got["team_count"] == 3
    assert sorted(t for t, _ in calls) == sorted({
        "阪神タイガース",
        "横浜DeNAベイスターズ",
        "広島東洋カープ",
    })


def test_verify_team_stats_accepts_official_rows_without_embedded_stable_ids(monkeypatch):
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return (
            "<html>fixture</html>",
            "2026-10-04T00:00:00+00:00",
        )

    monkeypatch.setattr(health, "fetch_team_page", fake_fetch)
    monkeypatch.setattr(
        health,
        "parse_stats_page",
        lambda body, kind: (
            [
                {"player_name": "選手A", "player_id": None},
                {"player_name": "選手B", "player_id": None},
            ],
            "2026-10-03",
        ),
    )
    got = health._verify_team_stats("千葉ロッテマリーンズ", 2026)
    assert len(calls) == 3
    assert all(v["status"] == "AVAILABLE" for v in got["sources"].values())
    assert all(v["data_status"] == "AVAILABLE" for v in got["sources"].values())
    assert all(v["identity_status"] == "PARTIAL" for v in got["sources"].values())
    assert all(v["stable_player_id_rows"] == 0 for v in got["sources"].values())
