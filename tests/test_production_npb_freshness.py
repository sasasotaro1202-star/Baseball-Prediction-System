from production_npb import official_starters


def _row(starter_home: str, starter_away: str):
    return [{
        "home": "横浜DeNAベイスターズ",
        "away": "広島東洋カープ",
        "home_starter": starter_home,
        "away_starter": starter_away,
        "confirmed_starters": True,
        "starter_evidence_status": "official_announced",
        "starter_source": "https://npb.jp/announcement/starter/",
        "official_start_time": "18:00",
    }]


def test_live_official_starters_override_existing_snapshot(monkeypatch):
    import production_npb as p

    calls = []

    monkeypatch.setattr(
        p,
        "_load_official_starter_snapshot",
        lambda target_date: _row("OLD-PITCHER", "OLD-PITCHER-2"),
    )
    monkeypatch.setattr(
        p,
        "fetch_text",
        lambda url: calls.append(url) or "LIVE-OFFICIAL-HTML",
    )
    monkeypatch.setattr(
        p,
        "parse_official_starters_html",
        lambda html, target_date: _row("LIVE-PITCHER", "LIVE-PITCHER-2"),
    )

    result = official_starters("2026-09-29")

    assert calls
    assert result[0]["home_starter"] == "LIVE-PITCHER"
    assert result[0]["away_starter"] == "LIVE-PITCHER-2"
