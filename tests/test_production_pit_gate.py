from production_pit_gate import check_game

CUTOFF = "2026-09-19T12:00:00+00:00"


def row(level="OFFICIAL_ANNOUNCEMENT", ts="2026-09-19T10:00:00+00:00"):
    return {
        "event_id": "MLB:1",
        "league": "MLB",
        "home_starter": "Home Pitcher",
        "away_starter": "Away Pitcher",
        "home_starter_evidence_level": level,
        "away_starter_evidence_level": level,
        "home_starter_announced_at": ts,
        "away_starter_announced_at": ts,
        "source": "official",
    }


def test_strict_gate_accepts_two_official_announcements():
    ok, reason = check_game(row(), CUTOFF)
    assert ok and reason == "eligible"


def test_retrieval_only_is_rejected():
    ok, reason = check_game(row("RETRIEVAL_ONLY"), CUTOFF)
    assert not ok and "not_strictly_eligible" in reason


def test_future_announcement_is_rejected():
    ok, reason = check_game(row(ts="2026-09-19T13:00:00+00:00"), CUTOFF)
    assert not ok


def test_missing_starter_is_rejected():
    r = row()
    r["away_starter"] = ""
    ok, reason = check_game(r, CUTOFF)
    assert not ok and reason == "away_starter_missing"
