from data.pit_acquisition import _explicit_announcement


def test_explicit_starter_announcement_is_accepted():
    game = {"home_starter_announced_at": "2026-09-14T10:30:00+00:00"}
    assert _explicit_announcement(game, "home") == "2026-09-14T10:30:00+00:00"


def test_retrieved_at_is_not_treated_as_announcement_time():
    game = {
        "home": {"probablePitcher": {"fullName": "Example Pitcher"}},
        "observed_at": "2026-09-14T10:30:00+00:00",
    }
    assert _explicit_announcement(game, "home") is None


def test_invalid_announcement_timestamp_fails_closed():
    game = {"away_starter_announced_at": "not-a-timestamp"}
    assert _explicit_announcement(game, "away") is None
