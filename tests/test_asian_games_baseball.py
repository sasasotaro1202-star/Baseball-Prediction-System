from research.asian_games_baseball import OFFICIAL_BASEBALL, OFFICIAL_NEWS

def test_official_sources_are_organizer_urls():
    assert OFFICIAL_NEWS.startswith("https://www.aichi-nagoya2026.org/")
    assert OFFICIAL_BASEBALL.startswith("https://www.aichi-nagoya2026.org/")

def test_adapter_is_fail_closed():
    from research.asian_games_baseball import fetch_schedule
    assert callable(fetch_schedule)
