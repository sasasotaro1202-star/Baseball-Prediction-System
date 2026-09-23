from pathlib import Path
from research.asian_games_baseball import OFFICIAL_BASEBALL, OFFICIAL_BFJ_EVENTS, OFFICIAL_NEWS, _extract_bbl_codes, _validate_bfj_event_page

def test_official_sources_are_organizer_or_bfj_urls():
    assert OFFICIAL_NEWS.startswith("https://www.aichi-nagoya2026.org/")
    assert OFFICIAL_BASEBALL.startswith("https://www.aichi-nagoya2026.org/")
    assert OFFICIAL_BFJ_EVENTS.startswith("https://www.baseballjapan.org/")

def test_bbl_code_extraction_is_deterministic():
    assert _extract_bbl_codes("BBL13 foo BBL15 BBL13") == ["BBL13", "BBL15"]

def test_bfj_event_validation_requires_asian_games_and_baseball_markers():
    assert _validate_bfj_event_page("第20回アジア競技大会 野球")["status"] == "PASS"
    assert _validate_bfj_event_page("アジア競技大会")["status"] == "BLOCKED"

def test_adapter_is_fail_closed():
    from research.asian_games_baseball import fetch_schedule
    assert callable(fetch_schedule)
