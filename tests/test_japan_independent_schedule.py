import json

from data.japan_independent_schedule import discover


class _Resp:
    def __init__(self, html): self.html = html
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return self.html.encode()


def test_iblj_and_bcl_urls_are_explicit():
    from data.japan_independent_schedule import URLS
    assert URLS["IBLJ"].endswith("/2026")
    assert URLS["BCL"].endswith("/2026")


def test_schedule_row_parser_discovers_game(monkeypatch):
    html = """<table><tr><td>9/28 18:00</td><td>徳島IS</td><td>2 - 3 試合終了</td><td>愛媛MP</td><td>球場A</td></tr></table>"""
    monkeypatch.setattr("data.japan_independent_schedule.urlopen", lambda *a, **k: _Resp(html))
    out = discover("IBLJ")
    assert out["status"] == "EXECUTED"
    assert out["game_count"] == 1
    game = out["games"][0]
    assert game["event_date"] == "2026-09-28"
    assert game["start_time_local"] == "18:00"
    assert game["home_team"] == "徳島IS"
    assert game["away_team"] == "愛媛MP"
    assert game["status_normalized"] == "COMPLETED"
    assert game["pit_status"] == "UNVERIFIED"