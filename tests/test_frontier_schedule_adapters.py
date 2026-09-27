import json

from data.kbo_public_schedule import _extract_game
from data.cpbl_public_schedule import discover_cpbl


class _Resp:
    def __init__(self, html): self.html = html
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return self.html.encode()


def test_kbo_extracts_time_score_team_layout():
    assert _extract_game(["18:30", "LG", "3", "SSG"]) == ("LG", "3") or _extract_game(["18:30", "LG", "VS", "SSG"]) == ("LG", "SSG")
    assert _extract_game(["LG vs SSG"]) == ("LG", "SSG")


def test_cpbl_discovery_preserves_source_boundary(monkeypatch):
    html = "<html><body>GAME GAME 未開始 進行中</body></html>"
    monkeypatch.setattr("data.cpbl_public_schedule.urlopen", lambda *a, **k: _Resp(html))
    out = discover_cpbl()
    assert out["status"] == "EXECUTED"
    assert out["game_token_count"] == 2
    assert out["pit_status"] == "UNVERIFIED"
    json.dumps(out, ensure_ascii=False)