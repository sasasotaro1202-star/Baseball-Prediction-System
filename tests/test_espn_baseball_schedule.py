import json

from data.espn_baseball_schedule import LEAGUE_SLUGS, fetch_espn_scoreboard


class _Resp:
    def __init__(self, payload):
        self.payload = payload
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self):
        return json.dumps(self.payload).encode()


def test_espn_competition_slugs_cover_core_frontier():
    assert LEAGUE_SLUGS["MLB"] == "mlb"
    assert LEAGUE_SLUGS["NCAA"] == "college-baseball"
    assert "LIDOM" in LEAGUE_SLUGS


def test_probables_extract_nested_athlete(monkeypatch):
    payload = {
        "events": [{
            "id": "e1",
            "date": "2026-09-28T20:00Z",
            "name": "Away at Home",
            "competitions": [{
                "id": "e1",
                "playByPlayAvailable": True,
                "competitors": [
                    {"homeAway": "home", "team": {"id": "1", "displayName": "Home"}},
                    {"homeAway": "away", "team": {"id": "2", "displayName": "Away"}},
                ],
                "probables": [{
                    "playerId": 123,
                    "athlete": {"id": "123", "displayName": "Starter X"},
                    "team": {"id": "2"},
                }],
                "venue": {"fullName": "Test Park"},
            }],
            "status": {"type": {"name": "STATUS_SCHEDULED", "detail": "Scheduled"}},
        }],
    }
    monkeypatch.setattr("data.espn_baseball_schedule.urlopen", lambda *a, **k: _Resp(payload))
    out = fetch_espn_scoreboard("MLB", date="2026-09-28")
    assert out["event_count"] == 1
    assert out["events"][0]["probables"][0]["name"] == "Starter X"
    assert out["events"][0]["play_by_play_available"] is True