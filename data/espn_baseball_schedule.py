"""Generic ESPN public baseball scoreboard adapter (research-only).

Designed for discovery/validation, not production. It records the retrieval
boundary and preserves raw event fields without claiming that embedded fields
were public before prediction_time.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from typing import Any


LEAGUE_SLUGS = {
    "MLB": "mlb",
    "NCAA": "college-baseball",
    "WBC": "world-baseball-classic",
    "CARIBBEAN_SERIES": "caribbean-series",
    "LIDOM": "dominican-winter-league",
    "LVBP": "venezuelan-winter-league",
    "LBPRC": "puerto-rican-winter-league",
    "OLYMPICS_BASEBALL": "olympics-baseball",
}


def fetch_espn_scoreboard(
    competition: str = "MLB",
    *,
    date: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    if competition not in LEAGUE_SLUGS:
        raise ValueError(f"unsupported ESPN baseball competition: {competition}")
    params = {"limit": str(max(1, min(int(limit), 500)))}
    if date:
        compact = str(date).replace("-", "")
        if len(compact) != 8 or not compact.isdigit():
            raise ValueError("date must be YYYY-MM-DD or YYYYMMDD")
        params["dates"] = compact
    url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/{LEAGUE_SLUGS[competition]}/scoreboard?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "Baseball-Prediction-System/ESPNResearch"})
    with urlopen(req, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    retrieved_at = datetime.now(timezone.utc).isoformat()

    events: list[dict[str, Any]] = []
    for event in payload.get("events", []) or []:
        competitions = event.get("competitions") or []
        comp = competitions[0] if competitions else {}
        competitors = comp.get("competitors") or []
        teams = []
        for item in competitors:
            team = item.get("team") or {}
            teams.append({
                "team_id": team.get("id"),
                "team_name": team.get("displayName"),
                "abbreviation": team.get("abbreviation"),
                "home_away": item.get("homeAway"),
            })
        events.append({
            "event_id": event.get("id"),
            "name": event.get("name"),
            "date": event.get("date"),
            "status": ((event.get("status") or {}).get("type") or {}).get("name"),
            "teams": teams,
            "venue": ((comp.get("venue") or {}).get("fullName")),
            "probable_pitchers": [
                p.get("displayName")
                for p in (comp.get("probables") or [])
                if p.get("displayName")
            ],
        })

    return {
        "status": "EXECUTED",
        "competition": competition,
        "league_slug": LEAGUE_SLUGS[competition],
        "source_url": url,
        "retrieved_at": retrieved_at,
        "events": events,
        "event_count": len(events),
        "pit_status": "UNVERIFIED",
        "production_eligible": False,
        "raw_provider_note": "Embedded event fields are not assumed to have individual historical publication timestamps.",
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("competition", nargs="?", default="MLB")
    parser.add_argument("--date")
    args = parser.parse_args()
    print(json.dumps(fetch_espn_scoreboard(args.competition, date=args.date), ensure_ascii=False, indent=2))
