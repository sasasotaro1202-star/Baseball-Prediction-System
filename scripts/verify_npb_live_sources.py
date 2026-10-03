#!/usr/bin/env python3
"""Live verification of NPB official sources used by the pregame context layer.

This is an operational source-availability check, not a model-performance test.
It deliberately fails closed when an official page cannot be fetched or parsed.
"""
from __future__ import annotations

from datetime import date
from html.parser import HTMLParser
from pathlib import Path
import json
import re
import sys
from urllib.parse import urljoin

from core.http import request as http_request, session as http_session
from data.npb_roster_context import parse_roster_page, BASE_URL as NPB_BASE_URL
from data.npb_team_player_context import parse_stats_page
from data.npb_team_player_context import STATS_URL, URL_KIND, TEAM_SUFFIX
from data.npb_team_player_context import _fetch as fetch_team_page
from data.npb_team_player_context import parse_profile_page, _fetch as fetch_profile_page

ROSTER_INDEX_URL = NPB_BASE_URL + "/announcement/roster/"
ROSTER_PAGE_PATTERN = re.compile(r"/announcement/roster/roster_(\d{4})\.html")
SESSION = http_session(user_agent="Baseball-Prediction-System/npb-live-source-smoke")
OUTPUT = Path("results/npb_live_source_health.json")


class _RosterIndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href") or ""
        if ROSTER_PAGE_PATTERN.search(href):
            self.links.append(urljoin(NPB_BASE_URL, href))


def _fetch_text(url: str) -> str:
    response = http_request(SESSION, url, timeout=(8, 45), retries=4)
    encoding = (response.apparent_encoding or response.encoding or "utf-8").lower().replace("-", "_")
    if "shift_jis" in encoding or "cp932" in encoding or "shiftjis" in encoding:
        return response.content.decode("cp932", errors="strict")
    return response.content.decode(response.apparent_encoding or response.encoding or "utf-8", errors="strict")


def _latest_roster_url(today: date) -> tuple[str, str]:
    body = _fetch_text(ROSTER_INDEX_URL)
    parser = _RosterIndexParser()
    parser.feed(body)
    candidates: list[tuple[date, str]] = []
    for url in sorted(set(parser.links)):
        match = ROSTER_PAGE_PATTERN.search(url)
        if not match:
            continue
        mmdd = match.group(1)
        try:
            d = date(today.year, int(mmdd[:2]), int(mmdd[2:]))
        except ValueError:
            continue
        if d <= today:
            candidates.append((d, url))
    if not candidates:
        raise RuntimeError("official NPB roster index did not expose a past/current year date-scoped roster page")
    d, url = max(candidates, key=lambda item: item[0])
    return d.isoformat(), url


def _verify_stats(team: str, season: int) -> dict:
    results = {}
    suffix = TEAM_SUFFIX[team]
    for kind in ("batting", "pitching", "fielding"):
        url = STATS_URL.format(season=season, kind=URL_KIND[kind], team=suffix)
        body, observed = fetch_team_page(url)
        rows, as_of = parse_stats_page(body, kind)
        if not rows:
            raise RuntimeError(f"{kind} source returned zero parsed player rows: {url}")
        stable_ids = sum(1 for row in rows if row.get("player_id"))
        if stable_ids == 0:
            raise RuntimeError(f"{kind} source returned rows but no stable player ids: {url}")
        results[kind] = {
            "url": url,
            "status": "AVAILABLE",
            "parsed_rows": len(rows),
            "stable_player_id_rows": stable_ids,
            "source_as_of_date": as_of,
            "observed_at_utc": observed,
        }
    return results


def main() -> int:
    today = date.today()
    report: dict = {
        "schema_version": "npb-live-source-health-v1",
        "status": "UNKNOWN",
        "checked_date": today.isoformat(),
        "sources": {},
    }
    roster_date, roster_url = _latest_roster_url(today)
    roster_html = _fetch_text(roster_url)
    roster = parse_roster_page(roster_html, roster_date)
    team_rows = roster.get("teams") or {}
    if not team_rows:
        raise RuntimeError("official roster page parsed successfully but contained no team/player rows")
    first_team = sorted(team_rows)[0]
    first_player = (team_rows[first_team] or [None])[0]
    if not first_player or not first_player.get("player_url"):
        raise RuntimeError("official roster page did not yield a player profile URL")

    report["sources"]["npb_official_roster_status"] = {
        "status": "AVAILABLE",
        "url": roster_url,
        "target_date": roster_date,
        "player_count": roster["player_count"],
        "team_count": len(team_rows),
        "registered_today_count": roster.get("registered_today_count", 0),
        "removed_today_count": roster.get("removed_today_count", 0),
    }
    report["sources"]["npb_official_team_stats"] = _verify_stats("阪神タイガース", today.year)

    profile_body, profile_observed = fetch_profile_page(first_player["player_url"])
    profile = parse_profile_page(profile_body)
    if not profile:
        raise RuntimeError(f"official player profile parsed empty: {first_player['player_url']}")
    if not profile.get("position") and not profile.get("handedness"):
        raise RuntimeError(f"official player profile lacks expected identity/role fields: {first_player['player_url']}")
    report["sources"]["npb_official_player_page"] = {
        "status": "AVAILABLE",
        "url": first_player["player_url"],
        "player_id": first_player.get("player_id"),
        "player_name": first_player.get("player_name"),
        "observed_at_utc": profile_observed,
        "fields_available": sorted(profile.keys()),
    }

    report["status"] = "VERIFIED"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        failure = {
            "schema_version": "npb-live-source-health-v1",
            "status": "SOURCE_FAILED",
            "checked_date": date.today().isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        }
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        raise
