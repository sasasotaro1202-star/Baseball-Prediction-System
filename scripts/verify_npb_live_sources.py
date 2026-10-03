#!/usr/bin/env python3
"""Live verification of NPB official sources.

This verifies that the sources used by the NPB pregame/player evidence layer
are reachable and parseable in a real GitHub Actions environment. It is not a
model-performance test. Any critical acquisition failure exits non-zero.
"""
from __future__ import annotations

from datetime import datetime, date
from pathlib import Path
import json
import re
from typing import Any
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

from core.http import request as http_request, session as http_session
from data.npb_pregame_context import collect_npb_pregame_context
from data.npb_roster_context import parse_roster_page, BASE_URL as NPB_BASE_URL
from data.npb_team_player_context import (
    TEAM_SUFFIX,
    STATS_URL,
    URL_KIND,
    _fetch as fetch_team_page,
    parse_profile_page,
    parse_stats_page,
)

ROSTER_INDEX_URL = NPB_BASE_URL + "/announcement/roster/"
ROSTER_PAGE_PATTERN = re.compile(r"/announcement/roster/roster_(d{4}).html")
SESSION = http_session(user_agent="Baseball-Prediction-System/npb-live-source-smoke")
OUTPUT = Path("results/npb_live_source_health.json")


class _RosterIndexParser:
    """Minimal parser for date-scoped official roster URLs."""

    def __init__(self) -> None:
        self.links: list[str] = []

    def feed(self, html: str) -> None:
        # Avoid coupling the health check to the project's larger roster parser.
        self.links = [
            urljoin(NPB_BASE_URL, match)
            for match in re.findall(
                r'href=["\']([^"\']*?/announcement/roster/roster_d{4}.html)["\']',
                html,
                flags=re.IGNORECASE,
            )
        ]


def _fetch_text(url: str) -> str:
    response = http_request(SESSION, url, timeout=(8, 45), retries=4)
    encoding = (response.apparent_encoding or response.encoding or "utf-8").lower().replace("-", "_")
    if "shift_jis" in encoding or "cp932" in encoding or "shiftjis" in encoding:
        return response.content.decode("cp932", errors="strict")
    return response.content.decode(
        response.apparent_encoding or response.encoding or "utf-8",
        errors="strict",
    )


def _latest_roster_url(today: date) -> tuple[str, str]:
    body = _fetch_text(ROSTER_INDEX_URL)
    match = re.search(r"(20d{2})年(d{1,2})月(d{1,2})日の出場選手登録", body)
    if match:
        d = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if d <= today:
            return d.isoformat(), urljoin(
                NPB_BASE_URL,
                f"/announcement/roster/roster_{d.month:02d}{d.day:02d}.html",
            )

    parser = _RosterIndexParser()
    parser.feed(body)
    candidates: list[tuple[date, str]] = []
    for url in sorted(set(parser.links)):
        m = ROSTER_PAGE_PATTERN.search(url)
        if not m:
            continue
        mmdd = m.group(1)
        try:
            d = date(today.year, int(mmdd[:2]), int(mmdd[2:]))
        except ValueError:
            continue
        if d <= today:
            candidates.append((d, url))
    if not candidates:
        raise RuntimeError("official NPB roster index exposed no usable current-year date-scoped page")
    d, url = max(candidates, key=lambda item: item[0])
    return d.isoformat(), url


def _verify_team_stats(team: str, season: int) -> dict[str, Any]:
    suffix = TEAM_SUFFIX[team]
    result: dict[str, Any] = {"team": team, "season": season, "sources": {}}
    for kind in ("batting", "pitching", "fielding"):
        url = STATS_URL.format(season=season, kind=URL_KIND[kind], team=suffix)
        body, observed = fetch_team_page(url)
        rows, as_of = parse_stats_page(body, kind)
        if not rows:
            raise RuntimeError(f"{kind} source parsed zero player rows: {url}")
        stable_ids = sum(1 for row in rows if row.get("player_id"))
        if stable_ids == 0:
            raise RuntimeError(f"{kind} source returned player rows but no stable player ids: {url}")
        result["sources"][kind] = {
            "status": "AVAILABLE",
            "url": url,
            "parsed_rows": len(rows),
            "stable_player_id_rows": stable_ids,
            "source_as_of_date": as_of,
            "observed_at_utc": observed,
        }
    return result


def _verify_player_profile(player_url: str, player_id: str, player_name: str) -> dict[str, Any]:
    body, observed = fetch_team_page(player_url)
    profile = parse_profile_page(body)
    if not profile:
        raise RuntimeError(f"official player profile parsed empty: {player_url}")
    expected = ("position", "handedness", "height_cm", "weight_kg", "birth_date", "career", "draft")
    available = [key for key in expected if profile.get(key) not in (None, "")]
    if not available:
        raise RuntimeError(f"official player profile lacks expected fields: {player_url}")
    return {
        "status": "AVAILABLE",
        "url": player_url,
        "player_id": player_id,
        "player_name": player_name,
        "observed_at_utc": observed,
        "fields_available": sorted(available),
    }


def main() -> int:
    checked_at = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
    today_jst = datetime.now(ZoneInfo("Asia/Tokyo")).date()

    report: dict[str, Any] = {
        "schema_version": "npb-live-source-health-v2",
        "status": "UNKNOWN",
        "checked_at_jst": checked_at,
        "checked_date_jst": today_jst.isoformat(),
        "sources": {},
        "failures": [],
    }

    # 1) Actual today's (JST) NPB game context: schedule + standings + weather.
    pregame = collect_npb_pregame_context(today_jst.isoformat())
    games = pregame.get("games") or []
    if not games:
        raise RuntimeError("pregame collector returned zero games for today's JST date")

    source_status = pregame.get("source_status") or []
    report["sources"]["npb_pregame_context"] = {
        "status": pregame.get("status", "UNKNOWN"),
        "target_date": today_jst.isoformat(),
        "game_count": len(games),
        "games": [
            {
                "game_id": g.get("game_id"),
                "home": g.get("home"),
                "away": g.get("away"),
                "official_start_time": g.get("official_start_time"),
                "venue": g.get("venue"),
                "standing_home_available": bool(g.get("standing_home")),
                "standing_away_available": bool(g.get("standing_away")),
                "weather_status": (g.get("weather") or {}).get("status"),
            }
            for g in games
        ],
        "source_status": source_status,
    }
    schedule_entries = [x for x in source_status if x.get("source_id") == "npb_official_game_schedule_context"]
    if not schedule_entries or schedule_entries[0].get("status") != "AVAILABLE":
        raise RuntimeError("official NPB schedule source was not AVAILABLE")
    standings = [x for x in source_status if str(x.get("source_id", "")).startswith("npb_official_standings_")]
    if len([x for x in standings if x.get("status") == "AVAILABLE"]) < 2:
        raise RuntimeError("both Central/Pacific official NPB standings sources were not AVAILABLE")
    weather_available = sum(
        1 for game in games if (game.get("weather") or {}).get("status") == "AVAILABLE"
    )
    report["weather_available_game_count"] = weather_available
    if weather_available == 0:
        raise RuntimeError("Open-Meteo weather source returned no AVAILABLE game weather records")

    # 2) Date-scoped official roster page, including registration transactions.
    roster_date, roster_url = _latest_roster_url(today_jst)
    roster_html = _fetch_text(roster_url)
    roster = parse_roster_page(roster_html, roster_date)
    if roster.get("player_count", 0) <= 0:
        raise RuntimeError("official roster page parsed zero players")
    report["sources"]["npb_official_roster_status"] = {
        "status": "AVAILABLE",
        "url": roster_url,
        "target_date": roster_date,
        "player_count": roster["player_count"],
        "team_count": len(roster.get("teams") or {}),
        "registered_today_count": roster.get("registered_today_count", 0),
        "removed_today_count": roster.get("removed_today_count", 0),
        "transaction_parser_status": roster.get("transaction_parser_status"),
    }

    # 3) Pick an actual rostered team/player from the live page and verify its
    # individual official profile and that team's three official stat tables.
    teams = roster.get("teams") or {}
    selected_team = next((team for team in sorted(teams) if teams[team]), None)
    if not selected_team:
        raise RuntimeError("no team with rostered players was parsed")
    selected_player = sorted(
        teams[selected_team],
        key=lambda p: (str(p.get("player_name") or ""), str(p.get("player_id") or "")),
    )[0]
    player_url = selected_player.get("player_url")
    if not player_url:
        raise RuntimeError("roster player lacked an official profile URL")
    report["sources"]["npb_official_player_page"] = _verify_player_profile(
        str(player_url),
        str(selected_player.get("player_id") or ""),
        str(selected_player.get("player_name") or ""),
    )
    report["sources"]["npb_official_team_stats"] = _verify_team_stats(selected_team, today_jst.year)

    # 4) All critical sources succeeded; expose exact source count and raw
    # evidence paths for artifact inspection.
    critical = [
        "npb_pregame_context",
        "npb_official_roster_status",
        "npb_official_player_page",
        "npb_official_team_stats",
    ]
    if any(report["sources"].get(name, {}).get("status") not in {"AVAILABLE", "VERIFIED"} for name in critical):
        raise RuntimeError("at least one critical NPB source did not verify")
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
            "schema_version": "npb-live-source-health-v2",
            "status": "SOURCE_FAILED",
            "checked_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        }
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        raise
