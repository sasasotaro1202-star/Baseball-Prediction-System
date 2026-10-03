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
from production_npb import (
    NPB_STARTER_URL,
    fetch_text as fetch_starter_text,
    parse_official_starters_html,
    _starter_rows_sane,
)

ROSTER_INDEX_URL = NPB_BASE_URL + "/announcement/roster/"
ROSTER_PAGE_PATTERN = re.compile(r"/announcement/roster/roster_(\d{4}).html")
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
                r'href=["\']([^"\']*?/announcement/roster/roster_\d{4}.html)["\']',
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
    match = re.search(r"(20\d{2})年(\d{1,2})月(\d{1,2})日の出場選手登録", body)
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
        names = sum(1 for row in rows if str(row.get("player_name") or "").strip())
        if names == 0:
            raise RuntimeError(f"{kind} source parsed rows but no player names: {url}")
        identity_status = "COMPLETE" if stable_ids == len(rows) else "PARTIAL"
        result["sources"][kind] = {
            "status": "AVAILABLE",
            "data_status": "AVAILABLE",
            "identity_status": identity_status,
            "url": url,
            "parsed_rows": len(rows),
            "player_name_rows": names,
            "stable_player_id_rows": stable_ids,
            "stable_player_id_rate": stable_ids / len(rows),
            "source_as_of_date": as_of,
            "observed_at_utc": observed,
        }
    return result


def _verify_game_team_stats(games: list[dict[str, Any]], season: int) -> dict[str, Any]:
    teams = sorted({
        str(team).strip()
        for game in games
        for team in (game.get("home"), game.get("away"))
        if str(team).strip() in TEAM_SUFFIX
    })
    if not teams:
        raise RuntimeError("no canonical participating teams were resolved from live NPB games")
    verified: dict[str, Any] = {}
    for team in teams:
        verified[team] = _verify_team_stats(team, season)
    return {
        "status": "AVAILABLE",
        "team_count": len(verified),
        "teams": verified,
    }


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


def _verify_official_starters(target_date: str) -> list[dict[str, Any]]:
    """Verify the dedicated official NPB announced-starter page live."""
    url = NPB_STARTER_URL + "?_ts=" + str(int(datetime.now().timestamp()))
    body = fetch_starter_text(url)
    rows = parse_official_starters_html(body, target_date)
    if not rows or not _starter_rows_sane(rows):
        raise RuntimeError(
            f"official NPB starter source returned no structurally sane target-day rows: {url}"
        )
    for row in rows:
        required = ("home", "away", "home_starter", "away_starter", "official_start_time")
        if any(not str(row.get(key) or "").strip() for key in required):
            raise RuntimeError(f"official NPB starter row incomplete: {row!r}")
    return rows


def _select_pregame_probe(today_jst: date, max_future_days: int = 7) -> tuple[str, dict[str, Any]]:
    """Select a real scheduled-game probe without masking source failures.

    Today is always tried first. An empty schedule is not a source failure,
    so the health check may probe the next few calendar days. A collector
    SOURCE_FAILED state is never bypassed.
    """
    checked: list[dict[str, Any]] = []
    for offset in range(0, max_future_days + 1):
        probe_date = today_jst.fromordinal(today_jst.toordinal() + offset)
        context = collect_npb_pregame_context(probe_date.isoformat())
        status = str(context.get("status") or "UNKNOWN")
        games = context.get("games") or []
        checked.append({
            "date": probe_date.isoformat(),
            "status": status,
            "game_count": len(games),
        })
        if status == "SOURCE_FAILED":
            raise RuntimeError(
                f"NPB pregame source failed for probe date {probe_date.isoformat()}: "
                f"{context.get('error') or 'unknown error'}"
            )
        if status == "AVAILABLE" and games:
            return probe_date.isoformat(), context
    raise RuntimeError(
        "official NPB pregame source returned no scheduled games within probe window: "
        + json.dumps(checked, ensure_ascii=False, sort_keys=True)
    )


def main() -> int:
    checked_at = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
    today_jst = datetime.now(ZoneInfo("Asia/Tokyo")).date()

    report: dict[str, Any] = {
        "schema_version": "npb-live-source-health-v5",
        "status": "UNKNOWN",
        "checked_at_jst": checked_at,
        "checked_date_jst": today_jst.isoformat(),
        "sources": {},
        "failures": [],
    }

    # 1) Actual NPB game context: schedule + standings + weather.
    # A no-game calendar date is not a transport/parser failure. Probe today,
    # then upcoming dates, while preserving any SOURCE_FAILED result.
    probe_date, pregame = _select_pregame_probe(today_jst)
    games = pregame.get("games") or []
    source_status = pregame.get("sources") or []
    report["sources"]["npb_pregame_context"] = {
        "status": "AVAILABLE",
        "target_date": probe_date,
        "checked_date_jst": today_jst.isoformat(),
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

    # 2) Dedicated official announced-starter page. This is kept separate
    # from roster checks because starter timing is a PIT-critical source.
    starter_rows = _verify_official_starters(probe_date)
    report["sources"]["npb_official_announced_starter"] = {
        "status": "AVAILABLE",
        "url": NPB_STARTER_URL,
        "target_date": probe_date,
        "game_count": len(starter_rows),
        "games": [
            {
                "home": row.get("home"),
                "away": row.get("away"),
                "home_starter": row.get("home_starter"),
                "away_starter": row.get("away_starter"),
                "official_start_time": row.get("official_start_time"),
                "starter_evidence_status": row.get("starter_evidence_status"),
            }
            for row in starter_rows
        ],
    }

    # 3) Date-scoped official roster page, including registration transactions.
    roster_date, roster_url = _latest_roster_url(today_jst)
    roster_html = _fetch_text(roster_url)
    roster = parse_roster_page(roster_html, roster_date)
    if roster.get("player_count", 0) <= 0:
        raise RuntimeError("official roster page parsed zero players")
    game_teams = {
        str(team).strip()
        for game in games
        for team in (game.get("home"), game.get("away"))
        if str(team).strip() in TEAM_SUFFIX
    }
    roster = resolve_roster_player_ids(roster, teams=game_teams)
    identity = roster.get("identity_resolution") or {}
    identity_team_rates: dict[str, float | None] = {}
    for team in sorted(game_teams):
        rows = roster.get("teams", {}).get(team, []) or {}
        named = [p for p in rows if str(p.get("player_name") or "").strip()]
        resolved = [p for p in named if str(p.get("player_id") or "").strip()]
        identity_team_rates[team] = (len(resolved) / len(named)) if named else None
    # Operational usability gate: each participating team's roster must be
    # predominantly resolvable through the official player-search identity path.
    low_coverage = {
        team: rate for team, rate in identity_team_rates.items()
        if rate is None or rate < 0.80
    }
    if low_coverage:
        raise RuntimeError(
            "official NPB roster-to-player identity coverage below 80%: "
            + json.dumps(low_coverage, ensure_ascii=False, sort_keys=True)
        )
    report["sources"]["npb_official_roster_status"] = {
        "status": "AVAILABLE",
        "url": roster_url,
        "target_date": roster_date,
        "player_count": roster["player_count"],
        "team_count": len(roster.get("teams") or {}),
        "registered_today_count": roster.get("registered_today_count", 0),
        "removed_today_count": roster.get("removed_today_count", 0),
        "transaction_parser_status": roster.get("transaction_parser_status"),
        "identity_resolution_requested_count": int(identity.get("requested_count", 0) or 0),
        "identity_resolution_count": int(identity.get("resolved_count", 0) or 0),
        "identity_resolution_rate": identity.get("resolved_rate"),
        "identity_resolution_rate_by_participating_team": identity_team_rates,
    }
    report["sources"]["npb_official_player_search"] = {
        "status": "AVAILABLE" if identity.get("resolved_count", 0) else "SOURCE_FAILED",
        "requested_count": int(identity.get("requested_count", 0) or 0),
        "resolved_count": int(identity.get("resolved_count", 0) or 0),
        "ambiguous_count": int(identity.get("ambiguous_count", 0) or 0),
        "not_found_count": int(identity.get("not_found_count", 0) or 0),
        "source_failed_count": int(identity.get("source_failed_count", 0) or 0),
        "resolved_rate": identity.get("resolved_rate"),
        "source_id": "npb_official_player_search",
    }

    # 3) Pick an actual participating rostered team/player from the live page
    # and require deterministic stable identity resolution before calling the
    # identity portion of this source stack operationally healthy.
    teams = roster.get("teams") or {}
    candidate_teams = sorted(game_teams & set(teams))
    selected_team = next(
        (
            team for team in candidate_teams
            if any(str(p.get("player_id") or "").strip() and p.get("player_url") for p in teams[team])
        ),
        None,
    )
    if not selected_team:
        raise RuntimeError(
            "no participating team had a roster player with deterministically resolved stable identity: "
            + json.dumps(identity, ensure_ascii=False, sort_keys=True)
        )
    selected_player = sorted(
        (
            p for p in teams[selected_team]
            if str(p.get("player_id") or "").strip() and p.get("player_url")
        ),
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
    report["sources"]["npb_official_team_stats"] = _verify_game_team_stats(games, today_jst.year)

    # A stats page can be fully usable as data while lacking embedded stable IDs.
    # Treat identity coverage as a separate quality dimension; do not turn a
    # usable official stats table into SOURCE_FAILED solely because links are absent.
    #
    # The pregame collector exposes source-level health in its "sources" list;
    # require schedule, both standings leagues, and at least one weather record.
    weather_source = next((x for x in source_status if x.get("source_id") == "open_meteo_forecast"), None)
    if not weather_source or weather_source.get("status") != "AVAILABLE":
        raise RuntimeError("Open-Meteo weather source was not AVAILABLE in pregame context")

    # 4) All critical sources succeeded; expose exact source count and raw
    # evidence paths for artifact inspection.
    critical = [
        "npb_pregame_context",
        "npb_official_announced_starter",
        "npb_official_roster_status",
        "npb_official_player_search",
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
