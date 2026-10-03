#!/usr/bin/env python3
"""Live verification of NPB official sources used by the pregame context layer.

This is an operational source-availability check, not a model-performance test.
It deliberately fails closed when an official page cannot be fetched or parsed.
"""
from __future__ import annotations

from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
import json
import re
import sys
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

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
    # The official landing page exposes the currently selected announcement date
    # in visible text. Prefer that over guessing from navigation anchors.
    match = re.search(r"(20\d{2})年(\d{1,2})月(\d{1,2})日の出場選手登録", body)
    if match:
        d = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if d <= today:
            return d.isoformat(), urljoin(NPB_BASE_URL, f"/announcement/roster/roster_{d.month:02d}{d.day:02d}.html")
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
