"""Japanese independent baseball schedule discovery adapters (research-only).

Targets official 2026 schedule/result pages for IBLJ Shikoku Island League plus
and Route-Inn BC League. The adapter discovers games and preserves status/source
metadata. Starter/lineup PIT and production eligibility remain separate gates.
"""
from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
import re
from typing import Any
from urllib.request import Request, urlopen


URLS = {
    "IBLJ": "https://data.iblj.co.jp/ibljdata/schedule/2026",
    "BCL": "https://www.bc-l-data.jp/schedule/2026",
}


class _RowParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        value = " ".join(data.split())
        if self._cell is not None and value:
            self._cell.append(value)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(" ".join(self._cell))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


_DATE_TIME = re.compile(r"^(\d{1,2}/\d{1,2})\s+(\d{1,2}:\d{2})$")
_SCORE = re.compile(r"^\d+\s*[-−]\s*\d+(?:\s+.*)?$")
_STATUS = re.compile(r"^(?:試合終了|中止|延期|中断|ノーゲーム|継続|未開始)$")


def _fetch(url: str) -> tuple[str, str]:
    req = Request(url, headers={"User-Agent": "Baseball-Prediction-System/IndependentDiscovery"})
    with urlopen(req, timeout=20) as response:
        return response.read().decode("utf-8", "ignore"), datetime.now(timezone.utc).isoformat()


def _row_to_game(cells: list[str], provider: str, source_url: str, retrieved_at: str) -> dict[str, Any] | None:
    clean = [" ".join(str(x).split()) for x in cells if " ".join(str(x).split())]
    date_time = next((x for x in clean if _DATE_TIME.match(x)), None)
    if not date_time:
        return None

    payload = [x for x in clean if x != date_time]
    status = next((x for x in payload if _STATUS.search(x) or _SCORE.match(x)), "UNVERIFIED")
    idx = payload.index(status) if status in payload else -1
    candidates = [x for i, x in enumerate(payload) if i != idx and not _STATUS.search(x) and not _SCORE.match(x)]
    if len(candidates) < 2:
        return None

    # Venue is typically the final cell; keep it when the row has enough fields.
    venue = candidates[-1] if len(candidates) >= 3 else None
    teams = candidates[:2] if venue else candidates[-2:]
    if len(teams) != 2 or not all(teams):
        return None

    date_part, time_part = _DATE_TIME.match(date_time).groups()
    month, day = map(int, date_part.split("/"))
    year = 2026
    game_status = "COMPLETED" if _SCORE.match(status) or status == "試合終了" else status
    return {
        "provider": provider,
        "competition_id": "JAPAN_INDEPENDENT",
        "event_date": f"{year:04d}-{month:02d}-{day:02d}",
        "start_time_local": time_part,
        "home_team": teams[0],
        "away_team": teams[1],
        "result_or_status": status,
        "status_normalized": game_status,
        "venue": venue,
        "source_url": source_url,
        "retrieved_at": retrieved_at,
        "pit_status": "UNVERIFIED",
        "availability_status": "DISCOVERED_NOT_PIT_VALIDATED",
    }


def discover(provider: str) -> dict[str, Any]:
    key = str(provider).upper()
    if key not in URLS:
        raise ValueError(f"unknown independent provider: {provider}")
    url = URLS[key]
    html, retrieved_at = _fetch(url)
    parser = _RowParser()
    parser.feed(html)
    games: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in parser.rows:
        game = _row_to_game(row, key, url, retrieved_at)
        if not game:
            continue
        sig = (game["event_date"], game["start_time_local"], game["home_team"], game["away_team"])
        if sig in seen:
            continue
        seen.add(sig)
        games.append(game)

    return {
        "status": "EXECUTED",
        "provider": key,
        "source_url": url,
        "retrieved_at": retrieved_at,
        "game_count": len(games),
        "games": games,
        "pit_status": "UNVERIFIED",
        "production_eligible": False,
    }


def discover_ibl_j() -> dict[str, Any]:
    return discover("IBLJ")


def discover_bcl() -> dict[str, Any]:
    return discover("BCL")


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=("IBLJ", "BCL"))
    args = parser.parse_args()
    print(json.dumps(discover(args.provider), ensure_ascii=False, indent=2))
