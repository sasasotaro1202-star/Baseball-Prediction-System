"""Official Aichi-Nagoya 2026 Asian Games baseball schedule adapter.

This adapter intentionally ingests only organizer-published matchup/schedule
information. It does not infer starters and therefore cannot make a production
prediction until separate PIT-safe official starter evidence is attached.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Iterable
import hashlib
import re
import requests


OFFICIAL_URL = "https://www.aichi-nagoya2026.org/ja/news-2050/"
SCHEDULE_URL = "https://www.aichi-nagoya2026.org/ja/sport/baseball/"
TIMEOUT = 20


@dataclass(frozen=True)
class AsianGamesGame:
    game_id: str
    date_jst: str
    time_jst: str
    visitor: str
    home: str
    venue: str
    source_url: str
    retrieved_at: str
    starter_evidence_status: str = "missing"


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            value = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            self._row.append(value)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def parse_matchup_rows(html: str) -> list[AsianGamesGame]:
    parser = _TextParser()
    parser.feed(html)
    games: list[AsianGamesGame] = []
    for row in parser.rows:
        text = " | ".join(_clean(x) for x in row if _clean(x))
        if not text or "Visitor" not in text and "vs." not in text:
            continue
        m = re.search(r"(9/2[1-7])\s+.*?(\d{1,2}:\d{2}).*?(BBL\d{2}).*?([A-Z]{3})\s+.*?([A-Z]{3})", text)
        if not m:
            continue
        date, time_jst, code, visitor, home = m.groups()
        month, day = date.split("/")
        venue = "Aichi-Nagoya 2026 organizer venue"
        game_hash = hashlib.sha256(f"{code}|{date}|{visitor}|{home}".encode()).hexdigest()[:16]
        games.append(AsianGamesGame(
            game_id=f"asian-games-2026-{code.lower()}-{game_hash}",
            date_jst=f"2026-{int(month):02d}-{int(day):02d}",
            time_jst=time_jst,
            visitor=visitor,
            home=home,
            venue=venue,
            source_url=OFFICIAL_URL,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        ))
    return games


def fetch_official_schedule() -> list[AsianGamesGame]:
    response = requests.get(SCHEDULE_URL, timeout=TIMEOUT, headers={"User-Agent": "Baseball-Prediction-System/1.0"})
    response.raise_for_status()
    games = parse_matchup_rows(response.text)
    if not games:
        raise RuntimeError("official Asian Games baseball page returned no parseable matchups")
    return games


def serialize_games(games: Iterable[AsianGamesGame]) -> list[dict]:
    return [asdict(g) for g in games]
