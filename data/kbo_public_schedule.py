"""KBO public schedule discovery adapter (research-only).

It discovers scheduled/completed KBO games from the public daily schedule page.
It does not infer starter announcements or production eligibility.
"""
from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any
from urllib.request import Request, urlopen


class _TableParser(HTMLParser):
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
        if self._cell is not None and data.strip():
            self._cell.append(data.strip())
    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(" ".join(self._cell))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def _fetch(url: str) -> tuple[str, str]:
    req = Request(url, headers={"User-Agent": "Baseball-Prediction-System/KBODiscovery"})
    with urlopen(req, timeout=20) as response:
        return response.read().decode("utf-8", "ignore"), datetime.now(timezone.utc).isoformat()


def _split_game(cell: str) -> tuple[str, str] | None:
    value = " ".join(str(cell).split())
    if not value:
        return None
    for sep in (" vs ", " VS ", " v ", " V ", "vs.", " VS. "):
        if sep in value:
            left, right = value.split(sep, 1)
            if left.strip() and right.strip():
                return left.strip(), right.strip()
    return None


def fetch_kbo_schedule(year: int, month: int) -> dict[str, Any]:
    url = "https://eng.koreabaseball.com/Schedule/DailySchedule.aspx"
    html, retrieved_at = _fetch(url)
    parser = _TableParser()
    parser.feed(html)
    rows: list[dict[str, Any]] = []
    current_date = None
    for cells in parser.rows:
        text = " | ".join(cells)
        if any(x in text.upper() for x in ("REGULAR", "PRESEASON", "POSTSEASON")):
            for token in text.split("|"):
                token = token.strip()
                if len(token) >= 8 and token[4:5] == "." and token[:4].isdigit():
                    current_date = token.replace(".", "-")
        if len(cells) < 3:
            continue
        game_pair = None
        for cell in cells:
            game_pair = _split_game(cell)
            if game_pair:
                break
        if not game_pair:
            continue
        home, away = game_pair
        time_value = next((x for x in cells if ":" in x and len(x.strip()) <= 5), "")
        venue = next((x for x in cells if x.strip().isupper() and len(x.strip()) >= 3), "")
        rows.append({
            "competition_id": "KBO",
            "league": "KBO",
            "home_team": home,
            "away_team": away,
            "event_date": current_date,
            "start_time_local": time_value or None,
            "venue": venue or None,
            "source_url": url,
            "retrieved_at": retrieved_at,
            "availability_status": "DISCOVERED_NOT_PIT_VALIDATED",
        })
    # Keep only structured rows; discovery failure is explicit, never zero.
    return {
        "status": "EXECUTED",
        "source_url": url,
        "retrieved_at": retrieved_at,
        "games": rows,
        "game_count": len(rows),
        "pit_status": "UNVERIFIED",
        "production_eligible": False,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(fetch_kbo_schedule(datetime.now().year, datetime.now().month), ensure_ascii=False, indent=2))
