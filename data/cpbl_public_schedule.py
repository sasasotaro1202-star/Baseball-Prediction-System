"""CPBL public schedule discovery adapter (research-only).

The current CPBL advanced-statistics schedule exposes current and upcoming
first-team and farm games. This adapter discovers the game slate; it does not
infer starters, announcement timestamps or production eligibility.
"""
from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any
from urllib.request import Request, urlopen


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
    def handle_data(self, data: str) -> None:
        value = " ".join(data.split())
        if value:
            self.parts.append(value)


def _fetch(url: str) -> tuple[str, str]:
    req = Request(url, headers={"User-Agent": "Baseball-Prediction-System/CPBLDiscovery"})
    with urlopen(req, timeout=20) as response:
        return response.read().decode("utf-8", "ignore"), datetime.now(timezone.utc).isoformat()


def discover_cpbl(url: str = "https://stats.cpbl.com.tw/schedule/2026-A") -> dict[str, Any]:
    html, retrieved_at = _fetch(url)
    parser = _TextParser()
    parser.feed(html)
    text = " ".join(parser.parts)
    # Preserve the raw visible text as evidence. Detailed game parsing is kept
    # deliberately conservative because the CPBL page is application-rendered.
    game_tokens = text.count("GAME")
    not_started = text.count("未開始")
    in_progress = text.count("進行中")
    return {
        "status": "EXECUTED",
        "source_url": url,
        "retrieved_at": retrieved_at,
        "visible_text_bytes": len(html.encode("utf-8", "ignore")),
        "game_token_count": game_tokens,
        "not_started_count": not_started,
        "in_progress_count": in_progress,
        "availability_status": "DISCOVERED_NOT_PARSED" if game_tokens else "SOURCE_UNRESOLVED",
        "pit_status": "UNVERIFIED",
        "production_eligible": False,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(discover_cpbl(), ensure_ascii=False, indent=2))
