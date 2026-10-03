"""Date-scoped NPB first-team roster context.

The official NPB roster announcement page exposes date-specific registration/
de-registration events and a per-team 出場選手一覧. This collector records the
observed roster snapshot for the requested date without treating that snapshot
as historical PIT evidence.
"""
from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import hashlib
import json
import re
from typing import Any
from urllib.parse import urljoin

from core.http import request as http_request, session as http_session

BASE_URL = "https://npb.jp"
ROSTER_URL = BASE_URL + "/announcement/roster/roster_{mmdd}.html"
HTTP_SESSION = http_session(user_agent="Baseball-Prediction-System/npb-roster-context")

TEAM_NAMES = {
    "読売ジャイアンツ", "東京ヤクルトスワローズ", "中日ドラゴンズ",
    "広島東洋カープ", "阪神タイガース", "横浜DeNAベイスターズ",
    "北海道日本ハムファイターズ", "オリックス・バファローズ",
    "東北楽天ゴールデンイーグルス", "福岡ソフトバンクホークス",
    "千葉ロッテマリーンズ", "埼玉西武ライオンズ",
}
TEAM_NORMALIZE = {
    "巨人": "読売ジャイアンツ", "読売": "読売ジャイアンツ",
    "ヤクルト": "東京ヤクルトスワローズ", "中日": "中日ドラゴンズ",
    "広島": "広島東洋カープ", "阪神": "阪神タイガース",
    "DeNA": "横浜DeNAベイスターズ", "横浜DeNA": "横浜DeNAベイスターズ",
    "日本ハム": "北海道日本ハムファイターズ", "日ハム": "北海道日本ハムファイターズ",
    "オリックス": "オリックス・バファローズ", "楽天": "東北楽天ゴールデンイーグルス",
    "西武": "埼玉西武ライオンズ", "ロッテ": "千葉ロッテマリーンズ",
    "ソフトバンク": "福岡ソフトバンクホークス",
}

def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("　", " ")).strip()

def normalize_team(value: str) -> str:
    text = _clean(value)
    return TEAM_NORMALIZE.get(text, text)

class _RosterParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.current_team: str | None = None
        self._heading_depth = 0
        self._heading_parts: list[str] = []
        self._link_href: str | None = None
        self._link_parts: list[str] = []
        self.players: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {k: v or "" for k, v in attrs}
        if tag in {"h4", "h5"}:
            self._heading_depth = 1
            self._heading_parts = []
        elif self._heading_depth:
            self._heading_depth += 1
        elif tag == "a":
            href = attrs_dict.get("href", "")
            if re.search(r"/bis/players/\d+\.html$", href):
                self._link_href = urljoin(BASE_URL, href)
                self._link_parts = []

    def handle_data(self, data: str) -> None:
        value = _clean(data)
        if not value:
            return
        if self._heading_depth:
            self._heading_parts.append(value)
        elif self._link_href is not None:
            self._link_parts.append(value)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._heading_depth and tag in {"h4", "h5"}:
            heading = normalize_team(" ".join(self._heading_parts))
            self.current_team = heading if heading in TEAM_NAMES else None
            self._heading_depth = 0
            self._heading_parts = []
            return
        if tag == "a" and self._link_href is not None:
            if self.current_team:
                m = re.search(r"/bis/players/(\d+)\.html$", self._link_href)
                name = _clean(" ".join(self._link_parts))
                if m and name:
                    self.players.append({
                        "team": self.current_team,
                        "player_id": m.group(1),
                        "player_name": name,
                        "player_url": self._link_href,
                        "identity_status": "VERIFIED_STABLE_ID",
                    })
            self._link_href = None
            self._link_parts = []

def parse_roster_page(html: str, target_date: str) -> dict[str, Any]:
    parser = _RosterParser()
    parser.feed(html)
    by_team: dict[str, list[dict[str, Any]]] = {team: [] for team in sorted(TEAM_NAMES)}
    seen: set[tuple[str, str]] = set()
    for player in parser.players:
        key = (player["team"], player["player_id"])
        if key in seen:
            continue
        seen.add(key)
        by_team[player["team"]].append(player)
    for team in by_team:
        by_team[team] = sorted(by_team[team], key=lambda p: (str(p["player_name"]), str(p["player_id"])))
    total = sum(len(rows) for rows in by_team.values())
    if total == 0:
        raise RuntimeError(f"official NPB roster parser found no player rows for {target_date}")
    return {
        "schema_version": "npb-roster-context-v1",
        "target_date": target_date,
        "status": "AVAILABLE",
        "teams": {k: v for k, v in by_team.items() if v},
        "player_count": total,
        "historical_oos_consumption": "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
    }

def collect_npb_roster_context(target_date: str) -> dict[str, Any]:
    mmdd = target_date[5:].replace("-", "")
    url = ROSTER_URL.format(mmdd=mmdd)
    try:
        response = http_request(HTTP_SESSION, url, timeout=(8, 45), retries=4)
        enc = (response.apparent_encoding or response.encoding or "utf-8").lower().replace("-", "_")
        if "shift_jis" in enc or "cp932" in enc or "shiftjis" in enc:
            body = response.content.decode("cp932", errors="strict")
        else:
            body = response.content.decode(response.apparent_encoding or response.encoding or "utf-8", errors="strict")
        observed = datetime.now(timezone.utc).isoformat()
        snapshot = parse_roster_page(body, target_date)
        snapshot["source"] = {
            "source_id": "npb_official_roster_status",
            "url": url,
            "status": "AVAILABLE",
            "retrieved_at_utc": observed,
            "available_at_utc": observed,
            "published_at_utc": None,
            "revision_time_utc": None,
            "historical_oos_consumption": "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
        }
    except Exception as exc:
        snapshot = {
            "schema_version": "npb-roster-context-v1",
            "target_date": target_date,
            "status": "SOURCE_FAILED",
            "teams": {},
            "player_count": 0,
            "source": {
                "source_id": "npb_official_roster_status",
                "url": url,
                "status": "SOURCE_FAILED",
                "retrieved_at_utc": None,
                "available_at_utc": None,
                "published_at_utc": None,
                "revision_time_utc": None,
                "historical_oos_consumption": "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
                "error": f"{type(exc).__name__}: {exc}",
            },
            "error": f"{type(exc).__name__}: {exc}",
            "historical_oos_consumption": "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
        }
        return snapshot
    raw = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    snapshot["snapshot_id"] = hashlib.sha256(raw).hexdigest()
    return snapshot
