"""Date-scoped NPB first-team roster context.

The official NPB roster announcement page exposes date-specific registration/
de-registration events and a per-team 出場選手一覧. This collector records the
observed roster snapshot for the requested date without treating that snapshot
as historical PIT evidence.
"""
from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
import re
from typing import Any
from urllib.parse import quote, urljoin
import os
import unicodedata

from core.http import request as http_request, session as http_session

BASE_URL = "https://npb.jp"
ROSTER_URL = BASE_URL + "/announcement/roster/roster_{mmdd}.html"
PLAYER_SEARCH_URL = BASE_URL + "/bis/players/search/result?search_keyword={keyword}"
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


def _identity_key(value: Any) -> str:
    """Normalize names for exact identity resolution without fuzzy matching."""
    value = _clean(value)
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\\s+", "", value)


class _PlayerSearchParser(HTMLParser):
    """Extract official player-search result links and their visible labels."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._href: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href") or ""
        if re.search(r"/bis/players/\\d+\\.html(?:[?#].*)?$", href):
            self._href = urljoin(BASE_URL, href)
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            value = _clean(data)
            if value:
                self._parts.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            match = re.search(r"/bis/players/(\\d+)\\.html(?:[?#].*)?$", self._href)
            if match:
                self.results.append({
                    "player_id": match.group(1),
                    "player_url": self._href,
                    "label": _clean(" ".join(self._parts)),
                })
            self._href = None
            self._parts = []


def _parse_player_search_results(html: str) -> list[dict[str, str]]:
    parser = _PlayerSearchParser()
    parser.feed(html)
    return parser.results


def _resolve_one_player_name(player_name: str, team: str) -> list[dict[str, str]]:
    keyword = quote(_clean(player_name), safe="")
    url = PLAYER_SEARCH_URL.format(keyword=keyword)
    body = _fetch_text(url)
    expected_name = _identity_key(player_name)
    expected_team = _identity_key(team)
    candidates: list[dict[str, str]] = []
    for row in _parse_player_search_results(body):
        label_key = _identity_key(row.get("label"))
        if expected_name and expected_name not in label_key:
            continue
        if expected_team and expected_team not in label_key:
            continue
        candidates.append(row)
    unique = {row["player_id"]: row for row in candidates}
    return list(unique.values())


def resolve_roster_player_ids(
    snapshot: dict[str, Any],
    teams: set[str] | None = None,
) -> dict[str, Any]:
    """Resolve name-only roster rows through the official NPB player search.

    Resolution is exact on normalized full name + canonical team. Ambiguous,
    missing, or source-failed cases remain explicitly unresolved. This helper
    is intended for current/future evidence snapshots; historical OOS remains
    blocked by the snapshot's PIT policy.
    """
    if not isinstance(snapshot, dict):
        raise TypeError("roster snapshot must be a dict")
    selected_teams = {normalize_team(t) for t in (teams or set()) if _clean(t)}
    if not selected_teams:
        selected_teams = set(str(t) for t in (snapshot.get("teams") or {}) if _clean(t))
    try:
        max_per_team = max(0, int(os.getenv("NPB_ROSTER_ID_RESOLUTION_LIMIT_PER_TEAM", "40")))
    except Exception as exc:
        raise ValueError("NPB_ROSTER_ID_RESOLUTION_LIMIT_PER_TEAM must be an integer >= 0") from exc

    resolved = 0
    ambiguous = 0
    not_found = 0
    failed = 0
    requested = 0
    cache: dict[tuple[str, str], tuple[str, list[dict[str, str]] | Exception]] = {}
    resolver_source = {
        "source_id": "npb_official_player_search",
        "url_template": PLAYER_SEARCH_URL,
        "historical_oos_consumption": "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
    }

    for team, players in (snapshot.get("teams") or {}).items():
        if selected_teams and normalize_team(str(team)) not in selected_teams:
            continue
        if not isinstance(players, list):
            continue
        for player in players[:max_per_team]:
            if not isinstance(player, dict):
                continue
            requested += 1
            existing_id = str(player.get("player_id") or "").strip()
            if existing_id:
                player["identity_resolution_status"] = player.get("identity_resolution_status") or "SOURCE_STABLE_ID"
                continue
            name = _clean(player.get("player_name"))
            canonical_team = normalize_team(str(team))
            if not name:
                player["identity_resolution_status"] = "IDENTITY_NAME_MISSING"
                not_found += 1
                continue
            key = (canonical_team, _identity_key(name))
            cached = cache.get(key)
            if cached is None:
                try:
                    rows = _resolve_one_player_name(name, canonical_team)
                    cache[key] = ("OK", rows)
                except Exception as exc:
                    cache[key] = ("ERROR", exc)
            state, value = cache[key]
            if state == "ERROR":
                player["identity_resolution_status"] = "IDENTITY_SEARCH_FAILED"
                player["identity_resolution_source"] = dict(resolver_source, status="SOURCE_FAILED", error=f"{type(value).__name__}: {value}")
                failed += 1
                continue
            rows = value  # type: ignore[assignment]
            if len(rows) == 1:
                row = rows[0]
                player["player_id"] = row["player_id"]
                player["player_url"] = player.get("player_url") or row["player_url"]
                player["identity_status"] = "VERIFIED_STABLE_ID"
                player["identity_resolution_status"] = "RESOLVED_EXACT_OFFICIAL_PLAYER_SEARCH"
                player["identity_resolution_source"] = dict(resolver_source, status="AVAILABLE")
                resolved += 1
            elif len(rows) > 1:
                player["identity_resolution_status"] = "IDENTITY_AMBIGUOUS"
                player["identity_resolution_source"] = dict(resolver_source, status="AMBIGUOUS")
                ambiguous += 1
            else:
                player["identity_resolution_status"] = "IDENTITY_NOT_FOUND"
                player["identity_resolution_source"] = dict(resolver_source, status="NOT_FOUND")
                not_found += 1

    total = resolved + ambiguous + not_found + failed
    snapshot["identity_resolution"] = {
        "status": (
            "COMPLETE" if total and resolved == total
            else "PARTIAL" if resolved
            else "SOURCE_FAILED" if failed and not_found == 0 and ambiguous == 0
            else "UNRESOLVED"
        ),
        "teams": sorted(selected_teams),
        "requested_count": int(requested),
        "resolved_count": int(resolved),
        "ambiguous_count": int(ambiguous),
        "not_found_count": int(not_found),
        "source_failed_count": int(failed),
        "resolved_rate": float(resolved / total) if total else None,
        "source": resolver_source,
        "historical_oos_consumption": snapshot.get("historical_oos_consumption") or "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
    }
    return snapshot

class _RosterParser(HTMLParser):
    """Parse date-scoped first-team roster rows plus registration transactions."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.current_team: str | None = None
        self.transaction_section: str | None = None
        self.roster_section = False
        self._heading_depth = 0
        self._heading_parts: list[str] = []
        self._link_href: str | None = None
        self._link_parts: list[str] = []
        self._row_active = False
        self._cell_parts: list[str] | None = None
        self._row_cells: list[str] = []
        self._row_player_id: str | None = None
        self._row_player_name: str = ""
        self._row_player_url: str | None = None
        self.players: list[dict[str, Any]] = []
        self.transactions: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {k: v or "" for k, v in attrs}
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_depth = 1
            self._heading_parts = []
            return
        if tag == "tr":
            self._row_active = True
            self._row_cells = []
            self._cell_parts = None
            self._row_player_id = None
            self._row_player_name = ""
            self._row_player_url = None
            return
        if self._row_active and tag in {"td", "th"}:
            self._cell_parts = []
            return
        if tag == "a":
            href = attrs_dict.get("href", "")
            if re.search(r"/bis/players/\d+\.html(?:[?#].*)?$", href):
                self._link_href = urljoin(BASE_URL, href)
                self._link_parts = []

    def handle_data(self, data: str) -> None:
        value = _clean(data)
        if not value:
            return
        if self._heading_depth:
            self._heading_parts.append(value)
            return
        if self._cell_parts is not None:
            self._cell_parts.append(value)
            return
        if self._link_href is not None:
            self._link_parts.append(value)

    @staticmethod
    def _transaction_team(cells: list[str], fallback: str | None) -> str | None:
        for cell in cells:
            team = normalize_team(cell)
            if team in TEAM_NAMES:
                return team
        return fallback

    @staticmethod
    def _transaction_position(cells: list[str]) -> str | None:
        for cell in cells:
            value = _clean(cell)
            if re.search(r"(投手|捕手|内野手|外野手)", value):
                return value
        return None

    @staticmethod
    def _transaction_number(cells: list[str]) -> str | None:
        for cell in cells:
            value = _clean(cell)
            if re.fullmatch(r"\d{1,3}", value):
                return value
        return None

    def _finish_transaction_row(self) -> None:
        if not self.transaction_section or not self._row_cells:
            return
        cells = [_clean(x) for x in self._row_cells if _clean(x)]
        if not cells or all(x in {"なし", "-", "－"} for x in cells):
            return
        team = self._transaction_team(cells, self.current_team)
        name = _clean(self._row_player_name or (cells[-1] if cells else ""))
        if not team or not name or name in {"なし", "-", "－"}:
            return
        self.transactions.append(
            {
                "transaction_status": self.transaction_section,
                "team": team,
                "player_id": self._row_player_id,
                "player_name": name,
                "player_url": self._row_player_url,
                "position": self._transaction_position(cells),
                "uniform_number": self._transaction_number(cells),
                "identity_status": (
                    "VERIFIED_STABLE_ID" if self._row_player_id else "NAME_ONLY_UNVERIFIED"
                ),
            }
        )

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self._heading_depth:
            heading = normalize_team(" ".join(self._heading_parts))
            if heading in TEAM_NAMES:
                self.current_team = heading
            elif "出場選手登録抹消" in heading:
                self.transaction_section = "REMOVED"
                self.roster_section = False
            elif heading == "出場選手登録":
                self.transaction_section = "REGISTERED"
                self.roster_section = False
            elif heading == "出場選手一覧":
                self.transaction_section = None
                self.roster_section = True
            self._heading_depth = 0
            self._heading_parts = []
            return

        if tag in {"td", "th"} and self._cell_parts is not None and self._row_active:
            self._row_cells.append(_clean(" ".join(self._cell_parts)))
            self._cell_parts = None
            return

        if tag == "a" and self._link_href is not None:
            match = re.search(r"/bis/players/(\d+)\.html(?:[?#].*)?$", self._link_href)
            name = _clean(" ".join(self._link_parts))
            if self._row_active and match:
                self._row_player_id = match.group(1)
                self._row_player_name = name
                self._row_player_url = self._link_href
            elif self.current_team and match and name:
                self.players.append(
                    {
                        "team": self.current_team,
                        "player_id": match.group(1),
                        "player_name": name,
                        "player_url": self._link_href,
                        "identity_status": "VERIFIED_STABLE_ID",
                    }
                )
            self._link_href = None
            self._link_parts = []
            return

        if tag == "tr" and self._row_active:
            self._finish_transaction_row()
            if self.roster_section and self.current_team and self._row_cells:
                cells = [_clean(x) for x in self._row_cells if _clean(x)]
                # Official NPB roster rows are position / uniform number / name.
                # These rows may not expose an <a> player link, so preserve the
                # name-only identity as UNVERIFIED instead of inventing a stable id.
                position = next((x for x in cells if re.search(r"(投手|捕手|内野手|外野手)", x)), None)
                number = next((x for x in cells if re.fullmatch(r"\d{1,3}", x)), None)
                name = _clean(self._row_player_name or (cells[-1] if cells else ""))
                if position and number and name and name not in {"選手名", "なし", "-", "－"}:
                    self.players.append({
                        "team": self.current_team,
                        "player_id": self._row_player_id,
                        "player_name": name,
                        "player_url": self._row_player_url,
                        "position": position,
                        "uniform_number": number,
                        "identity_status": (
                            "VERIFIED_STABLE_ID" if self._row_player_id else "NAME_ONLY_UNVERIFIED"
                        ),
                    })
            self._row_active = False
            self._cell_parts = None
            self._row_cells = []
            self._row_player_id = None
            self._row_player_name = ""
            self._row_player_url = None

def parse_roster_page(html: str, target_date: str) -> dict[str, Any]:
    parser = _RosterParser()
    parser.feed(html)
    by_team: dict[str, list[dict[str, Any]]] = {team: [] for team in sorted(TEAM_NAMES)}
    seen: set[tuple[str, str]] = set()
    for player in parser.players:
        identity = str(player.get("player_id") or "").strip()
        name = _clean(player.get("player_name"))
        number = _clean(player.get("uniform_number"))
        key = (str(player.get("team") or ""), identity or name, number)
        if key in seen:
            continue
        seen.add(key)
        by_team[player["team"]].append(player)
    for team in by_team:
        by_team[team] = sorted(by_team[team], key=lambda p: (str(p["player_name"]), str(p["player_id"])))
    total = sum(len(rows) for rows in by_team.values())
    if total == 0:
        raise RuntimeError(f"official NPB roster parser found no player rows for {target_date}")
    transactions = parser.transactions
    registered_today = [
        row for row in transactions if row.get("transaction_status") == "REGISTERED"
    ]
    removed_today = [
        row for row in transactions if row.get("transaction_status") == "REMOVED"
    ]
    tx_by_key: dict[tuple[str, str], set[str]] = {}
    for row in transactions:
        pid = str(row.get("player_id") or "").strip()
        team = normalize_team(str(row.get("team") or ""))
        if not pid or team not in TEAM_NAMES:
            continue
        tx_by_key.setdefault((team, pid), set()).add(str(row.get("transaction_status") or "UNKNOWN"))
    for team, players in by_team.items():
        for player in players:
            statuses = tx_by_key.get((team, str(player.get("player_id") or "")), set())
            if str(player.get("player_id") or "").strip() == "":
                player["roster_transaction_status"] = "IDENTITY_UNVERIFIED"
            elif statuses == {"REGISTERED"}:
                player["roster_transaction_status"] = "REGISTERED_TODAY"
            elif statuses == {"REMOVED"}:
                player["roster_transaction_status"] = "REMOVED_TODAY"
            elif statuses:
                player["roster_transaction_status"] = "TRANSACTION_CONFLICT"
            else:
                player["roster_transaction_status"] = "NO_TRANSACTION_RECORDED"
    transaction_parser_status = "AVAILABLE"
    if transactions and not all(row.get("player_id") for row in transactions):
        transaction_parser_status = "PARTIAL_STABLE_ID"
    return {
        "schema_version": "npb-roster-context-v1",
        "target_date": target_date,
        "status": "AVAILABLE",
        "teams": {k: v for k, v in by_team.items() if v},
        "player_count": total,
        "transactions": transactions,
        "registered_today": registered_today,
        "removed_today": removed_today,
        "registered_today_count": len(registered_today),
        "removed_today_count": len(removed_today),
        "transaction_parser_status": transaction_parser_status,
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
