"""Team-level NPB player context collector.

Fetches official NPB regular-season individual batting, pitching, and fielding
tables for the two teams in a target game. The snapshot is evidence/context only:
current-page aggregates are not treated as historical PIT evidence.

The collector keeps stable player ids when the official stats page exposes them,
preserves source-level timestamps, and distinguishes unavailable/source-failed
data instead of converting it to zero.
"""
from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
import os
import hashlib
import json
import re
from typing import Any
from urllib.parse import urljoin

from core.http import request as http_request, session as http_session

BASE_URL = "https://npb.jp"
STATS_URL = BASE_URL + "/bis/{season}/stats/{kind}_{team}.html"
HTTP_SESSION = http_session(user_agent="Baseball-Prediction-System/npb-team-player-context")

TEAM_SUFFIX = {
    "読売ジャイアンツ": "g",
    "東京ヤクルトスワローズ": "s",
    "中日ドラゴンズ": "d",
    "広島東洋カープ": "c",
    "阪神タイガース": "t",
    "横浜DeNAベイスターズ": "db",
    "北海道日本ハムファイターズ": "f",
    "オリックス・バファローズ": "b",
    "東北楽天ゴールデンイーグルス": "e",
    "福岡ソフトバンクホークス": "h",
    "千葉ロッテマリーンズ": "m",
    "埼玉西武ライオンズ": "l",
}
TEAM_NORMALIZE = {
    "巨人": "読売ジャイアンツ", "読売": "読売ジャイアンツ",
    "ヤクルト": "東京ヤクルトスワローズ", "東京ヤクルト": "東京ヤクルトスワローズ",
    "中日": "中日ドラゴンズ", "広島": "広島東洋カープ",
    "阪神": "阪神タイガース", "DeNA": "横浜DeNAベイスターズ",
    "横浜DeNA": "横浜DeNAベイスターズ", "ＤｅＮＡ": "横浜DeNAベイスターズ",
    "日本ハム": "北海道日本ハムファイターズ", "日ハム": "北海道日本ハムファイターズ",
    "北海道日本ハム": "北海道日本ハムファイターズ",
    "オリックス": "オリックス・バファローズ", "楽天": "東北楽天ゴールデンイーグルス",
    "東北楽天": "東北楽天ゴールデンイーグルス", "西武": "埼玉西武ライオンズ",
    "ロッテ": "千葉ロッテマリーンズ", "千葉ロッテ": "千葉ロッテマリーンズ",
    "ソフトバンク": "福岡ソフトバンクホークス",
    "福岡ソフトバンク": "福岡ソフトバンクホークス",
}

STAT_KIND = ("batting", "pitching", "fielding")
URL_KIND = {"batting": "idb1", "pitching": "idp1", "fielding": "idf1"}


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("　", " ")).strip()


def normalize_team(team: str) -> str:
    value = _clean(team)
    return TEAM_NORMALIZE.get(value, value)


def _fetch(url: str) -> tuple[str, str]:
    response = http_request(HTTP_SESSION, url, timeout=(8, 45), retries=4)
    enc = (response.apparent_encoding or response.encoding or "utf-8").lower().replace("-", "_")
    if "shift_jis" in enc or "cp932" in enc or "shiftjis" in enc:
        body = response.content.decode("cp932", errors="strict")
    else:
        body = response.content.decode(
            response.apparent_encoding or response.encoding or "utf-8",
            errors="strict",
        )
    return body, datetime.now(timezone.utc).isoformat()


def _num(value: Any) -> float | None:
    text = _clean(value).replace(",", "")
    if text in {"", "-", "－", "―"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _as_of_date(html: str) -> str | None:
    match = re.search(r"(20\d{2})年(\d{1,2})月(\d{1,2})日\s*現在", html)
    if not match:
        return None
    return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"


class _StatsParser(HTMLParser):
    """Extract section headings, table rows, and player-page links."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.headings: list[str] = []
        self._heading_depth = 0
        self._heading_parts: list[str] = []
        self.rows: list[dict[str, Any]] = []
        self._table = False
        self._row: list[dict[str, Any]] | None = None
        self._cell: dict[str, Any] | None = None
        self._headers: list[str] = []
        self._player_href: str | None = None
        self._section: str | None = None
        self._links_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {k: v or "" for k, v in attrs}
        if tag in {"h2", "h3", "h4", "h5", "h6"}:
            self._heading_depth = 1
            self._heading_parts = []
            return
        if self._heading_depth:
            self._heading_depth += 1
            return
        if tag == "table":
            self._table = True
            return
        if not self._table:
            return
        if tag == "tr":
            self._row = []
            return
        if tag in {"td", "th"} and self._row is not None:
            self._cell = {"text": [], "href": None, "tag": tag}
            return
        if tag == "a" and self._cell is not None:
            href = attrs_dict.get("href", "")
            if re.search(r"/bis/players/\d+\.html$", href):
                self._cell["href"] = urljoin(BASE_URL, href)

    def handle_data(self, data: str) -> None:
        value = _clean(data)
        if self._heading_depth:
            if value:
                self._heading_parts.append(value)
            return
        if self._cell is not None and value:
            self._cell["text"].append(value)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._heading_depth:
            if tag in {"h2", "h3", "h4", "h5", "h6"}:
                heading = _clean(" ".join(self._heading_parts))
                if heading:
                    self.headings.append(heading)
                    if heading not in {"2026年度", "2026年度 公式戦成績"}:
                        self._section = heading
                self._heading_depth = 0
                self._heading_parts = []
            return
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append({
                "text": _clean(" ".join(self._cell["text"])),
                "href": self._cell["href"],
                "tag": self._cell["tag"],
            })
            self._cell = None
            return
        if tag == "tr" and self._row is not None:
            values = [cell["text"] for cell in self._row]
            links = [cell["href"] for cell in self._row if cell["href"]]
            if any(cell["tag"] == "th" for cell in self._row):
                self._headers = values
            elif self._headers and values:
                record = dict(zip(self._headers, values))
                if values:
                    record["_player_name_raw"] = values[0]
                if links:
                    record["_player_url"] = links[0]
                    m = re.search(r"/bis/players/(\d+)\.html$", links[0])
                    record["_player_id"] = m.group(1) if m else None
                record["_section"] = self._section
                self.rows.append(record)
            self._row = None
            return
        if tag == "table":
            self._table = False
            self._headers = []
            self._row = None
            self._cell = None


def parse_stats_page(html: str, kind: str) -> tuple[list[dict[str, Any]], str | None]:
    if kind not in STAT_KIND:
        raise ValueError(f"unsupported stats kind: {kind}")
    parser = _StatsParser()
    parser.feed(html)
    rows = []
    for row in parser.rows:
        name = _clean(row.pop("_player_name_raw", ""))
        if not name:
            continue
        name = re.sub(r"^[*+]+", "", name).strip()
        row["player_name"] = name
        row["player_id"] = row.pop("_player_id", None)
        row["player_url"] = row.pop("_player_url", None)
        if kind == "fielding":
            row["position"] = row.get("_section")
        row.pop("_section", None)
        rows.append(row)
    return rows, _as_of_date(html)


def _find_num(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in row:
            value = _num(row[key])
            if value is not None:
                return value
    return None


def _derive_batting(row: dict[str, Any]) -> dict[str, float]:
    pa = _find_num(row, "打席")
    ab = _find_num(row, "打数")
    h = _find_num(row, "安打")
    doubles = _find_num(row, "二塁打")
    triples = _find_num(row, "三塁打")
    hr = _find_num(row, "本塁打")
    bb = _find_num(row, "四球")
    hbp = _find_num(row, "死球")
    so = _find_num(row, "三振")
    sb = _find_num(row, "盗塁")
    cs = _find_num(row, "盗塁刺")
    out: dict[str, float] = {}
    if pa and pa > 0:
        if bb is not None:
            out["bb_pct"] = bb / pa
        if so is not None:
            out["k_pct"] = so / pa
        if hr is not None:
            out["hr_per_pa"] = hr / pa
        if hbp is not None:
            out["hbp_pct"] = hbp / pa
        if sb is not None or cs is not None:
            out["sb_attempt_rate"] = ((sb or 0.0) + (cs or 0.0)) / pa
    if ab and ab > 0 and h is not None:
        singles = h - (doubles or 0.0) - (triples or 0.0) - (hr or 0.0)
        total_bases = singles + 2.0 * (doubles or 0.0) + 3.0 * (triples or 0.0) + 4.0 * (hr or 0.0)
        avg = h / ab
        slg = total_bases / ab
        out["avg"] = avg
        out["slg"] = slg
        out["iso_from_totals"] = max(0.0, slg - avg)
        denom = ab + (bb or 0.0) + (hbp or 0.0) + (_find_num(row, "犠飛") or 0.0)
        if denom > 0:
            out["obp_from_totals"] = (h + (bb or 0.0) + (hbp or 0.0)) / denom
            out["ops_from_totals"] = out["obp_from_totals"] + slg
        if hr is not None:
            out["hr_per_ab"] = hr / ab
    if pa and pa > 0:
        runs = _find_num(row, "得点", "得点数")
        rbi = _find_num(row, "打点", "RBI")
        if runs is not None:
            out["runs_per_pa"] = runs / pa
        if rbi is not None:
            out["rbi_per_pa"] = rbi / pa
        xbh = sum(x or 0.0 for x in (doubles, triples, hr))
        if xbh:
            out["extra_base_hits_per_pa"] = xbh / pa
        games = _find_num(row, "試合")
        if games and games > 0:
            out["pa_per_game"] = pa / games
    if sb is not None or cs is not None:
        attempts = (sb or 0.0) + (cs or 0.0)
        if attempts > 0:
            out["sb_success_rate"] = (sb or 0.0) / attempts
    if bb is not None and so is not None and so > 0:
        out["bb_k_ratio"] = bb / so
    if h is not None and h > 0 and (doubles is not None or triples is not None or hr is not None):
        out["extra_base_hit_rate"] = ((doubles or 0.0) + (triples or 0.0) + (hr or 0.0)) / h
    return out


def _derive_pitching(row: dict[str, Any]) -> dict[str, float]:
    ip = _find_num(row, "投球回")
    so = _find_num(row, "三振")
    bb = _find_num(row, "四球")
    hr = _find_num(row, "本塁打")
    h = _find_num(row, "安打")
    tbf = _find_num(row, "打者")
    out: dict[str, float] = {}
    if ip and ip > 0:
        if so is not None:
            out["k9"] = 9.0 * so / ip
        if bb is not None:
            out["bb9"] = 9.0 * bb / ip
        if hr is not None:
            out["hr9"] = 9.0 * hr / ip
        if h is not None and bb is not None:
            out["whip"] = (h + bb) / ip
        games = _find_num(row, "登板")
        if games and games > 0:
            out["ip_per_game"] = ip / games
        starts = _find_num(row, "先発")
        if starts is not None and games and games > 0:
            out["start_share"] = starts / games
    if tbf and tbf > 0:
        if so is not None:
            out["k_pct"] = so / tbf
        if bb is not None:
            out["bb_pct"] = bb / tbf
        if hr is not None:
            out["hr_pct"] = hr / tbf
        if so is not None and bb is not None:
            out["k_minus_bb_pct"] = (so - bb) / tbf
        if so is not None and bb is not None and bb > 0:
            out["k_bb_ratio"] = so / bb
    starts = _find_num(row, "先発")
    games = _find_num(row, "登板")
    wins = _find_num(row, "勝")
    losses = _find_num(row, "敗")
    if wins is not None and losses is not None and (wins + losses) > 0:
        out["decision_win_rate"] = wins / (wins + losses)
    if starts is not None and games and games > 0:
        out["start_share"] = starts / games
    return out

def _derive_fielding(row: dict[str, Any]) -> dict[str, float]:
    games = _find_num(row, "試合")
    errors = _find_num(row, "失策")
    putouts = _find_num(row, "刺殺")
    assists = _find_num(row, "補殺")
    double_plays = _find_num(row, "併殺")
    out: dict[str, float] = {}
    chances = sum(x or 0.0 for x in (putouts, assists, errors))
    if chances > 0 and errors is not None:
        out["error_rate"] = errors / chances
    if games and games > 0:
        if chances > 0:
            out["chances_per_game"] = chances / games
        if double_plays is not None:
            out["double_plays_per_game"] = double_plays / games
    return out


def _merge_players(team: str, pages: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for kind in STAT_KIND:
        payload = pages.get(kind) or {}
        for raw in payload.get("rows", []):
            player_id = str(raw.get("player_id") or "").strip()
            name = _clean(raw.get("player_name"))
            key = f"id:{player_id}" if player_id else f"name:{name}"
            item = merged.setdefault(
                key,
                {
                    "team": team,
                    "player_id": player_id or None,
                    "identity_status": "VERIFIED_STABLE_ID" if player_id else "NAME_ONLY_UNVERIFIED",
                    "player_name": name,
                    "player_url": raw.get("player_url"),
                    "position": raw.get("position"),
                    "batting": None,
                    "batting_derived": {},
                    "pitching": None,
                    "pitching_derived": {},
                    "fielding": [],
                    "fielding_derived": [],
                    "profile": None,
                    "profile_status": "NOT_SELECTED",
                },
            )
            if raw.get("position") and not item.get("position"):
                item["position"] = raw["position"]
            if raw.get("player_url") and not item.get("player_url"):
                item["player_url"] = raw["player_url"]
            if kind == "batting":
                item["batting"] = {k: v for k, v in raw.items() if not k.startswith("_") and k not in {"player_name", "player_id", "player_url", "position"}}
                item["batting_derived"] = _derive_batting(raw)
            elif kind == "pitching":
                item["pitching"] = {k: v for k, v in raw.items() if not k.startswith("_") and k not in {"player_name", "player_id", "player_url", "position"}}
                item["pitching_derived"] = _derive_pitching(raw)
            else:
                field = {k: v for k, v in raw.items() if not k.startswith("_") and k not in {"player_name", "player_id", "player_url"}}
                item["fielding"].append(field)
                item["fielding_derived"].append(_derive_fielding(raw))
    players = sorted(merged.values(), key=lambda x: (str(x.get("player_name") or ""), str(x.get("player_id") or "")))
    players, _ = _enrich_profiles(players)
    return players


class _ProfileParser(HTMLParser):
    """Parse label/value pairs from an official NPB personal profile page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row = []
            self._cell = None
        elif tag in {"th", "td"} and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            value = _clean(data)
            if value:
                self._cell.append(value)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"th", "td"} and self._cell is not None and self._row is not None:
            value = _clean(" ".join(self._cell))
            self._row.append(value)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None
            self._cell = None


def parse_profile_page(html: str) -> dict[str, Any]:
    parser = _ProfileParser()
    parser.feed(html)
    fields: dict[str, str] = {}
    for row in parser.rows:
        if len(row) < 2:
            continue
        # Support both two-cell label/value rows and compact four-cell
        # label/value/label/value profile tables.
        for i in range(0, len(row) - 1, 2):
            label = _clean(row[i])
            value = _clean(row[i + 1])
            if label and value and label not in fields:
                fields[label] = value
    handedness = fields.get("投打") or fields.get("投打ち") or ""
    hand_match = re.search(r"([左右両])投([左右両])打", handedness)
    return {
        "profile_fields": fields,
        "position": fields.get("守備位置") or fields.get("ポジション"),
        "handedness": handedness or None,
        "throws": hand_match.group(1) if hand_match else None,
        "bats": hand_match.group(2) if hand_match else None,
        "height_cm": _num(fields.get("身長")),
        "weight_kg": _num(fields.get("体重")),
        "birth_date": fields.get("生年月日"),
        "career": fields.get("経歴"),
        "draft": fields.get("ドラフト"),
    }


def _profile_priority(player: dict[str, Any]) -> float:
    """Stable deterministic priority for bounded profile enrichment."""
    batting = player.get("batting") or {}
    pitching = player.get("pitching") or {}
    fielding = player.get("fielding") or []
    pa = _find_num(batting, "打席") or 0.0
    ip = _find_num(pitching, "投球回") or 0.0
    field_games = sum((_find_num(x, "試合") or 0.0) for x in fielding)
    return float(pa + 20.0 * ip + 2.0 * field_games)


def _enrich_profiles(players: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    try:
        limit = max(0, int(os.getenv("NPB_PLAYER_PROFILE_LIMIT_PER_TEAM", "24")))
    except Exception as exc:
        raise ValueError("NPB_PLAYER_PROFILE_LIMIT_PER_TEAM must be an integer >= 0") from exc
    selected = sorted(
        players,
        key=lambda p: (-_profile_priority(p), str(p.get("player_id") or ""), str(p.get("player_name") or "")),
    )[:limit]
    selected_ids = {str(p.get("player_id") or "").strip() for p in selected}
    resolved = 0
    for player in players:
        player["profile_status"] = "NOT_SELECTED"
        if str(player.get("player_id") or "").strip() not in selected_ids:
            continue
        url = _clean(player.get("player_url"))
        if not url:
            player["profile_status"] = "NO_PROFILE_URL"
            continue
        try:
            body, observed = _fetch(url)
            profile = parse_profile_page(body)
            profile.update(
                {
                    "schema_version": "npb-player-profile-v1",
                    "player_id": player.get("player_id"),
                    "player_name": player.get("player_name"),
                    "player_url": url,
                    "source": {
                        "source_id": "npb_official_player_page",
                        "url": url,
                        "status": "AVAILABLE",
                        "retrieved_at_utc": observed,
                        "available_at_utc": observed,
                        "published_at_utc": None,
                        "revision_time_utc": None,
                        "historical_oos_consumption": "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
                    },
                    "historical_oos_consumption": "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
                }
            )
            player["profile"] = profile
            player["profile_status"] = "AVAILABLE"
            resolved += 1
        except Exception as exc:
            player["profile"] = None
            player["profile_status"] = "SOURCE_FAILED"
            player["profile_error"] = f"{type(exc).__name__}: {exc}"
    return players, resolved


def collect_team(team: str, season: int = 2026) -> dict[str, Any]:
    canonical_team = normalize_team(team)
    suffix = TEAM_SUFFIX.get(canonical_team)
    if suffix is None:
        return {
            "team": canonical_team,
            "status": "UNKNOWN_TEAM",
            "season": season,
            "players": [],
            "sources": [],
            "historical_oos_consumption": "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
        }
    pages: dict[str, dict[str, Any]] = {}
    sources = []
    for kind in STAT_KIND:
        url = STATS_URL.format(season=season, kind=URL_KIND[kind], team=suffix)
        try:
            body, observed = _fetch(url)
            rows, as_of = parse_stats_page(body, kind)
            pages[kind] = {"rows": rows}
            sources.append({
                "source_id": f"npb_official_team_{kind}",
                "url": url,
                "status": "AVAILABLE",
                "retrieved_at_utc": observed,
                "available_at_utc": observed,
                "published_at_utc": None,
                "revision_time_utc": None,
                "source_as_of_date": as_of,
                "historical_oos_consumption": "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
            })
        except Exception as exc:
            pages[kind] = {"rows": []}
            sources.append({
                "source_id": f"npb_official_team_{kind}",
                "url": url,
                "status": "SOURCE_FAILED",
                "retrieved_at_utc": None,
                "available_at_utc": None,
                "published_at_utc": None,
                "revision_time_utc": None,
                "source_as_of_date": None,
                "error": f"{type(exc).__name__}: {exc}",
                "historical_oos_consumption": "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
            })
    available = sum(1 for x in sources if x["status"] == "AVAILABLE")
    status = "AVAILABLE" if available == len(STAT_KIND) else "PARTIAL" if available else "SOURCE_FAILED"
    snapshot = {
        "schema_version": "npb-team-player-context-v1",
        "team": canonical_team,
        "season": int(season),
        "status": status,
        "sources": sources,
        "players": _merge_players(canonical_team, pages),
        "historical_oos_consumption": "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
    }
    canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    snapshot["snapshot_id"] = hashlib.sha256(canonical).hexdigest()
    snapshot["player_count"] = len(snapshot["players"])
    snapshot["profile_count"] = int(sum(1 for p in snapshot["players"] if p.get("profile_status") == "AVAILABLE"))
    snapshot["profile_limit_per_team"] = int(os.getenv("NPB_PLAYER_PROFILE_LIMIT_PER_TEAM", "24") or 24)
    return snapshot


def collect_teams(teams: list[str], season: int = 2026) -> dict[str, Any]:
    unique = []
    for team in teams:
        normalized = normalize_team(team)
        if normalized not in unique:
            unique.append(normalized)
    team_contexts = [collect_team(team, season=season) for team in unique]
    snapshot = {
        "schema_version": "npb-team-player-context-v1",
        "season": int(season),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "teams_requested": unique,
        "teams": {ctx["team"]: ctx for ctx in team_contexts},
        "historical_oos_consumption": "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
    }
    canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    snapshot["snapshot_id"] = hashlib.sha256(canonical).hexdigest()
    return snapshot
