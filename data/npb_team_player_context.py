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
        out["iso_from_totals"] = max(0.0, total_bases / ab - h / ab)
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
    if tbf and tbf > 0:
        if so is not None:
            out["k_pct"] = so / tbf
        if bb is not None:
            out["bb_pct"] = bb / tbf
        if hr is not None:
            out["hr_pct"] = hr / tbf
        if so is not None and bb is not None:
            out["k_minus_bb_pct"] = (so - bb) / tbf
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
    return sorted(merged.values(), key=lambda x: (str(x.get("player_name") or ""), str(x.get("player_id") or "")))


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
