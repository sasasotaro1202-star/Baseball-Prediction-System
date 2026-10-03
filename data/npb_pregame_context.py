"""PIT-aware pregame game-context acquisition for NPB.

The context layer deliberately separates observed-at-prediction-time data from
historical postgame evidence. Current/future snapshots can be persisted, while
historical OOS consumption remains disabled unless an independent availability
boundary is proven.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from core.http import request as http_request, session as http_session

ROOT = Path(__file__).resolve().parents[1]
HTTP_SESSION = http_session(user_agent="Baseball-Prediction-System/pregame-context")

NPB_DAY_URL = "https://npb.jp/bis/{year}/games/gm{date}.html"
NPB_MONTH_DETAIL_URL = "https://npb.jp/games/{year}/schedule_{month:02d}_detail.html"
NPB_STANDINGS_URL = "https://npb.jp/bis/eng/{year}/stats/std_{league}.html"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

TEAM_ALIASES = {
    "Yomiuri": "読売ジャイアンツ", "Yomiuri Giants": "読売ジャイアンツ", "巨人": "読売ジャイアンツ",
    "Hanshin": "阪神タイガース", "Hanshin Tigers": "阪神タイガース", "阪神": "阪神タイガース",
    "DeNA": "横浜DeNAベイスターズ", "YOKOHAMA DeNA BAYSTARS": "横浜DeNAベイスターズ", "横浜DeNA": "横浜DeNAベイスターズ",
    "横浜DeNAベイスターズ": "横浜DeNAベイスターズ",
    "Chunichi": "中日ドラゴンズ", "Chunichi Dragons": "中日ドラゴンズ", "中日": "中日ドラゴンズ",
    "Hiroshima": "広島東洋カープ", "Hiroshima Toyo Carp": "広島東洋カープ", "広島": "広島東洋カープ",
    "Yakult": "東京ヤクルトスワローズ", "Tokyo Yakult Swallows": "東京ヤクルトスワローズ", "ヤクルト": "東京ヤクルトスワローズ",
    "SoftBank": "福岡ソフトバンクホークス", "Fukuoka SoftBank Hawks": "福岡ソフトバンクホークス", "ソフトバンク": "福岡ソフトバンクホークス",
    "Seibu": "埼玉西武ライオンズ", "Saitama Seibu Lions": "埼玉西武ライオンズ", "西武": "埼玉西武ライオンズ",
    "Nippon-Ham": "北海道日本ハムファイターズ", "Hokkaido Nippon-Ham Fighters": "北海道日本ハムファイターズ", "日本ハム": "北海道日本ハムファイターズ",
    "ORIX": "オリックス・バファローズ", "ORIX Buffaloes": "オリックス・バファローズ", "オリックス": "オリックス・バファローズ",
    "Lotte": "千葉ロッテマリーンズ", "Chiba Lotte Marines": "千葉ロッテマリーンズ", "ロッテ": "千葉ロッテマリーンズ",
    "Rakuten": "東北楽天ゴールデンイーグルス", "Tohoku Rakuten Golden Eagles": "東北楽天ゴールデンイーグルス", "楽天": "東北楽天ゴールデンイーグルス",
}

STADIUMS = {
    "東京ドーム": {"stadium_id": "tokyo_dome", "lat": 35.7056, "lon": 139.7519, "roof_class": "DOME"},
    "明治神宮野球場": {"stadium_id": "meiji_jingu", "lat": 35.6745, "lon": 139.7175, "roof_class": "OPEN"},
    "横浜スタジアム": {"stadium_id": "yokohama_stadium", "lat": 35.4437, "lon": 139.6400, "roof_class": "OPEN"},
    "バンテリンドーム ナゴヤ": {"stadium_id": "vandteline_nagoya", "lat": 35.1859, "lon": 136.9476, "roof_class": "DOME"},
    "マツダ スタジアム": {"stadium_id": "mazda", "lat": 34.3915, "lon": 132.4840, "roof_class": "OPEN"},
    "阪神甲子園球場": {"stadium_id": "koshien", "lat": 34.7214, "lon": 135.3617, "roof_class": "OPEN"},
    "楽天モバイルパーク宮城": {"stadium_id": "rakuten_mobile", "lat": 38.2564, "lon": 140.9022, "roof_class": "OPEN"},
    "ベルーナドーム": {"stadium_id": "belluna", "lat": 35.7685, "lon": 139.4205, "roof_class": "DOME"},
    "ZOZOマリンスタジアム": {"stadium_id": "zozo_marine", "lat": 35.6456, "lon": 140.0302, "roof_class": "OPEN"},
    "京セラドーム大阪": {"stadium_id": "kyocera_osaka", "lat": 34.6694, "lon": 135.4762, "roof_class": "DOME"},
    "みずほPayPayドーム福岡": {"stadium_id": "paypay_fukuoka", "lat": 33.5953, "lon": 130.3620, "roof_class": "DOME"},
    "エスコンフィールドHOKKAIDO": {"stadium_id": "es_con_field", "lat": 42.9860, "lon": 141.5413, "roof_class": "DOME"},
}

VENUE_ALIASES = {
    "東京ドーム": "東京ドーム", "Tokyo Dome": "東京ドーム",
    "神宮": "明治神宮野球場", "明治神宮野球場": "明治神宮野球場", "Jingu": "明治神宮野球場",
    "横浜": "横浜スタジアム", "横浜スタジアム": "横浜スタジアム",
    "バンテリン": "バンテリンドーム ナゴヤ", "バンテリンドーム ナゴヤ": "バンテリンドーム ナゴヤ", "ナゴヤドーム": "バンテリンドーム ナゴヤ",
    "マツダ": "マツダ スタジアム", "MAZDA Zoom-Zoom スタジアム広島": "マツダ スタジアム", "マツダ スタジアム": "マツダ スタジアム",
    "甲子園": "阪神甲子園球場", "阪神甲子園球場": "阪神甲子園球場",
    "楽天モバイル": "楽天モバイルパーク宮城", "楽天モバイルパーク宮城": "楽天モバイルパーク宮城",
    "ベルーナドーム": "ベルーナドーム",
    "ZOZOマリン": "ZOZOマリンスタジアム", "ZOZOマリンスタジアム": "ZOZOマリンスタジアム",
    "京セラドーム": "京セラドーム大阪", "京セラドーム大阪": "京セラドーム大阪",
    "みずほPayPayドーム": "みずほPayPayドーム福岡", "PayPayドーム": "みずほPayPayドーム福岡", "みずほPayPayドーム福岡": "みずほPayPayドーム福岡",
    "エスコンフィールド": "エスコンフィールドHOKKAIDO", "エスコンフィールドHOKKAIDO": "エスコンフィールドHOKKAIDO",
}

TEAM_SET = set(TEAM_ALIASES) | set(TEAM_ALIASES.values())
CANONICAL_TEAMS = set(TEAM_ALIASES.values())
TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("　", " ")).strip()


def canonical_team(value: str) -> str:
    text = _clean(value)
    return TEAM_ALIASES.get(text, text)


def canonical_venue(value: str) -> str:
    text = _clean(value)
    for raw, canonical in VENUE_ALIASES.items():
        if raw in text:
            return canonical
    return text


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {key: value or "" for key, value in attrs}
        if tag in {"script", "style", "noscript", "template"}:
            self._hidden += 1
            return
        if self._hidden:
            return
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
        elif self._cell is not None and tag == "img":
            alt = attrs_dict.get("alt", "")
            title = attrs_dict.get("title", "")
            label = attrs_dict.get("aria-label", "")
            for value in (alt, title, label):
                value = _clean(value)
                if value and value not in self._cell:
                    self._cell.append(value)
        elif self._cell is not None and tag == "a":
            title = attrs_dict.get("title", "")
            label = attrs_dict.get("aria-label", "")
            for value in (title, label):
                value = _clean(value)
                if value and value not in self._cell:
                    self._cell.append(value)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template"}:
            self._hidden = max(0, self._hidden - 1)
            return
        if self._hidden:
            return
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(_clean(" ".join(self._cell)))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._hidden:
            return
        if self._cell is not None:
            value = _clean(data)
            if value:
                self._cell.append(value)


def _fetch_text(url: str, timeout: tuple[int, int] = (8, 45)) -> tuple[str, str]:
    response = http_request(HTTP_SESSION, url, timeout=timeout, retries=4)
    enc = (response.apparent_encoding or response.encoding or "utf-8").lower().replace("-", "_")
    if "shift_jis" in enc or "cp932" in enc or "shiftjis" in enc:
        body = response.content.decode("cp932", errors="strict")
    else:
        body = response.content.decode(response.apparent_encoding or response.encoding or "utf-8", errors="strict")
    now = datetime.now(timezone.utc).isoformat()
    return body, now


def _extract_team_sequence(value: str) -> list[str]:
    text = _clean(value)
    hits: list[tuple[int, int, str]] = []
    aliases: list[tuple[str, str]] = list(TEAM_ALIASES.items()) + [
        (canonical, canonical) for canonical in CANONICAL_TEAMS
    ]
    for alias, canonical in sorted(aliases, key=lambda item: len(item[0]), reverse=True):
        for match in re.finditer(re.escape(alias), text):
            start, end = match.span()
            if any(start >= a and end <= b for a, b, _ in hits):
                continue
            hits.append((start, end, canonical))
    selected: list[tuple[int, int, str]] = []
    for candidate in sorted(hits, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(not (candidate[1] <= a or candidate[0] >= b) for a, b, _ in selected):
            continue
        selected.append(candidate)
    return [canonical for _, _, canonical in sorted(selected, key=lambda item: item[0])]

class _LinearScheduleParser(HTMLParser):
    """Collect visible schedule tokens when the official page is not table-shaped."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tokens: list[str] = []
        self._hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {key: value or "" for key, value in attrs}
        if tag in {"script", "style", "noscript", "template"}:
            self._hidden += 1
            return
        if self._hidden:
            return
        if tag in {"img", "a"}:
            for value in (
                attrs_dict.get("alt"),
                attrs_dict.get("title"),
                attrs_dict.get("aria-label"),
            ):
                cleaned = _clean(value)
                if cleaned:
                    self.tokens.append(cleaned)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "template"}:
            self._hidden = max(0, self._hidden - 1)

    def handle_data(self, data: str) -> None:
        if self._hidden:
            return
        value = _clean(data)
        if value:
            self.tokens.append(value)


def _linear_schedule_tokens(page_html: str) -> list[str]:
    parser = _LinearScheduleParser()
    parser.feed(page_html)
    # Remove immediate duplicates caused by anchor title/label plus visible text.
    result: list[str] = []
    for token in parser.tokens:
        if not result or token != result[-1]:
            result.append(token)
    return result


def _linear_time_positions(tokens: list[str]) -> list[tuple[int, str]]:
    positions: list[tuple[int, str]] = []
    for index, token in enumerate(tokens):
        cleaned = _clean(token)
        if TIME_RE.fullmatch(cleaned):
            positions.append((index, cleaned))
            continue
        match = re.search(r"(?<!\d)(\d{1,2}:\d{2})(?!\d)", cleaned)
        if match:
            positions.append((index, match.group(1)))
    return positions


def _nearest_team_pair(
    tokens: list[str],
    time_index: int,
    used_ids: set[str],
    max_distance: int = 8,
) -> tuple[str, str] | None:
    before: list[tuple[int, str]] = []
    after: list[tuple[int, str]] = []
    for distance in range(1, max_distance + 1):
        for index in (time_index - distance, time_index + distance):
            if index < 0 or index >= len(tokens):
                continue
            for team in _extract_team_sequence(tokens[index]):
                if team in used_ids:
                    continue
                if index < time_index:
                    before.append((distance, team))
                elif index > time_index:
                    after.append((distance, team))
    if not before or not after:
        return None
    before.sort(key=lambda item: (item[0], item[1]))
    after.sort(key=lambda item: (item[0], item[1]))
    return before[0][1], after[0][1]


def _parse_linear_daily_schedule(page_html: str, target_date: str) -> list[dict[str, Any]]:
    """Fallback parser for current NPB daily pages whose game blocks are not tables."""
    tokens = _linear_schedule_tokens(page_html)
    time_positions = _linear_time_positions(tokens)
    if not time_positions:
        return []

    out: list[dict[str, Any]] = []
    used_team_ids: set[str] = set()
    for time_index, start_time in time_positions:
        pair = _nearest_team_pair(tokens, time_index, used_team_ids)
        if pair is None:
            continue
        home, away = pair
        used_team_ids.update((home, away))
        venue = "UNKNOWN"
        for distance in range(1, 5):
            for index in (time_index - distance, time_index + distance):
                if index < 0 or index >= len(tokens):
                    continue
                candidate = canonical_venue(tokens[index])
                if candidate in STADIUMS:
                    venue = candidate
                    break
            if venue != "UNKNOWN":
                break
        out.append({
            "game_id": f"NPB-{target_date}-{len(out)+1}",
            "game_date": target_date,
            "home": home,
            "away": away,
            "official_start_time": start_time,
            "venue": venue,
            "schedule_source": NPB_DAY_URL.format(year=target_date[:4], date=target_date.replace("-", "")),
        })
    return out


def _parse_linear_month_schedule(page_html: str, target_date: str) -> list[dict[str, Any]]:
    """Fallback parser for official monthly schedule pages."""
    tokens = _linear_schedule_tokens(page_html)
    if not tokens:
        return []
    month_day = f"{int(target_date[5:7])}/{int(target_date[8:10])}"
    day_only = str(int(target_date[8:10]))

    def is_target_date_token(token: str) -> bool:
        value = _clean(token)
        return bool(
            re.fullmatch(rf"{re.escape(month_day)}(?:[（(][^)）]*[)）])?", value)
            or re.fullmatch(rf"{re.escape(day_only)}(?:[（(][^)）]*[)）])?", value)
        )

    date_positions = [i for i, token in enumerate(tokens) if is_target_date_token(token)]
    if not date_positions:
        return []

    out: list[dict[str, Any]] = []
    used_team_ids: set[str] = set()
    for date_index in date_positions:
        end = next((x for x in date_positions if x > date_index), len(tokens))
        segment = tokens[date_index:end]
        # Re-index time positions to the segment.
        for local_time_index, start_time in _linear_time_positions(segment):
            pair = _nearest_team_pair(segment, local_time_index, used_team_ids)
            if pair is None:
                continue
            home, away = pair
            used_team_ids.update((home, away))
            venue = "UNKNOWN"
            for distance in range(1, 5):
                for index in (local_time_index - distance, local_time_index + distance):
                    if index < 0 or index >= len(segment):
                        continue
                    candidate = canonical_venue(segment[index])
                    if candidate in STADIUMS:
                        venue = candidate
                        break
                if venue != "UNKNOWN":
                    break
            out.append({
                "game_id": f"NPB-{target_date}-{len(out)+1}",
                "game_date": target_date,
                "home": home,
                "away": away,
                "official_start_time": start_time,
                "venue": venue,
                "schedule_source": NPB_MONTH_DETAIL_URL.format(
                    year=target_date[:4], month=int(target_date[5:7])
                ),
            })
    return out


def parse_official_schedule_detail(page_html: str, target_date: str) -> list[dict[str, Any]]:
    parser = _TableParser()
    parser.feed(page_html)
    month_day = f"{int(target_date[5:7])}/{int(target_date[8:10])}"
    day_only = str(int(target_date[8:10]))
    out: list[dict[str, Any]] = []
    for row in parser.rows:
        if not row:
            continue
        def is_target_date_cell(cell: str) -> bool:
            value = _clean(cell)
            return bool(
                re.fullmatch(rf"{re.escape(month_day)}(?:[（(][^)）]*[)）])?", value)
                or re.fullmatch(rf"{re.escape(day_only)}(?:[（(][^)）]*[)）])?", value)
            )
        if not any(is_target_date_cell(cell) for cell in row):
            continue
        teams: list[str] = []
        for cell in row:
            for team in _extract_team_sequence(cell):
                if team not in teams:
                    teams.append(team)
            if len(teams) >= 2:
                break
        if len(teams) != 2:
            continue
        times = [cell for cell in row if TIME_RE.fullmatch(_clean(cell))]
        if len(times) != 1:
            for cell in row:
                match = re.search(r"(?:^|\\s)(\\d{1,2}:\\d{2})(?:\\s|$)", _clean(cell))
                if match:
                    times = [match.group(1)]
                    break
        if len(times) != 1:
            continue
        venue = ""
        for cell in row:
            candidate = canonical_venue(cell)
            if candidate in STADIUMS:
                venue = candidate
                break
        out.append({
            "game_id": f"NPB-{target_date}-{len(out)+1}",
            "game_date": target_date,
            "home": teams[0],
            "away": teams[1],
            "official_start_time": times[0],
            "venue": venue or "UNKNOWN",
            "schedule_source": NPB_MONTH_DETAIL_URL.format(year=target_date[:4], month=int(target_date[5:7])),
        })
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for row in out:
        key = (row["home"], row["away"], row["official_start_time"])
        if key not in seen:
            seen.add(key)
            result.append(row)
    if not result:
        fallback = _parse_linear_month_schedule(page_html, target_date)
        if fallback:
            return fallback
        raise RuntimeError(f"official NPB monthly schedule parser found no game rows for {target_date}")
    return result
def parse_official_games(page_html: str, target_date: str) -> list[dict[str, Any]]:
    parser = _TableParser()
    parser.feed(page_html)
    out: list[dict[str, Any]] = []
    for row in parser.rows:
        teams: list[str] = []
        for cell in row:
            for team in _extract_team_sequence(cell):
                if team not in teams:
                    teams.append(team)
            if len(teams) >= 2:
                break
        if len(teams) != 2:
            continue
        times = [cell for cell in row if TIME_RE.fullmatch(_clean(cell))]
        if len(times) != 1:
            for cell in row:
                match = re.search(r"(?:^|\s)(\d{1,2}:\d{2})(?:\s|$)", _clean(cell))
                if match:
                    times = [match.group(1)]
                    break
        if len(times) != 1:
            continue
        venue = ""
        for cell in row:
            candidate = canonical_venue(cell)
            if candidate in STADIUMS:
                venue = candidate
                break
        if not venue:
            for cell in row:
                if not TIME_RE.fullmatch(cell) and canonical_team(cell) not in TEAM_SET and cell:
                    venue = canonical_venue(cell)
                    break
        out.append({
            "game_id": f"NPB-{target_date}-{len(out)+1}",
            "game_date": target_date,
            "home": teams[0],
            "away": teams[1],
            "official_start_time": times[0],
            "venue": venue or "UNKNOWN",
            "schedule_source": NPB_DAY_URL.format(year=target_date[:4], date=target_date.replace("-", "")),
        })
    seen: set[tuple[str, str, str]] = set()
    result = []
    for row in out:
        key = (row["home"], row["away"], row["official_start_time"])
        if key not in seen:
            seen.add(key)
            result.append(row)
    if not result:
        fallback = _parse_linear_daily_schedule(page_html, target_date)
        if fallback:
            return fallback
        raise RuntimeError(f"official NPB schedule context parser found no game rows for {target_date}")
    return result


def _parse_number(text: str) -> float | None:
    m = re.search(r"-?(?:\d+(?:\.\d+)?|\.\d+)", text.replace(",", ""))
    return float(m.group(0)) if m else None


def parse_standings(page_html: str, source_url: str, available_at_utc: str) -> dict[str, dict[str, Any]]:
    parser = _TableParser()
    parser.feed(page_html)
    result: dict[str, dict[str, Any]] = {}
    for row in parser.rows:
        if len(row) < 6:
            continue
        team = next((canonical_team(cell) for cell in row if canonical_team(cell) in CANONICAL_TEAMS), "")
        if not team:
            continue
        pct = next((_parse_number(x) for x in row[1:] if re.fullmatch(r"\.\d{3}", x)), None)
        ints = [_parse_number(x) for x in row[1:5]]
        if len(ints) < 4 or any(x is None for x in ints) or pct is None:
            continue
        games, wins, losses, ties = [int(x) for x in ints[:4]]
        gb = _parse_number(row[6]) if len(row) > 6 else None
        result[team] = {
            "games": games, "wins": wins, "losses": losses, "draws": ties,
            "win_pct": float(pct), "games_back": None if gb is None else float(gb),
            "home_record": row[7] if len(row) > 7 else None,
            "road_record": row[8] if len(row) > 8 else None,
            "source_url": source_url,
            "published_or_observed_at_utc": available_at_utc,
        }
    return result


@dataclass(frozen=True)
class WeatherPoint:
    status: str
    source: str
    available_at_utc: str | None
    requested_time_jst: str
    latitude: float | None
    longitude: float | None
    temperature_c: float | None
    apparent_temperature_c: float | None
    relative_humidity_pct: float | None
    dew_point_c: float | None
    precipitation_probability_pct: float | None
    precipitation_mm: float | None
    rain_mm: float | None
    wind_speed_kmh: float | None
    wind_gust_kmh: float | None
    wind_direction_deg: float | None
    pressure_msl_hpa: float | None
    cloud_cover_pct: float | None
    weather_code: float | None


def _hour_at_target(payload: dict[str, Any], target_dt_utc: pd.Timestamp) -> dict[str, Any] | None:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return None
    parsed = [pd.Timestamp(t).tz_localize(None) if pd.Timestamp(t).tzinfo is None else pd.Timestamp(t).tz_convert("Asia/Tokyo").tz_localize(None) for t in times]
    target = target_dt_utc.tz_convert("Asia/Tokyo").tz_localize(None)
    distances = [abs((t - target).total_seconds()) for t in parsed]
    idx = min(range(len(distances)), key=distances.__getitem__)
    if distances[idx] > 3600:
        return None
    def val(key: str) -> float | None:
        values = hourly.get(key) or []
        if idx >= len(values) or values[idx] is None:
            return None
        try:
            return float(values[idx])
        except (TypeError, ValueError):
            return None
    return {
        "temperature_c": val("temperature_2m"),
        "apparent_temperature_c": val("apparent_temperature"),
        "relative_humidity_pct": val("relative_humidity_2m"),
        "dew_point_c": val("dew_point_2m"),
        "precipitation_probability_pct": val("precipitation_probability"),
        "precipitation_mm": val("precipitation"),
        "rain_mm": val("rain"),
        "wind_speed_kmh": val("wind_speed_10m"),
        "wind_gust_kmh": val("wind_gusts_10m"),
        "wind_direction_deg": val("wind_direction_10m"),
        "pressure_msl_hpa": val("pressure_msl"),
        "cloud_cover_pct": val("cloud_cover"),
        "weather_code": val("weather_code"),
    }


def fetch_weather(venue: str, game_time_utc: pd.Timestamp) -> dict[str, Any]:
    canonical = canonical_venue(venue)
    meta = STADIUMS.get(canonical)
    requested = game_time_utc.tz_convert("Asia/Tokyo").isoformat()
    if not meta:
        return asdict(WeatherPoint("UNAVAILABLE_VENUE_UNKNOWN", "open_meteo_forecast", None, requested, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None))
    params = {
        "latitude": meta["lat"], "longitude": meta["lon"],
        "hourly": ",".join((
            "temperature_2m", "apparent_temperature", "relative_humidity_2m", "dew_point_2m",
            "precipitation_probability", "precipitation", "rain", "wind_speed_10m",
            "wind_gusts_10m", "wind_direction_10m", "pressure_msl", "cloud_cover", "weather_code",
        )),
        "start_date": game_time_utc.tz_convert("Asia/Tokyo").strftime("%Y-%m-%d"),
        "end_date": game_time_utc.tz_convert("Asia/Tokyo").strftime("%Y-%m-%d"),
        "timezone": "Asia/Tokyo",
    }
    url = OPEN_METEO_URL + "?" + urlencode(params)
    try:
        response = http_request(HTTP_SESSION, url, timeout=(8, 45), retries=3)
        point = _hour_at_target(response.json(), game_time_utc)
        observed = datetime.now(timezone.utc).isoformat()
        if point is None:
            return asdict(WeatherPoint("UNAVAILABLE_TARGET_HOUR", "open_meteo_forecast", observed, requested, float(meta["lat"]), float(meta["lon"]), None, None, None, None, None, None, None, None, None, None, None, None, None))
        return {
            "status": "AVAILABLE", "source": "open_meteo_forecast", "available_at_utc": observed,
            "requested_time_jst": requested, "latitude": float(meta["lat"]), "longitude": float(meta["lon"]), **point,
        }
    except Exception as exc:
        return {
            "status": "SOURCE_FAILED", "source": "open_meteo_forecast", "available_at_utc": None,
            "requested_time_jst": requested, "latitude": float(meta["lat"]), "longitude": float(meta["lon"]),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _load_standings(target_date: str) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    year = target_date[:4]
    all_rows: dict[str, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    for league_code, league_name in (("c", "CENTRAL"), ("p", "PACIFIC")):
        url = NPB_STANDINGS_URL.format(year=year, league=league_code)
        try:
            html, observed = _fetch_text(url)
            rows = parse_standings(html, url, observed)
            for team, values in rows.items():
                all_rows[team] = {**values, "league": league_name}
            provenance.append({"source_id": f"npb_official_standings_{league_name.lower()}", "status": "AVAILABLE", "url": url, "available_at_utc": observed, "rows": len(rows)})
        except Exception as exc:
            provenance.append({"source_id": f"npb_official_standings_{league_name.lower()}", "status": "SOURCE_FAILED", "url": url, "available_at_utc": None, "error": f"{type(exc).__name__}: {exc}"})
    return all_rows, provenance


def _snapshot_id(obj: dict[str, Any]) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def collect_npb_pregame_context(target_date: str, *, now_utc: pd.Timestamp | None = None) -> dict[str, Any]:
    now = now_utc or pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    url = NPB_DAY_URL.format(year=target_date[:4], date=target_date.replace("-", ""))
    schedule_primary_error = None
    schedule_variant = "DAILY_GAME_PAGE"
    try:
        html, schedule_observed = _fetch_text(url)
        games = parse_official_games(html, target_date)
    except Exception as exc:
        schedule_primary_error = f"{type(exc).__name__}: {exc}"
        fallback_url = NPB_MONTH_DETAIL_URL.format(year=target_date[:4], month=int(target_date[5:7]))
        fallback_html, fallback_observed = _fetch_text(fallback_url)
        games = parse_official_schedule_detail(fallback_html, target_date)
        url = fallback_url
        schedule_observed = fallback_observed
        schedule_variant = "MONTH_DETAIL_FALLBACK"
    for game in games:
        game["schedule_available_at_utc"] = schedule_observed
        start = pd.Timestamp(f"{target_date} {game['official_start_time']}").tz_localize("Asia/Tokyo").tz_convert("UTC")
        game["scheduled_start_utc"] = start.isoformat()
        game["minutes_to_start"] = round((start - now).total_seconds() / 60.0, 3)
        venue = canonical_venue(game["venue"])
        meta = STADIUMS.get(venue)
        game["venue"] = venue
        game["stadium_id"] = meta["stadium_id"] if meta else None
        game["roof_class"] = meta["roof_class"] if meta else "UNKNOWN"

    standings, standing_provenance = _load_standings(target_date)

    def enrich(game: dict[str, Any]) -> dict[str, Any]:
        start = pd.Timestamp(game["scheduled_start_utc"])
        return {
            **game,
            "standing_home": standings.get(game["home"]),
            "standing_away": standings.get(game["away"]),
            "weather": fetch_weather(game["venue"], start),
        }

    enriched: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(games)))) as executor:
        future_map = {executor.submit(enrich, game): game for game in games}
        for future in as_completed(future_map):
            enriched.append(future.result())
    enriched.sort(key=lambda row: (row["scheduled_start_utc"], row["home"], row["away"]))

    available_weather = sum(1 for game in enriched if game.get("weather", {}).get("status") == "AVAILABLE")
    source_status = [
        {
            "source_id": "npb_official_game_schedule_context",
            "status": "AVAILABLE",
            "available_at_utc": schedule_observed,
            "url": url,
            "rows": len(games),
            "endpoint_variant": schedule_variant,
            "primary_daily_endpoint": NPB_DAY_URL.format(year=target_date[:4], date=target_date.replace("-", "")),
            "primary_daily_endpoint_error": schedule_primary_error,
        },
        *standing_provenance,
        {
            "source_id": "open_meteo_forecast", "status": "AVAILABLE" if available_weather else "SOURCE_FAILED_OR_UNAVAILABLE",
            "rows": available_weather, "total_games": len(enriched),
            "available_at_utc": max((game.get("weather", {}).get("available_at_utc") for game in enriched if game.get("weather", {}).get("available_at_utc")), default=None),
        },
    ]

    snapshot = {
        "schema_version": "npb-pregame-context-v1",
        "target_date": target_date,
        "generated_at_utc": now.isoformat(),
        "prediction_cutoff_utc": now.isoformat(),
        "historical_oos_consumption": "DISABLED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
        "game_count": len(enriched),
        "games": enriched,
        "sources": source_status,
    }
    snapshot["snapshot_id"] = _snapshot_id(snapshot)
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYY-MM-DD JST")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    snapshot = collect_npb_pregame_context(args.date)
    out = Path(args.out) if args.out else ROOT / "results" / f"npb_pregame_context_{args.date}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "EXECUTED", "snapshot_id": snapshot["snapshot_id"], "games": snapshot["game_count"], "output": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
