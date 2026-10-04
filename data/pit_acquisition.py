#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acquire real NPB/MLB source observations into the PIT ledger.

The collector records what was actually observable at the instant each source
response was received. It never backdates starter/lineup announcements or
claims historical availability that was not explicitly published by the source.
The immutable ledger is later replayed against an arbitrary prediction cutoff.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from html.parser import HTMLParser
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from core.http import get_json as http_get_json, get_text as http_get_text, session as http_session
from core.pit_snapshot import append_snapshot, make_snapshot, payload_hash
from core.mlb_pit_policy import derive_first_observed_evidence

ROOT = Path(__file__).resolve().parents[1]
PIT_DIR = ROOT / "data" / "pit"
SNAPSHOT_LOG = PIT_DIR / "source_snapshots.jsonl"
EVENT_LOG = PIT_DIR / "event_observations.jsonl"
AVAILABILITY_LOG = PIT_DIR / "availability_observations.jsonl"
RUN_LOG = PIT_DIR / "acquisition_runs.jsonl"

MLB_API = "https://statsapi.mlb.com/api/v1"
MLB_SCHEDULE_URL = f"{MLB_API}/schedule"
NPB_URLS = (
    "https://npb.jp/",
    "https://spaia.jp/baseball/npb/api/weekly_schedule",
)
TIMEOUT = max(10, int(os.getenv("PIT_ACQ_TIMEOUT", "45")))
HTTP_CONNECT_TIMEOUT = max(2, int(os.getenv("PIT_ACQ_CONNECT_TIMEOUT", "8")))
HTTP_RETRIES = max(1, int(os.getenv("PIT_ACQ_RETRIES", "4")))
LOOKAHEAD_DAYS = int(os.getenv("PIT_LOOKAHEAD_DAYS", "3"))
LOOKBACK_DAYS = int(os.getenv("PIT_LOOKBACK_DAYS", "1"))
# Expensive per-game MLB probes are optional; schedule acquisition is the PIT-critical path.
ENABLE_MLB_GAME_PROBES = os.getenv("PIT_ENABLE_MLB_GAME_PROBES", "0").strip().lower() in {"1", "true", "yes"}
PROBE_MIN_INTERVAL_MINUTES = max(1, int(os.getenv("PIT_MLB_PROBE_MIN_INTERVAL_MINUTES", "60")))
PROBE_LOOKAHEAD_HOURS = max(1, int(os.getenv("PIT_MLB_PROBE_LOOKAHEAD_HOURS", "48")))
PROBE_MAX_GAMES = max(1, int(os.getenv("PIT_MLB_PROBE_MAX_GAMES", "16")))
PROBE_TIMEOUT_SECONDS = max(3, int(os.getenv("PIT_MLB_PROBE_TIMEOUT_SECONDS", "8")))
PROBE_RETRIES = max(1, int(os.getenv("PIT_MLB_PROBE_RETRIES", "2")))
PROBE_ENTITY_TYPES = {"game_feed_timestamps", "game_content"}

SESSION = http_session(user_agent="Baseball-PIT-Acquisition/1.2")
SESSION.headers.update({
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
})


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def get_json(url: str, params: dict[str, Any] | None = None) -> tuple[Any, str]:
    payload = http_get_json(
        SESSION,
        url,
        params=params,
        timeout=(HTTP_CONNECT_TIMEOUT, TIMEOUT),
        retries=HTTP_RETRIES,
    )
    return payload, now_utc()


def get_json_probe(url: str) -> tuple[Any, str] | None:
    """Fetch optional MLB supporting evidence with a strict bounded time budget."""
    try:
        payload = http_get_json(
            SESSION,
            url,
            timeout=(min(HTTP_CONNECT_TIMEOUT, 5), PROBE_TIMEOUT_SECONDS),
            retries=PROBE_RETRIES,
        )
        return payload, now_utc()
    except Exception as exc:
        print(f"[PIT][MLB][PROBE] unavailable: {url}: {exc}")
        return None


def get_text(url: str) -> tuple[str, str]:
    text = http_get_text(
        SESSION,
        url,
        timeout=(HTTP_CONNECT_TIMEOUT, TIMEOUT),
        retries=HTTP_RETRIES,
    )
    return text, now_utc()


def _find_value(obj: Any, names: set[str]) -> Any:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in names and v not in (None, "", []):
                return v
        for v in obj.values():
            found = _find_value(v, names)
            if found not in (None, "", []):
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_value(v, names)
            if found not in (None, "", []):
                return found
    return None


def _candidate_games(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        keys = {str(k).lower() for k in obj}
        gameish = (
            {"gamepk", "gameid", "home", "away"} & keys
            or ("teams" in keys and ("gamepk" in keys or "status" in keys))
            or ("home_team" in keys and "away_team" in keys)
        )
        if gameish:
            yield obj
        for v in obj.values():
            yield from _candidate_games(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _candidate_games(v)


def _mlb_game_id(g: dict[str, Any]) -> str | None:
    for k in ("gamePk", "gamepk", "gameId", "game_id", "id"):
        if g.get(k) not in (None, ""):
            return str(g[k])
    return None


def _mlb_game_start(g: dict[str, Any]) -> datetime | None:
    """Parse the first-party schedule start time without guessing a timezone."""
    for key in ("gameDate", "game_date", "gameDateTime", "game_datetime", "startTime", "start_time"):
        value = g.get(key)
        if value in (None, ""):
            continue
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            continue
        if dt.tzinfo is None:
            continue
        return dt.astimezone(timezone.utc)
    return None


def _mlb_teams(g: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    teams = g.get("teams") if isinstance(g.get("teams"), dict) else {}
    return teams.get("home", {}) or {}, teams.get("away", {}) or {}


def _team_name(side: dict[str, Any]) -> str:
    t = side.get("team") if isinstance(side.get("team"), dict) else {}
    return str(t.get("name") or side.get("name") or "")


def _starter_name(side: dict[str, Any]) -> str | None:
    p = side.get("probablePitcher") or side.get("probable_pitcher")
    if isinstance(p, dict):
        return str(p.get("fullName") or p.get("full_name") or p.get("name") or "") or None
    return None

def _starter_id(side: dict[str, Any]) -> str | None:
    """Extract stable probable-pitcher identity when the source provides it."""
    p = side.get("probablePitcher") or side.get("probable_pitcher")
    if not isinstance(p, dict):
        return None
    value = p.get("id") or p.get("playerId") or p.get("player_id")
    return str(value) if value not in (None, "") else None


def _explicit_timestamp(
    g: dict[str, Any],
    side: str,
    kind: str,
) -> str | None:
    kind = str(kind).strip().lower()
    names = {
        "announcement": {
            f"{side}starterannouncedat", f"{side}_starter_announced_at",
            f"{side}probablepitcherannouncedat", f"{side}_probable_pitcher_announced_at",
            f"{side}pitcherannouncedat", f"{side}_pitcher_announced_at",
        },
        "published": {
            f"{side}starterpublishedat", f"{side}_starter_published_at",
            f"{side}probablepitcherpublishedat", f"{side}_probable_pitcher_published_at",
        },
        "available": {
            f"{side}starteravailableat", f"{side}_starter_available_at",
            f"{side}probablepitcheravailableat", f"{side}_probable_pitcher_available_at",
        },
    }.get(kind)
    if names is None:
        raise ValueError(f"unsupported starter timestamp kind: {kind}")
    value = _find_value(g, names)
    if value in (None, ""):
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def _explicit_announcement(g: dict[str, Any], side: str) -> str | None:
    return _explicit_timestamp(g, side, "announcement")



class _NpbScheduleTableParser(HTMLParser):
    """Parse NPB.jp rows, preserving multiple game elements within one row."""

    _TARGETS = {"team1", "team2", "place"}
    _VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[dict[str, Any]] = []
        self._row: dict[str, Any] | None = None
        self._cell: list[str] | None = None
        self._cells: list[str] = []
        self._target_capture: tuple[str, str, list[str]] | None = None

    @staticmethod
    def _attrs(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
        return {str(k).lower(): str(v or "") for k, v in attrs}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._finish_row()
            attr_map = self._attrs(attrs)
            self._row = {
                "row_id": attr_map.get("id", ""),
                "team1": [],
                "team2": [],
                "place": [],
                "raw": [],
                "cells": [],
            }
            self._cell = None
            self._cells = []
            self._target_capture = None
            return

        if self._row is None or tag in self._VOID_TAGS:
            return

        if tag in {"td", "th"}:
            self._cell = []

        class_names = set(self._attrs(attrs).get("class", "").split())
        target = next((name for name in self._TARGETS if name in class_names), None)
        if target is not None and self._target_capture is None:
            self._target_capture = (tag, target, [])

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._row is None:
            return

        if self._target_capture is not None:
            opening_tag, target, parts = self._target_capture
            if tag == opening_tag:
                value = _clean_npb_text(" ".join(parts))
                if value:
                    self._row[target].append(value)
                self._target_capture = None

        if tag in {"td", "th"} and self._cell is not None:
            self._cells.append(_clean_npb_text(" ".join(self._cell)))
            self._cell = None

        if tag == "tr":
            self._finish_row()
            self._row = None
            self._cell = None
            self._cells = []
            self._target_capture = None

    def handle_data(self, data: str) -> None:
        if self._row is None:
            return
        if data.strip():
            self._row["raw"].append(data)
        if self._cell is not None:
            self._cell.append(data)
        if self._target_capture is not None:
            self._target_capture[2].append(data)

    def _finish_row(self) -> None:
        if self._row is None:
            return
        if self._cell is not None:
            self._cells.append(_clean_npb_text(" ".join(self._cell)))
            self._cell = None
        self._row["cells"] = list(self._cells)
        if self._row.get("team1") or self._row.get("team2") or self._cells:
            self.rows.append(self._row)


def _clean_npb_text(value: Any) -> str:
    text = str(value or "").replace("\u3000", " ")
    return re.sub(r"\s+", " ", text).strip()


def _parse_npb_schedule_html(
    html_text: str,
    *,
    year: int,
    month: int,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> list[dict[str, Any]]:
    """Extract exact scheduled games from an NPB.jp month-detail page."""

    parser = _NpbScheduleTableParser()
    parser.feed(html_text)
    parser.close()

    start_day = start_date.date() if start_date else None
    end_day = end_date.date() if end_date else None
    seen_keys: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    last_explicit_date: tuple[int, int] | None = None

    for row_index, row in enumerate(parser.rows):
        raw = _clean_npb_text(" ".join(row.get("raw", [])))
        date_match = re.search(r"(?<!\d)(\d{1,2})/(\d{1,2})(?!\d)", raw)
        if date_match is not None:
            last_explicit_date = (int(date_match.group(1)), int(date_match.group(2)))
        if last_explicit_date is None:
            continue

        row_month, day = last_explicit_date
        if row_month != month or not 1 <= day <= 31:
            continue
        try:
            game_date = datetime(year, row_month, day).date()
        except ValueError:
            continue
        if start_day and game_date < start_day:
            continue
        if end_day and game_date > end_day:
            continue

        pairs: list[tuple[str, str, str]] = []
        team1s = [x for x in row.get("team1", []) if _clean_npb_text(x)]
        team2s = [x for x in row.get("team2", []) if _clean_npb_text(x)]
        places = [x for x in row.get("place", []) if _clean_npb_text(x)]
        if team1s and team2s:
            for idx, (home_team, away_team) in enumerate(zip(team1s, team2s)):
                place = places[idx] if idx < len(places) else (places[0] if len(places) == 1 else "")
                pairs.append((home_team, away_team, place))

        # Fallback for DOM variants without team1/team2 classes.
        if not pairs:
            cells = row.get("cells", [])
            if len(cells) >= 2:
                matchup = cells[1]
                venue_cell = cells[2] if len(cells) >= 3 else ""
                score_match = re.match(
                    r"^\s*(.+?)\s+(\d+)\s*-\s*(\d+)\s+(.+?)\s*$", matchup
                )
                if score_match:
                    home_team, _, _, away_team = score_match.groups()
                    pairs.append((_clean_npb_text(home_team), _clean_npb_text(away_team), venue_cell))
                else:
                    # A combined cell can contain several "A - B" pairs separated
                    # by line breaks; recover only clean two-team tokens.
                    tokens = [x.strip() for x in re.split(r"\s{2,}|\n|<br\s*/?>", matchup) if x.strip()]
                    for token in tokens:
                        basic_match = re.match(r"^\s*(.+?)\s*-\s*(.+?)\s*$", token)
                        if basic_match:
                            pairs.append((
                                _clean_npb_text(basic_match.group(1)),
                                _clean_npb_text(basic_match.group(2)),
                                venue_cell,
                            ))

        for home_team, away_team, place in pairs:
            home_team = _clean_npb_text(home_team)
            away_team = _clean_npb_text(away_team)
            combined = f"{home_team} {away_team}"
            if not home_team or not away_team or home_team == away_team:
                continue
            if any(token in combined for token in ("予備日", "中止", "延期", "中断")):
                continue
            if home_team in {"セ・リーグ", "パ・リーグ"}:
                continue

            place = _clean_npb_text(place)
            time_match = re.search(r"(?<!\d)(\d{1,2}:\d{2})(?!\d)", place)
            start_time = time_match.group(1) if time_match else None
            venue = _clean_npb_text(
                place[:time_match.start()] if time_match else place
            ).strip(" -/")
            base_key = "|".join([
                game_date.isoformat(),
                home_team,
                away_team,
                venue,
                start_time or "",
            ])
            seen_keys[base_key] = seen_keys.get(base_key, 0) + 1
            occurrence = seen_keys[base_key]
            identity_key = f"{base_key}|occurrence={occurrence}"
            digest = hashlib.sha256(identity_key.encode("utf-8")).hexdigest()[:24]
            out.append({
                "source_event_key": base_key,
                "event_id": "NPB:official:" + digest,
                "game_id": "NPB-OFFICIAL-" + digest,
                "league": "NPB",
                "home_team": home_team,
                "away_team": away_team,
                "event_date": game_date.isoformat(),
                "start_time_local": start_time,
                "venue": venue,
                "source_row_id": str(row.get("row_id") or row_index),
            })
    return out


def _npb_official_months(start_date: datetime, end_date: datetime) -> list[tuple[int, int]]:
    cursor = datetime(start_date.year, start_date.month, 1)
    finish = datetime(end_date.year, end_date.month, 1)
    months: list[tuple[int, int]] = []
    while cursor <= finish:
        months.append((cursor.year, cursor.month))
        if cursor.month == 12:
            cursor = datetime(cursor.year + 1, 1, 1)
        else:
            cursor = datetime(cursor.year, cursor.month + 1, 1)
    return months


def _load_last_probe_times() -> dict[tuple[str, str], datetime]:
    """Return the latest successful supporting-probe time per MLB game/entity."""
    latest: dict[tuple[str, str], datetime] = {}
    if not SNAPSHOT_LOG.exists():
        return latest
    for line in SNAPSHOT_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("league") != "MLB" or row.get("entity_type") not in PROBE_ENTITY_TYPES:
            continue
        if row.get("status", "KNOWN") != "KNOWN":
            continue
        value = row.get("retrieved_at")
        if not value:
            continue
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            continue
        if dt.tzinfo is None:
            continue
        key = (str(row.get("entity_type")), str(row.get("entity_id")))
        previous = latest.get(key)
        if previous is None or dt > previous:
            latest[key] = dt.astimezone(timezone.utc)
    return latest


def _probe_due(last_seen: datetime | None, now: datetime) -> bool:
    """Check cadence without treating future observations as due."""
    if last_seen is None:
        return True
    return (now - last_seen).total_seconds() >= PROBE_MIN_INTERVAL_MINUTES * 60


def _record_snapshot(*, event_id: str, league: str, entity_type: str,
                     entity_id: str, source: str, payload: Any,
                     retrieved_at: str, available_at: str | None,
                     status: str = "KNOWN") -> None:
    # The prediction cutoff is the observation instant for a current acquisition.
    # Historical PIT replay uses the stored available_at/retrieved_at values and
    # therefore remains independent of this acquisition run's wall-clock time.
    snap = make_snapshot(
        event_id=event_id, league=league, entity_type=entity_type,
        entity_id=entity_id, source=source, payload=payload,
        prediction_cutoff=retrieved_at, available_at=available_at,
        source_timestamp=None, retrieved_at=retrieved_at, status=status,
    )
    append_snapshot(snap, SNAPSHOT_LOG)


def acquire_mlb_game_timestamps(game_id: str) -> tuple[Any, str] | None:
    """Fetch MLB's public game-feed timestamp index as supporting PIT evidence.

    Timestamps are recorded for audit/replay diagnostics only. They do not, by
    themselves, prove when a probable starter was officially announced.
    """
    try:
        return get_json_probe(f"{MLB_API}/game/{game_id}/feed/live/timestamps")
    except Exception:
        return None
def acquire_mlb_game_content(game_id: str) -> tuple[Any, str] | None:
    """Fetch MLB game editorial/content metadata as supporting PIT evidence.

    Content publication timestamps can help reconstruct an information timeline,
    but an article/content timestamp is not treated as a starter announcement
    timestamp unless the payload explicitly identifies the starter announcement.
    """
    try:
        return get_json_probe(f"{MLB_API}/game/{game_id}/content")
    except Exception:
        return None


def _load_mlb_event_history() -> dict[str, list[dict[str, Any]]]:
    """Load immutable prior MLB event observations grouped by canonical game ID."""
    history: dict[str, list[dict[str, Any]]] = {}
    if not EVENT_LOG.exists():
        return history
    for line in EVENT_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("league") != "MLB":
            continue
        game_id = str(row.get("game_id") or "")
        if not game_id:
            continue
        history.setdefault(game_id, []).append(row)
    return history


def acquire_mlb() -> int:
    start = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).date()
    end = (datetime.now(timezone.utc) + timedelta(days=LOOKAHEAD_DAYS)).date()
    payload, retrieved = get_json(
        f"{MLB_API}/schedule",
        {"sportId": 1, "startDate": str(start), "endDate": str(end),
         "hydrate": "probablePitcher,linescore,venue"},
    )
    _record_snapshot(event_id="MLB-SCHEDULE", league="MLB", entity_type="schedule",
                     entity_id=f"{start}:{end}", source=MLB_SCHEDULE_URL,
                     payload=payload, retrieved_at=retrieved, available_at=retrieved)
    count = 0
    probe_now = datetime.fromisoformat(retrieved)
    probe_deadline = probe_now + timedelta(hours=PROBE_LOOKAHEAD_HOURS)
    last_probe_at = _load_last_probe_times() if ENABLE_MLB_GAME_PROBES else {}
    history_by_game = _load_mlb_event_history()
    probe_games = 0
    seen: set[str] = set()
    for g in _candidate_games(payload):
        gid = _mlb_game_id(g)
        if not gid or gid in seen:
            continue
        seen.add(gid)
        home, away = _mlb_teams(g)
        hname, aname = _team_name(home), _team_name(away)
        h_ann = _explicit_timestamp(g, "home", "announcement")
        a_ann = _explicit_timestamp(g, "away", "announcement")
        h_pub = _explicit_timestamp(g, "home", "published")
        a_pub = _explicit_timestamp(g, "away", "published")
        h_avail = _explicit_timestamp(g, "home", "available") or h_ann
        a_avail = _explicit_timestamp(g, "away", "available") or a_ann
        row = {
            "event_id": f"MLB:{gid}", "league": "MLB", "game_id": gid,
            "home_team": hname, "away_team": aname,
            "home_starter": _starter_name(home), "away_starter": _starter_name(away),
            "home_starter_id": _starter_id(home), "away_starter_id": _starter_id(away),
            "home_starter_announced_at": h_ann,
            "away_starter_announced_at": a_ann,
            "home_starter_published_at": h_pub,
            "away_starter_published_at": a_pub,
            "home_starter_available_at": h_avail,
            "away_starter_available_at": a_avail,
            "home_starter_evidence_level": "OFFICIAL_ANNOUNCEMENT" if h_ann else "RETRIEVAL_ONLY",
            "away_starter_evidence_level": "OFFICIAL_ANNOUNCEMENT" if a_ann else "RETRIEVAL_ONLY",
            "observed_at": retrieved, "prediction_cutoff": retrieved,
            # Use the canonical first-party MLB URL rather than a symbolic
            # provider label so the production PIT gate can independently
            # verify the source hostname. Side-specific fields avoid ambiguity
            # when starter evidence is consumed downstream.
            "source": MLB_SCHEDULE_URL,
            "home_starter_source": MLB_SCHEDULE_URL,
            "away_starter_source": MLB_SCHEDULE_URL,
            "payload_hash": payload_hash(g),
        }
        observations = history_by_game.get(gid, []) + [row]
        home_first = derive_first_observed_evidence(
            observations,
            game_id=gid,
            side="home",
            starter_id=row["home_starter_id"],
            starter_name=row["home_starter"],
        )
        away_first = derive_first_observed_evidence(
            observations,
            game_id=gid,
            side="away",
            starter_id=row["away_starter_id"],
            starter_name=row["away_starter"],
        )
        row["home_starter_first_observed_at"] = (
            home_first.timestamp.isoformat() if home_first.timestamp is not None else None
        )
        row["away_starter_first_observed_at"] = (
            away_first.timestamp.isoformat() if away_first.timestamp is not None else None
        )
        row["home_starter_availability_evidence"] = home_first.evidence_class.value
        row["away_starter_availability_evidence"] = away_first.evidence_class.value
        _append_jsonl(EVENT_LOG, row)
        _append_jsonl(AVAILABILITY_LOG, {
            **row,
            "starter_status": "ANNOUNCED" if row["home_starter_announced_at"] and row["away_starter_announced_at"] else "OBSERVED_UNVERIFIABLE_ANNOUNCEMENT_TIME",
            "starter_pit_proof": (
                "EXPLICIT_ANNOUNCEMENT_TIMESTAMP"
                if row["home_starter_announced_at"] and row["away_starter_announced_at"]
                else "OBSERVED_ONLY_UNVERIFIABLE_ANNOUNCEMENT_TIME"
            ),
            "lineup_status": "UNVERIFIABLE",
        })
        _record_snapshot(event_id=f"MLB:{gid}", league="MLB", entity_type="game",
                         entity_id=gid, source="MLB_STATS_API", payload=g,
                         retrieved_at=retrieved, available_at=retrieved)
        if ENABLE_MLB_GAME_PROBES:
            game_start = _mlb_game_start(g)
            probe_window_ok = game_start is not None and probe_now <= game_start <= probe_deadline
            probe_budget_ok = probe_games < PROBE_MAX_GAMES
            if not (probe_window_ok and probe_budget_ok):
                count += 1
                continue
            probe_games += 1
            timestamp_key = ("game_feed_timestamps", gid)
            if _probe_due(last_probe_at.get(timestamp_key), probe_now):
                timestamp_probe = acquire_mlb_game_timestamps(gid)
                if timestamp_probe is not None:
                    ts_payload, ts_retrieved = timestamp_probe
                    _record_snapshot(
                        event_id=f"MLB:{gid}", league="MLB", entity_type="game_feed_timestamps",
                        entity_id=gid, source=f"{MLB_API}/game/{gid}/feed/live/timestamps",
                        payload=ts_payload, retrieved_at=ts_retrieved, available_at=ts_retrieved,
                    )
                    last_probe_at[timestamp_key] = datetime.fromisoformat(ts_retrieved)

            content_key = ("game_content", gid)
            if _probe_due(last_probe_at.get(content_key), probe_now):
                content_probe = acquire_mlb_game_content(gid)
                if content_probe is not None:
                    content_payload, content_retrieved = content_probe
                    _record_snapshot(
                        event_id=f"MLB:{gid}", league="MLB", entity_type="game_content",
                        entity_id=gid, source=f"{MLB_API}/game/{gid}/content",
                        payload=content_payload, retrieved_at=content_retrieved, available_at=content_retrieved,
                    )
                    last_probe_at[content_key] = datetime.fromisoformat(content_retrieved)
        count += 1
    return count



def acquire_npb() -> int:
    start_dt = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    end_dt = datetime.now(timezone.utc) + timedelta(days=LOOKAHEAD_DAYS)
    start = start_dt.date()
    end = end_dt.date()
    count = 0

    # First-party NPB.jp month-detail pages are the primary schedule identity
    # source for the acquisition window.
    for year, month in _npb_official_months(start_dt, end_dt):
        url = f"https://npb.jp/games/{year}/schedule_{month:02d}_detail.html"
        try:
            html_text, retrieved = get_text(url)
            rows = _parse_npb_schedule_html(
                html_text, year=year, month=month, start_date=start_dt, end_date=end_dt
            )
            _record_snapshot(
                event_id="NPB-SCHEDULE",
                league="NPB",
                entity_type="official_schedule_month",
                entity_id=f"{year}:{month:02d}",
                source=url,
                payload={
                    "html_sha256": hashlib.sha256(html_text.encode("utf-8", "ignore")).hexdigest(),
                    "parsed_rows": len(rows),
                    "window_start": start.isoformat(),
                    "window_end": end.isoformat(),
                },
                retrieved_at=retrieved,
                available_at=retrieved,
            )
            for row in rows:
                # A current schedule observation proves visibility at retrieval,
                # not the time a starter was first announced. Keep strict starter
                # evidence unresolved unless another source supplies an explicit
                # announcement timestamp.
                base = {
                    "event_id": row["event_id"],
                    "league": "NPB",
                    "game_id": row["game_id"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "home_starter": None,
                    "away_starter": None,
                    "home_starter_announced_at": None,
                    "away_starter_announced_at": None,
                    "home_starter_evidence_level": "RETRIEVAL_ONLY",
                    "away_starter_evidence_level": "RETRIEVAL_ONLY",
                    "observed_at": retrieved,
                    "prediction_cutoff": retrieved,
                    "source": url,
                    "home_starter_source": url,
                    "away_starter_source": url,
                    "payload_hash": payload_hash(row),
                    "event_date": row["event_date"],
                    "start_time_local": row["start_time_local"],
                    "venue": row["venue"],
                    "source_event_key": row["source_event_key"],
                    "source_row_id": row["source_row_id"],
                    "starter_status": "OBSERVED_UNVERIFIABLE_ANNOUNCEMENT_TIME",
                    "lineup_status": "UNVERIFIABLE",
                }
                _append_jsonl(EVENT_LOG, base)
                _append_jsonl(AVAILABILITY_LOG, base)
                _record_snapshot(
                    event_id=row["event_id"],
                    league="NPB",
                    entity_type="official_schedule_game",
                    entity_id=row["game_id"],
                    source=url,
                    payload=row,
                    retrieved_at=retrieved,
                    available_at=retrieved,
                )
                count += 1
        except Exception as exc:
            retrieved = now_utc()
            _record_snapshot(
                event_id="NPB-SCHEDULE",
                league="NPB",
                entity_type="official_schedule_month",
                entity_id=f"{year}:{month:02d}",
                source=url,
                payload={"error": str(exc)},
                retrieved_at=retrieved,
                available_at=None,
                status="UNAVAILABLE",
            )
            print(f"[PIT][NPB][OFFICIAL] source unavailable: {url}: {exc}")

    # Keep the legacy SPAIA source as a separate research path.
    for url in NPB_URLS:
        if "npb.jp/" in url:
            continue
        try:
            payload, retrieved = get_json(url)
            _record_snapshot(
                event_id="NPB-SCHEDULE",
                league="NPB",
                entity_type="legacy_schedule",
                entity_id=f"{start}:{end}:{url}",
                source=url,
                payload=payload,
                retrieved_at=retrieved,
                available_at=retrieved,
            )
            seen: set[str] = set()
            for idx, g in enumerate(_candidate_games(payload)):
                gid = _find_value(g, {"gameid", "game_id", "gamepk", "id"})
                gid = str(gid) if gid is not None else f"{payload_hash(g)[:16]}-{idx}"
                if gid in seen:
                    continue
                seen.add(gid)
                home = _find_value(g, {"home", "home_team", "hometeam"})
                away = _find_value(g, {"away", "away_team", "awayteam"})
                if isinstance(home, dict):
                    home = home.get("name") or home.get("team")
                if isinstance(away, dict):
                    away = away.get("name") or away.get("team")
                if not home or not away:
                    continue
                h_ann = _explicit_announcement(g, "home")
                a_ann = _explicit_announcement(g, "away")
                row = {
                    "event_id": f"NPB:{gid}",
                    "league": "NPB",
                    "game_id": gid,
                    "home_team": str(home),
                    "away_team": str(away),
                    "home_starter": _find_value(g, {"homestarter", "home_starter", "homepitcher"}),
                    "away_starter": _find_value(g, {"awaystarter", "away_starter", "awaypitcher"}),
                    "home_starter_announced_at": h_ann,
                    "away_starter_announced_at": a_ann,
                    "observed_at": retrieved,
                    "prediction_cutoff": retrieved,
                    "source": url,
                    "payload_hash": payload_hash(g),
                }
                _append_jsonl(EVENT_LOG, row)
                _append_jsonl(
                    AVAILABILITY_LOG,
                    {
                        **row,
                        "starter_status": "ANNOUNCED" if h_ann and a_ann else "OBSERVED_UNVERIFIABLE_ANNOUNCEMENT_TIME",
                        "starter_evidence_contract": "STRICT_OFFICIAL_ANNOUNCEMENT_ONLY",
                        "lineup_status": "UNVERIFIABLE",
                    },
                )
                _record_snapshot(
                    event_id=f"NPB:{gid}",
                    league="NPB",
                    entity_type="legacy_game",
                    entity_id=gid,
                    source=url,
                    payload=g,
                    retrieved_at=retrieved,
                    available_at=retrieved,
                )
                count += 1
        except Exception as exc:
            retrieved = now_utc()
            _record_snapshot(
                event_id="NPB-SCHEDULE",
                league="NPB",
                entity_type="legacy_schedule",
                entity_id=f"{start}:{end}:{url}",
                source=url,
                payload={"error": str(exc)},
                retrieved_at=retrieved,
                available_at=None,
                status="UNAVAILABLE",
            )
            print(f"[PIT][NPB] source unavailable: {url}: {exc}")
    return count

def main() -> None:
    PIT_DIR.mkdir(parents=True, exist_ok=True)
    started = now_utc()
    results: dict[str, Any] = {"run_started_at": started}
    for league, fn in (("MLB", acquire_mlb), ("NPB", acquire_npb)):
        try:
            results[f"{league.lower()}_events"] = fn()
            results[f"{league.lower()}_status"] = "OK"
        except Exception as exc:
            results[f"{league.lower()}_events"] = 0
            results[f"{league.lower()}_status"] = "UNAVAILABLE"
            results[f"{league.lower()}_error"] = str(exc)
            print(f"[PIT][{league}] acquisition failed: {exc}")
    results["run_finished_at"] = now_utc()
    _append_jsonl(RUN_LOG, results)
    if results.get("mlb_status") != "OK" and results.get("npb_status") != "OK":
        raise SystemExit("Both NPB and MLB PIT sources failed")
    print(json.dumps(results, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
