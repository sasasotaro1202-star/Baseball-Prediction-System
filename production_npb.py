#!/usr/bin/env python3
"""Production NPB prediction entrypoint.

Contract:
official NPB schedule/announced starters -> PIT gate -> chronological historical
features -> full-history ensemble -> coherent score distribution -> validated JSON.

The target game itself is never appended to historical training data.
"""
from __future__ import annotations
import time
import argparse, json, re, html as html_lib
from html.parser import HTMLParser
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from core.atomic_io import atomic_write_json

from baseball_backtest import BaseballBacktest, norm_team, score_candidates, low_high_probs
from research.correlated_score import npb_final_outcomes
from core.pit_evidence import _is_official_source

ROOT = Path(__file__).resolve().parent
TIMEOUT = 30
NPB_STARTER_URL = "https://npb.jp/announcement/starter/"
NPB_DAY_URL = "https://npb.jp/bis/eng/2026/games/gm{date}.html"

TEAM_MAP = {
    "Yomiuri":"読売ジャイアンツ","Yakult":"東京ヤクルトスワローズ",
    "Chunichi":"中日ドラゴンズ","Hiroshima":"広島東洋カープ",
    "Hanshin":"阪神タイガース","DeNA":"横浜DeNAベイスターズ",
    "Nippon-Ham":"北海道日本ハムファイターズ","ORIX":"オリックス・バファローズ",
    "Rakuten":"東北楽天ゴールデンイーグルス","SoftBank":"福岡ソフトバンクホークス",
    "Lotte":"千葉ロッテマリーンズ","Seibu":"埼玉西武ライオンズ",
}

def fetch_text(url: str) -> str:
    """Fetch an official NPB page with bounded transient-error recovery."""
    transient_statuses = {429, 502, 503, 504}
    attempts = 4
    last_exc = None
    for attempt in range(attempts):
        try:
            r = requests.get(
                url,
                timeout=TIMEOUT,
                headers={"User-Agent":"Baseball-Prediction-System/production"},
            )
            if r.status_code in transient_statuses and attempt < attempts - 1:
                retry_after = r.headers.get("Retry-After")
                try:
                    delay = min(8.0, max(1.0, float(retry_after)))
                except (TypeError, ValueError):
                    delay = float(2 ** attempt)
                time.sleep(delay)
                continue
            r.raise_for_status()
            enc = (r.apparent_encoding or r.encoding or "utf-8").lower().replace("-", "_")
            if "shift_jis" in enc or "cp932" in enc or "shiftjis" in enc:
                return r.content.decode("cp932", errors="strict")
            return r.content.decode(r.apparent_encoding or r.encoding or "utf-8", errors="strict")
        except requests.RequestException as exc:
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status not in transient_statuses or attempt >= attempts - 1:
                raise
            time.sleep(float(2 ** attempt))
    raise RuntimeError(f"official NPB page fetch failed after {attempts} attempts: {url}") from last_exc

def _clean_name(value: str) -> str:
    return re.sub(r"\s+", " ", html_lib.unescape(value)).replace("　", " ").strip()

class _VisibleTextParser(__import__("html.parser", fromlist=["HTMLParser"]).HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
    def handle_starttag(self, tag, attrs):
        if tag.lower() == "img":
            alt = dict(attrs).get("alt", "")
            value = _clean_name(alt)
            if value:
                self.parts.append(value)
    def handle_data(self, data):
        value = _clean_name(data)
        if value:
            self.parts.append(value)


class _DailyScheduleTextParser(HTMLParser):
    """Collect visible daily-schedule text while excluding script/style payloads."""
    def __init__(self):
        super().__init__()
        self.parts = []
        self._hidden_depth = 0
    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template"}:
            self._hidden_depth += 1
            return
        if self._hidden_depth or tag != "img":
            return
        value = _clean_name(dict(attrs).get("alt", ""))
        if value:
            self.parts.append(value)
    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style", "noscript", "template"}:
            self._hidden_depth = max(0, self._hidden_depth - 1)
    def handle_data(self, data):
        if self._hidden_depth:
            return
        value = _clean_name(data)
        if value:
            self.parts.append(value)

class _StarterGameCardParser(HTMLParser):
    """Extract one official start time only from a structural NPB game card."""
    def __init__(self, aliases: dict[str, str]):
        super().__init__()
        self.aliases = aliases
        self.depth = 0
        self.unit_depth = None
        self.teams = []
        self.times = []
        self.cards = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attrs_dict = {k: v or "" for k, v in attrs}
        if tag == "div" and "unit" in set((attrs_dict.get("class") or "").split()) and self.unit_depth is None:
            self.unit_depth = self.depth
            self.teams = []
            self.times = []
        if self.unit_depth is None:
            self.depth += 1
            return
        if tag == "img":
            alt = _clean_name(attrs_dict.get("alt", ""))
            if alt in self.aliases:
                canon = self.aliases[alt]
                if canon not in self.teams:
                    self.teams.append(canon)
        self.depth += 1

    def handle_data(self, data):
        if self.unit_depth is None:
            return
        value = _clean_name(data)
        if value and re.fullmatch(r"\d{1,2}:\d{2}", value):
            self.times.append(value)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if self.unit_depth is None:
            return
        self.depth = max(0, self.depth - 1)
        if tag == "div" and self.depth == self.unit_depth:
            if len(self.teams) == 2 and len(self.times) == 1:
                self.cards.append((list(self.teams), list(self.times)))
            self.unit_depth = None
            self.teams = []
            self.times = []

    def parsed_times(self):
        return [(teams[0], teams[1], times[0]) for teams, times in self.cards]

class _UnitStarterParser(HTMLParser):
    """Extract team/starter pairs only from official game-card .unit containers.

    This intentionally ignores duplicated responsive/accessibility text outside
    the structural game card, while still failing closed when a real .unit
    contains the same team more than once or a team appears in multiple cards.
    """
    def __init__(self, teams: list[str]):
        super().__init__()
        self.teams = set(teams)
        self.depth = 0
        self.unit_depth = None
        self.team_left_depth = None
        self.fallback_depth = None
        self.current_team = None
        self.current_name = []
        self.results = []
        self.position = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attrs_dict = {k: v or "" for k, v in attrs}
        classes = set((attrs_dict.get("class") or "").split())
        if tag == "div" and "unit" in classes and self.unit_depth is None:
            self.unit_depth = self.depth
        if self.unit_depth is None:
            self.depth += 1
            return
        if tag == "img" and self.current_team is None:
            alt = _clean_name(attrs_dict.get("alt", ""))
            if alt in self.teams:
                self.current_team = alt
                self.current_name = []
        if tag == "div" and "team_left" in classes and self.current_team is not None and self.team_left_depth is None:
            self.team_left_depth = self.depth
            self.current_name = []
        elif tag in {"p", "a"} and self.current_team is not None and self.team_left_depth is None and self.fallback_depth is None:
            self.fallback_depth = self.depth
            self.current_name = []
        self.depth += 1

    def handle_data(self, data):
        if self.unit_depth is not None and (self.team_left_depth is not None or self.fallback_depth is not None):
            value = _clean_name(data)
            if value:
                self.current_name.append(value)

    def handle_endtag(self, tag):
        tag = tag.lower()
        self.depth = max(0, self.depth - 1)
        if self.unit_depth is None:
            return
        if self.team_left_depth is not None and self.depth == self.team_left_depth:
            name = _clean_name(" ".join(self.current_name))
            if self.current_team and name:
                self.results.append((self.position, self.current_team, name))
                self.position += 1
            self.current_team = None
            self.current_name = []
            self.team_left_depth = None
        if self.fallback_depth is not None and self.depth == self.fallback_depth:
            name = _clean_name(" ".join(self.current_name))
            if self.current_team and name:
                self.results.append((self.position, self.current_team, name))
                self.position += 1
            self.current_team = None
            self.current_name = []
            self.fallback_depth = None
        if tag == "div" and self.depth == self.unit_depth:
            self.unit_depth = None
            self.team_left_depth = None
            self.fallback_depth = None
            self.current_team = None
            self.current_name = []


def _parse_starters_by_units(section: str, teams: list[str]) -> list[tuple[int,str,str]]:
    parser = _UnitStarterParser(teams)
    parser.feed(section)
    found = parser.results
    if not found:
        return []
    seen = {}
    for pos, team, name in found:
        seen.setdefault(team, []).append((pos, team, name))
    if any(len(items) > 1 for items in seen.values()):
        raise RuntimeError("PIT starter gate failed: duplicate team tokens in official starter order.")
    return found


def _parse_starters_by_visible_text(section: str, teams: list[str]) -> list[tuple[int,str,str]]:
    parser = _VisibleTextParser()
    parser.feed(section)
    text = parser.parts
    found = []
    for i, token in enumerate(text):
        token_clean = _clean_name(token)
        for team in teams:
            if token_clean == team or team in token_clean:
                j = i + 1
                while j < len(text):
                    cand = _clean_name(text[j])
                    if cand and cand not in teams and not re.fullmatch(r"\d{1,2}:\d{2}", cand) and "予告先発投手" not in cand:
                        found.append((i, team, cand))
                        break
                    j += 1
                break
    teams_seen = {}
    for pos, team, name in found:
        teams_seen.setdefault(team, []).append((pos, team, name))
    # Preserve duplicate evidence instead of silently collapsing it. The caller
    # must fail closed when the official visible sequence contains a team token
    # more than once in the target slate.
    duplicates = {team: items for team, items in teams_seen.items() if len(items) > 1}
    if duplicates:
        raise RuntimeError("PIT starter gate failed: duplicate team tokens in official starter order.")
    return sorted(item[0] for item in teams_seen.values())

def parse_official_starters_html(page_html: str, target_date: str) -> list[dict]:
    """Parse NPB's announced-starter section with semantic fail-closed validation."""
    month_day = f"{int(target_date[5:7])}月{int(target_date[8:10])}日"
    heading = re.search(
        rf"<h4[^>]*>\s*{re.escape(month_day)}の予告先発投手\s*</h4>",
        page_html, re.I,
    )
    if not heading:
        raise RuntimeError(
            f"Official NPB starter page does not contain {month_day}; refusing prediction."
        )
    tail = page_html[heading.end():]
    next_heading = re.search(r"<h4\b", tail, re.I)
    section = tail[:next_heading.start()] if next_heading else tail

    teams = [
        "読売ジャイアンツ","東京ヤクルトスワローズ","中日ドラゴンズ","広島東洋カープ",
        "阪神タイガース","横浜DeNAベイスターズ","北海道日本ハムファイターズ",
        "オリックス・バファローズ","東北楽天ゴールデンイーグルス","福岡ソフトバンクホークス",
        "千葉ロッテマリーンズ","埼玉西武ライオンズ",
    ]
    occurrences = []
    for team in teams:
        team_pat = re.escape(team)
        # On the live NPB page each game is a .unit whose team logo is
        # immediately associated with a player <span>. Bound the search so
        # footer/navigation links can never become a pitcher.
        m = re.search(
            rf'<div[^>]+class=["\'][^"\']*unit[^"\']*["\'][^>]*>.*?'
            rf'<img[^>]+alt=["\']{team_pat}["\'][^>]*>.*?'
            rf'<(?:div|p)[^>]+class=["\'][^"\']*team_left[^"\']*["\'][^>]*>.*?'
            rf'<span[^>]*>\s*([^<]+?)\s*</span>',
            section, re.I | re.S,
        )
        if not m:
            # Deterministic fixture/backward-compatible fallback: still bound
            # extraction to a short window after the team's logo.
            tm = re.search(rf'alt=["\']{team_pat}["\']', section, re.I)
            if tm:
                window = section[tm.end():tm.end()+1200]
                sm = re.search(r'<span[^>]*>\s*([^<]+?)\s*</span>', window, re.I | re.S)
                if not sm:
                    sm = re.search(r'<a[^>]*>\s*([^<]+?)\s*</a>', window, re.I | re.S)
                if sm:
                    m = sm
                    name = _clean_name(sm.group(1))
                    occurrences.append((tm.start(), team, name))
                    continue
            continue
        name = _clean_name(m.group(1))
        occurrences.append((m.start(), team, name))

    occurrences.sort()
    structural_occurrences = _parse_starters_by_units(section, teams)

    aliases_for_cards = {team: team for team in teams}
    card_parser = _StarterGameCardParser(aliases_for_cards)
    card_parser.feed(section)
    card_times = card_parser.parsed_times()

    time_matches = list(re.finditer(r"(?<!\d)(\d{1,2}:\d{2})(?!\d)", section))
    if not time_matches:
        raise RuntimeError("PIT starter gate failed: no official game times found.")
    # Prefer the number of structurally resolved official game cards. This
    # prevents unrelated clock-like text later in the page (e.g. average game
    # duration) from becoming a phantom game start time.
    team_bounded_games = (
        len(occurrences) // 2
        if occurrences and len(occurrences) % 2 == 0
        else 0
    )
    structural_games = (
        len(structural_occurrences) // 2
        if structural_occurrences and len(structural_occurrences) % 2 == 0
        else 0
    )
    # The starter team/starter pairs are the authoritative cardinality. Clock
    # tokens are only usable as an upper bound because the page may contain
    # unrelated statistics such as average game duration ("3:05").
    available_game_counts = [
        game_count for game_count in (team_bounded_games, structural_games)
        if game_count > 0
    ]
    if not available_game_counts or not time_matches:
        raise RuntimeError("PIT starter gate failed: could not deterministically resolve official game cardinality.")
    # Use the strongest deterministic team/starter structure available, bounded
    # by the number of visible clock tokens. A partial per-team fallback must
    # never reduce a complete structural extraction from six games to three.
    expected_games = min(max(available_game_counts), len(time_matches))
    if len(time_matches) < expected_games:
        raise RuntimeError(
            f"PIT starter gate failed: expected {expected_games} official game times, "
            f"found {len(time_matches)}."
        )
    # Some official NPB revisions use one .unit per team rather than one .unit
    # per game. Do not let a structurally valid-but-cardinality-incomplete
    # extraction erase the broader team-bounded extraction above. Structural
    # evidence is preferred only when it resolves every expected team.
    if structural_occurrences and len(structural_occurrences) == 2 * expected_games:
        occurrences = structural_occurrences

    # The official page can use responsive/CSS ordering in which the DOM
    # positions of the game cards are not the visual game order. A strict
    # per-team regex can therefore silently pair a valid starter with the
    # wrong team. Prefer the semantic visible-text stream whenever it yields
    # the expected team/starter cardinality; this follows the accessible
    # sequence exposed by the official page and is safer than character-
    # distance inference.
    try:
        visible = _parse_starters_by_visible_text(section, teams)
    except RuntimeError as exc:
        # Responsive NPB markup can expose a team logo alt-text and the same
        # team name again in accessible visible text. That duplication is a
        # presentation artifact, not proof of two games. Fall back to the
        # already bounded per-unit extraction, which still requires unique
        # team/starter pairing and exact time cardinality.
        if "duplicate team tokens" in str(exc):
            visible = []
        else:
            raise
    if len(visible) >= 2 * expected_games:
        visible = visible[: 2 * expected_games]
        visible_teams = [x[1] for x in visible]
        if len(set(visible_teams)) != len(visible_teams):
            raise RuntimeError("PIT starter gate failed: duplicate team tokens in official starter order.")
        occurrences = visible

    pair_candidates = []
    if len(occurrences) >= 2:
        for i in range(0, len(occurrences) - 1, 2):
            pair_candidates.append((i // 2, occurrences[i], occurrences[i + 1]))
    if len(pair_candidates) < expected_games:
        raise RuntimeError(
            f"PIT starter gate failed: expected at least {expected_games} team/starter pairs "
            f"for {expected_games} officially timed games, got {len(pair_candidates)}."
        )

    # In the common official markup each game's time appears in the same
    # sequential unit as its two starters. When the parser has exactly one
    # pair per official time, preserve that authoritative page order rather
    # than using character-distance matching (which is unsafe when all times
    # are rendered in a footer-like block after the starter cards).
    if len(pair_candidates) == expected_games:
        timed_pairs = {
            i: (pair_candidates[i][1], pair_candidates[i][2], card_times[i][2] if len(card_times) == expected_games else time_matches[i].group(1))
            for i in range(expected_games)
        }
    else:
        # Do not use raw character-distance heuristics to pair game times with
        # starters. An unexpected cardinality means the official page structure
        # is not deterministically understood, so fail closed.
        raise RuntimeError(
            f"PIT starter gate failed: resolved {len(pair_candidates)} starter pairs "
            f"for {expected_games} visible official game times."
        )

    selected_pairs = [timed_pairs[i] for i in sorted(timed_pairs)]
    occurrences = [item for pair in selected_pairs for item in pair[:2]]
    times = [pair[2] for pair in selected_pairs]

    invalid = {"一般社団法人日本野球機構について", "採用情報", "プライバシーポリシー"}
    for _, team, name in occurrences:
        if not name or name in invalid or name in teams or "日本野球機構" in name:
            raise RuntimeError(f"PIT starter gate failed: invalid starter extracted for {team}: {name!r}")

    out = []
    for i in range(0, len(occurrences), 2):
        home_team, home_starter = occurrences[i][1], occurrences[i][2]
        away_team, away_starter = occurrences[i+1][1], occurrences[i+1][2]
        if home_starter == away_starter:
            raise RuntimeError(
                "PIT starter gate failed: identical starter assigned to both teams "
                f"in one official game ({home_team} vs {away_team}): {home_starter!r}."
            )
        out.append({
            "home": home_team, "away": away_team,
            "home_starter": home_starter, "away_starter": away_starter,
            "confirmed_starters": True, "starter_evidence_status": "official_announced",
            "starter_source": NPB_STARTER_URL, "official_start_time": times[i//2],
        })
    return out

def parse_official_league_starters_html(
    page_html: str,
    target_date: str,
    source_url: str,
    *,
    min_games: int = 3,
) -> list[dict]:
    """Parse one official NPB league page when the dedicated starter page has rolled forward.

    The league pages are first-party NPB pages and expose the target-date
    announced starters in the same game-card order. This fallback is used only
    when the dedicated starter page lacks the target-date heading; it never
    fills missing starters with third-party or probable-pitcher data.
    """
    month_day = f"{int(target_date[5:7])}月{int(target_date[8:10])}日"
    heading = re.search(
        rf"<h[3-6][^>]*>[^<]*{re.escape(month_day)}[^<]*予告先発[^<]*</h[3-6]>",
        page_html, re.I,
    )
    if not heading:
        raise RuntimeError(
            f"Official NPB league page does not contain {month_day} announced starters."
        )

    tail = page_html[heading.end():]
    next_heading = re.search(r"<h[3-6]\b", tail, re.I)
    section = tail[:next_heading.start()] if next_heading else tail
    parser = _VisibleTextParser()
    parser.feed(section)
    tokens = [_clean_name(x) for x in parser.parts if _clean_name(x)]

    league_teams = {
        "https://npb.jp/cl/": [
            "読売ジャイアンツ","東京ヤクルトスワローズ","中日ドラゴンズ",
            "広島東洋カープ","阪神タイガース","横浜DeNAベイスターズ",
        ],
        "https://npb.jp/pl/": [
            "北海道日本ハムファイターズ","オリックス・バファローズ",
            "東北楽天ゴールデンイーグルス","福岡ソフトバンクホークス",
            "千葉ロッテマリーンズ","埼玉西武ライオンズ",
        ],
    }
    teams = league_teams.get(source_url, [])
    if not teams:
        raise RuntimeError("Unsupported official NPB league source.")

    found: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for i, token in enumerate(tokens):
        matched = next((team for team in teams if token == team or team in token), None)
        if matched is None:
            continue
        if matched in seen:
            raise RuntimeError(
                "PIT starter gate failed: duplicate team tokens in official league starter order."
            )
        pitcher = ""
        for j in range(i + 1, min(i + 8, len(tokens))):
            candidate = tokens[j]
            if not candidate or candidate in teams:
                continue
            if re.fullmatch(r"\d{1,2}:\d{2}", candidate):
                continue
            if "予告先発" in candidate or "公式戦" in candidate:
                continue
            pitcher = candidate
            break
        if not pitcher:
            raise RuntimeError(
                f"PIT starter gate failed: no starter found for official team {matched}."
            )
        found.append((i, matched, pitcher))
        seen.add(matched)
        if len(found) == 6:
            break

    time_values = [x for x in tokens if re.fullmatch(r"\d{1,2}:\d{2}", x)]
    expected_games = len(time_values)
    required_records = 2 * expected_games
    if expected_games < int(min_games):
        raise RuntimeError(
            f"PIT starter gate failed: official league page resolved only {expected_games} "
            f"games; minimum required is {int(min_games)}."
        )
    if expected_games <= 0 or len(found) < required_records:
        raise RuntimeError(
            f"PIT starter gate failed: official league page resolved {len(found)} "
            f"team/starter records and {expected_games} times; expected at least "
            f"{required_records} records for {expected_games} games."
        )

    found = found[:required_records]
    out = []
    for game_idx in range(expected_games):
        a = found[2 * game_idx]
        b = found[2 * game_idx + 1]
        if a[2] == b[2]:
            raise RuntimeError(
                "PIT starter gate failed: identical starter assigned to both teams "
                f"in league-page evidence ({a[1]} vs {b[1]}): {a[2]!r}."
            )
        out.append({
            "home": a[1], "away": b[1],
            "home_starter": a[2], "away_starter": b[2],
            "confirmed_starters": True,
            "starter_evidence_status": "official_announced",
            "starter_source": source_url,
            "official_start_time": time_values[game_idx],
        })
    return out

def _load_official_starter_snapshot(target_date: str) -> list[dict] | None:
    path = ROOT / "data" / "official_starters" / f"{target_date}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "npb-official-starter-snapshot-v1":
        raise RuntimeError("Official starter snapshot schema mismatch; refusing prediction.")
    if payload.get("target_date") != target_date or payload.get("source_type") != "NPB_OFFICIAL":
        raise RuntimeError("Official starter snapshot provenance mismatch; refusing prediction.")
    retrieved = pd.Timestamp(payload.get("retrieved_at_utc"))
    if pd.isna(retrieved):
        raise RuntimeError("Official starter snapshot retrieval timestamp is invalid.")
    if retrieved.tzinfo is None:
        retrieved = retrieved.tz_localize("UTC")
    if retrieved > pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=5):
        raise RuntimeError("Official starter snapshot retrieval timestamp is in the future; refusing prediction.")
    games = payload.get("games", [])
    if not games:
        raise RuntimeError("Official starter snapshot contains no games.")
    if any(not isinstance(g, dict) or not g.get("official_start_time") for g in games):
        raise RuntimeError("Official starter snapshot contains a game without an official start time.")
    # NPB slates are not fixed at six games. Validate required fields while
    # allowing valid 5/6/other-game slates.
    required = ("home", "away", "home_starter", "away_starter", "official_start_time")
    if any(any(not str(g.get(k, "")).strip() for k in required) for g in games):
        raise RuntimeError("Official starter snapshot contains an incomplete game record.")
    for g in games:
        g["confirmed_starters"] = True
        g["starter_evidence_status"] = "official_announced_snapshot"
        g["starter_source"] = payload["source_url"]
    return games

def _starter_rows_sane(rows: list[dict]) -> bool:
    """Validate target-day starter rows before trusting parser output."""
    if not rows:
        return False
    teams = []
    starters = []
    for row in rows:
        home = _clean_name(str(row.get("home", "")))
        away = _clean_name(str(row.get("away", "")))
        hs = _clean_name(str(row.get("home_starter", "")))
        aws = _clean_name(str(row.get("away_starter", "")))
        if not home or not away or not hs or not aws:
            return False
        if home == away or hs == aws:
            return False
        teams.extend((home, away))
        starters.extend((hs, aws))
    if len(set(teams)) != len(teams):
        return False
    if len(set(starters)) != len(starters):
        return False
    return True


def official_starters(target_date: str) -> list[dict]:
    # An existing immutable snapshot is evidence, not an optional cache.
    # If it exists but is malformed, refuse prediction rather than silently
    # replacing a provenance problem with a live fetch.
    snapshot = _load_official_starter_snapshot(target_date)
    dedicated_error = None
    try:
        dedicated_rows = parse_official_starters_html(
            fetch_text(NPB_STARTER_URL + "?_ts=" + str(int(time.time()))), target_date
        )
        if _starter_rows_sane(dedicated_rows):
            return dedicated_rows
        dedicated_error = RuntimeError(
            "PIT starter gate failed: dedicated official starter parser produced "
            "structurally suspect target-day pairs."
        )
    except RuntimeError as exc:
        dedicated_error = exc
    try:
        if dedicated_error is not None and (
            "does not contain" not in str(dedicated_error)
            or True
        ):
            # First-party Central/Pacific pages are an independent official
            # reconciliation path. Use them whenever dedicated-page extraction
            # is suspect, not only when the page has rolled forward.
            league_rows = []
            for source_url in ("https://npb.jp/cl/", "https://npb.jp/pl/"):
                league_rows.extend(
                    parse_official_league_starters_html(
                        fetch_text(source_url + "?_ts=" + str(int(time.time()))),
                        target_date,
                        source_url,
                        min_games=1,
                    )
                )
            if _starter_rows_sane(league_rows):
                return league_rows
    except RuntimeError:
        league_rows = []
    if dedicated_error is not None:
        # A sane immutable official snapshot may be used as an evidence fallback;
        # otherwise preserve the original dedicated-parser failure explicitly.
        if snapshot is not None and _starter_rows_sane(snapshot):
            return snapshot
        raise dedicated_error

def _situation_tags(xrow: pd.DataFrame, regime_label: str) -> list[str]:
    """Convert PIT-safe matchup deltas into compact, auditable situation tags."""
    row = xrow.iloc[0]
    def side(value: float, margin: float) -> str:
        if value > margin:
            return "home"
        if value < -margin:
            return "away"
        return "balanced"
    tags = [f"regime:{regime_label}"]
    tags.append(f"starter:{side(float(row.get('starter_x_quality_proxy', 0.0)), 0.20)}")
    tags.append(f"starter_recent:{side(-float(row.get('starter_recent_form_gap', 0.0)), 0.15)}")
    tags.append(f"offense:{side(float(row.get('offense_power_gap_10', 0.0)), 0.01)}")
    tags.append(f"bullpen:{side(-float(row.get('bullpen_quality_era_diff', 0.0)), 0.15)}")
    tags.append(f"rest:{side(float(row.get('h_rest_days', 0.0)) - float(row.get('a_rest_days', 0.0)), 0.50)}")
    env = float(row.get('expected_env', 0.0))
    tags.append("environment:high" if env >= 4.8 else "environment:low" if env <= 3.2 else "environment:neutral")
    tags.append(f"trend:{side(float(row.get('run_trend_gap_20', 0.0)), 0.12)}")
    return tags

def _official_daily_start_times(target_date: str) -> dict[tuple[str, str], str]:
    """Read official game start times from NPB's date-specific schedule page.

    The English daily page publishes each game as team -> venue/time -> team.
    Parsing is bounded to team-pair segments and rejects ambiguous evidence.
    """
    url = NPB_DAY_URL.format(date=target_date.replace("-", ""))
    page_html = fetch_text(url)
    parser = _DailyScheduleTextParser()
    parser.feed(page_html)
    parts = [_clean_name(x) for x in parser.parts if _clean_name(x)]

    aliases = {
        **TEAM_MAP,
        "広島": "広島東洋カープ",
        "広島東洋カープ": "広島東洋カープ",
        "巨人": "読売ジャイアンツ",
        "読売": "読売ジャイアンツ",
        "ヤクルト": "東京ヤクルトスワローズ",
        "中日": "中日ドラゴンズ",
        "阪神": "阪神タイガース",
        "DeNA": "横浜DeNAベイスターズ",
        "横浜DeNAベイスターズ": "横浜DeNAベイスターズ",
        "日本ハム": "北海道日本ハムファイターズ",
        "オリックス": "オリックス・バファローズ",
        "楽天": "東北楽天ゴールデンイーグルス",
        "ソフトバンク": "福岡ソフトバンクホークス",
        "ロッテ": "千葉ロッテマリーンズ",
        "西武": "埼玉西武ライオンズ",
    }
    team_names = set(aliases) | set(aliases.values())
    time_re = re.compile(r"^\d{1,2}:\d{2}$")

    def canon(value: str) -> str:
        return aliases.get(value, value)

    candidates: dict[tuple[str, str], set[str]] = {}
    for i, token in enumerate(parts):
        if token not in team_names:
            continue
        home = canon(token)
        # The opponent is the next team label; venue and time may occur in
        # either order, so only the intervening time token is semantically used.
        for j in range(i + 1, min(i + 16, len(parts))):
            candidate = parts[j]
            if candidate not in team_names:
                continue
            away = canon(candidate)
            between = parts[i + 1:j]
            times = [x for x in between if time_re.fullmatch(x)]
            if len(times) == 1:
                candidates.setdefault((home, away), set()).add(times[0])
            # Once the next team is reached, do not skip it to another slate.
            break

    if not candidates:
        raise RuntimeError(
            f"Official NPB daily schedule did not yield game times for {target_date}."
        )

    resolved: dict[tuple[str, str], str] = {}
    for pair, values in candidates.items():
        if len(values) != 1:
            raise RuntimeError(
                f"Official NPB daily schedule has ambiguous start time for "
                f"{pair[0]} vs {pair[1]}: {sorted(values)}"
            )
        resolved[pair] = next(iter(values))

    # A duplicate reverse pairing with a different time is contradictory
    # evidence; fail closed rather than choosing one.
    for (home, away), value in resolved.items():
        reverse = (away, home)
        if reverse in resolved and resolved[reverse] != value:
            raise RuntimeError(
                f"Official NPB daily schedule contains contradictory home/away "
                f"pairs for {home} vs {away}."
            )
    return resolved


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def build_target_rows(target_date: str) -> pd.DataFrame:
    rows=official_starters(target_date)
    daily_times = _official_daily_start_times(target_date)
    now_utc=_utc_now()
    output=[]
    for i,r in enumerate(rows):
        r["league"]="NPB"; r["game_id"]=f"NPB-{target_date}-{i+1}"
        starter_time = str(r.get("official_start_time") or "").strip()
        pair = (str(r.get("home") or "").strip(), str(r.get("away") or "").strip())
        schedule_time = daily_times.get(pair)
        if not schedule_time:
            raise RuntimeError(
                f"Official NPB daily schedule has no exact time for {pair[0]} vs {pair[1]}; refusing prediction."
            )
        # The dedicated NPB starter page can contain non-game clock values
        # (for example the displayed average game duration). Treat the
        # official date-specific schedule as the sole authoritative source for
        # the target game's start time. The starter page remains authoritative
        # for starter identity/evidence, but its auxiliary time field is not
        # used as a competing clock signal.
        start_time = schedule_time
        r["official_start_time"] = start_time
        r["start_time_source"] = NPB_DAY_URL.format(date=target_date.replace("-", ""))
        r["datetime"]=pd.Timestamp(f"{target_date} {start_time}").tz_localize("Asia/Tokyo").tz_convert("UTC")
        # A completed or already-started game is never a valid future prediction
        # target. Keeping it in the production set would turn a day-of schedule
        # refresh into a post-start prediction.
        if r["datetime"] <= now_utc:
            continue
        r["home_score"]=float("nan"); r["away_score"]=float("nan")
        evidence_status = str(r.get("starter_evidence_status") or "").strip()
        if evidence_status not in {"official_announced", "official_announced_snapshot"}:
            raise RuntimeError(
                f"Unsupported NPB starter evidence status: {evidence_status!r}; refusing prediction."
            )
        source = str(r.get("starter_source") or "").strip()
        if not _is_official_source(source):
            raise RuntimeError(
                f"NPB starter evidence source is not allowlisted official NPB: {source!r}; refusing prediction."
            )
        r["starter_evidence_status"] = evidence_status
        r["starter_source"] = source
        r["starter_evidence_observed_at_utc"] = now_utc.isoformat()
        r["prediction_cutoff_utc"] = now_utc.isoformat()
        output.append(r)
    return pd.DataFrame(output)

def direct_pit_safe_lambdas(hist: pd.DataFrame, row: pd.Series, bt: BaseballBacktest) -> tuple[float,float,float]:
    """Fast independent PIT-safe run-rate model from historical games only.

    It deliberately avoids target-game-derived state and uses exponentially weighted
    team offense/defense plus announced-starter historical quality. This is the
    production fallback when the ML score ensemble is numerically degenerate.
    """
    league="NPB"; cutoff=pd.Timestamp(row["datetime"])
    h=norm_team(row["home"],league); a=norm_team(row["away"],league)
    hist=hist.copy()
    hist["_pit_dt"]=pd.to_datetime(hist["datetime"],errors="coerce",utc=True)
    if hist["_pit_dt"].isna().any():
        raise RuntimeError("PIT-safe fallback refused: historical datetime is malformed.")
    cutoff_utc=cutoff.tz_convert("UTC") if cutoff.tzinfo is not None else cutoff.tz_localize("UTC")
    # Never let a fallback model consume games at or after the target cutoff.
    hist=hist.loc[hist["_pit_dt"] < cutoff_utc].copy()
    if hist.empty:
        raise RuntimeError("PIT-safe fallback refused: no historical games before prediction cutoff.")
    hist=hist.sort_values(["_pit_dt","game_id"]).drop(columns=["_pit_dt"])
    def ew_team(team, side, scored, default):
        rows=[]
        for _,g in hist.iterrows():
            if norm_team(g["home"],league)==team:
                val=g["home_score"] if scored else g["away_score"]
                rows.append((g["datetime"],float(val)))
            elif norm_team(g["away"],league)==team:
                val=g["away_score"] if scored else g["home_score"]
                rows.append((g["datetime"],float(val)))
        if not rows:return default
        vals=np.asarray([v for _,v in rows[-30:]],dtype=float)
        w=np.exp(np.linspace(-2.2,0,len(vals)))
        return float(np.average(vals,weights=w))
    league_h=float(pd.to_numeric(hist["home_score"],errors="coerce").mean())
    league_a=float(pd.to_numeric(hist["away_score"],errors="coerce").mean())
    league_mean=max(1.0,0.5*(league_h+league_a))
    h_for=ew_team(h,"any",True,league_mean); h_against=ew_team(h,"any",False,league_mean)
    a_for=ew_team(a,"any",True,league_mean); a_against=ew_team(a,"any",False,league_mean)
    # Matchup run rates are team-specific and chronology-preserving. A small
    # home-field multiplier is applied after the PIT-safe team interaction.
    lh=max(.65,min(7.0,0.55*h_for+0.45*a_against))
    la=max(.65,min(7.0,0.55*a_for+0.45*h_against))
    lh*=1.035
    hs=bt.starter_features(league,str(row.get("home_starter","") or ""),cutoff,"hs_")
    aas=bt.starter_features(league,str(row.get("away_starter","") or ""),cutoff,"as_")
    # Shrunk starter adjustment: stronger historical FIP suppresses opponent scoring.
    lh*=float(np.exp(np.clip((float(aas.get("as_fip",4.0))-4.0)*0.055,-0.20,0.20)))
    la*=float(np.exp(np.clip((float(hs.get("hs_fip",4.0))-4.0)*0.055,-0.20,0.20)))
    return float(max(.65,min(7.0,lh))),float(max(.65,min(7.0,la))),0.0

def blend_classifier_run_share(lh: float, la: float, p: np.ndarray, weight: float = 0.25) -> tuple[float,float]:
    """Use the chronological classifier only for home/away run-share direction.

    The coherent score model controls total expected runs; the classifier adds a
    bounded matchup signal without allowing its NPB draw class to inflate final
    draws. All inputs are target-time PIT-safe.
    """
    total=max(float(lh)+float(la),1e-6)
    score_share=float(lh)/total
    hp=float(p[0]); ap=float(p[2])
    denom=max(hp+ap,1e-9)
    clf_share=hp/denom
    share=(1.0-weight)*score_share+weight*clf_share
    share=max(0.15,min(0.85,share))
    return total*share,total*(1.0-share)


def robust_target_lambdas(bt: BaseballBacktest, hist: pd.DataFrame, row: pd.Series) -> tuple[float,float,float]:
    """Fail-safe target-specific run model used only when the fitted score ensemble degenerates."""
    league="NPB"; dt=pd.Timestamp(row["datetime"])
    cutoff_utc=dt.tz_convert("UTC") if dt.tzinfo is not None else dt.tz_localize("UTC")
    hist=hist.copy()
    hist["_pit_dt"]=pd.to_datetime(hist["datetime"],errors="coerce",utc=True)
    if hist["_pit_dt"].isna().any():
        raise RuntimeError("PIT-safe fallback refused: historical datetime is malformed.")
    hist=hist.loc[hist["_pit_dt"] < cutoff_utc].copy()
    if hist.empty:
        raise RuntimeError("PIT-safe fallback refused: no historical games before prediction cutoff.")
    hist=hist.drop(columns=["_pit_dt"]).sort_values(["datetime","game_id"])
    hteam,a_team=norm_team(row["home"],league),norm_team(row["away"],league)
    hf=bt._team_features(league,hteam,"home",dt); af=bt._team_features(league,a_team,"away",dt)
    if hf.get("matches",0.0)<5 or af.get("matches",0.0)<5:
        raise RuntimeError("Target-specific state coverage too low; refusing fallback prediction.")
    gh=float(hist["home_score"].mean()); ga=float(hist["away_score"].mean())
    h_attack=0.65*hf.get("gf_10",gh)+0.35*gh
    a_attack=0.65*af.get("gf_10",ga)+0.35*ga
    h_def=0.65*af.get("ga_10",gh)+0.35*gh
    a_def=0.65*hf.get("ga_10",ga)+0.35*ga
    lh=0.52*h_attack+0.48*h_def
    la=0.52*a_attack+0.48*a_def
    hs=bt.starter_features(league,str(row.get("home_starter","") or ""),dt,"hs_")
    aas=bt.starter_features(league,str(row.get("away_starter","") or ""),dt,"as_")
    lh*=float(__import__("math").exp(0.07*(float(aas.get("as_fip",4.0))-4.0)))
    la*=float(__import__("math").exp(0.07*(float(hs.get("hs_fip",4.0))-4.0)))
    lh*=1.035
    lh=max(0.8,min(6.0,lh)); la=max(0.8,min(6.0,la))
    return lh,la,0.0

# Production starter ingestion hardening is regression-tested; preserve strict PIT blocking.
def predict(target_date: str, data_dir: str) -> dict:
    try:
        games=build_target_rows(target_date)
    except RuntimeError as exc:
        # Missing/insufficient official starter evidence is a valid fail-closed
        # production state. Persist it as BLOCKED_STARTERS so schedulers can
        # retry later without converting expected data unavailability into an
        # application failure. Ambiguity/corruption still raises below.
        message = str(exc)
        blocked_markers = (
            "Official NPB starter page does not contain",
            "PIT starter gate failed: expected at least",
            "PIT starter gate failed: no official game times found.",
            "PIT starter gate failed: identical starter assigned to both teams",
        )
        if not any(marker in message for marker in blocked_markers):
            raise
        result={
            "schema_version":"npb-production-v1", "target_date":target_date,
            "execution_status":"BLOCKED_STARTERS", "pit_status":"NOT_RUN",
            "starter_gate":"BLOCKED", "model_status":"NOT_RUN",
            "git_commit":__import__("os").environ.get("GITHUB_SHA","unknown"),
            "predictions":[], "block_reason":message,
            "prediction_generated_at":datetime.now(timezone.utc).isoformat(),
        }
        out=ROOT/"results"/f"npb_production_{target_date}.json"; out.parent.mkdir(exist_ok=True)
        atomic_write_json(out, result)
        return result
    if games.empty:
        result={
            "schema_version":"npb-production-v1", "target_date":target_date,
            "execution_status":"NO_FUTURE_GAMES", "pit_status":"PASS",
            "starter_gate":"PASS", "model_status":"NOT_RUN",
            "git_commit":__import__("os").environ.get("GITHUB_SHA","unknown"),
            "predictions":[], "block_reason":"all scheduled games for the requested JST date have already started or finished",
            "prediction_generated_at":datetime.now(timezone.utc).isoformat(),
        }
        out=ROOT/"results"/f"npb_production_{target_date}.json"; out.parent.mkdir(exist_ok=True)
        atomic_write_json(out, result)
        return result
    if not bool(games["confirmed_starters"].all()):
        raise RuntimeError("PIT gate failed: every target game must have confirmed official starters.")

    bt=BaseballBacktest(Path(data_dir))
    raw=bt.load_npb_pbp()
    hist=bt.aggregate_npb_games(raw)
    hist=hist[hist["datetime"] < games["datetime"].min()].copy()
    if hist["datetime"].max() >= games["datetime"].min():
        raise RuntimeError("PIT history contamination: historical data reaches target cutoff.")
    if hist["game_id"].isin(games["game_id"]).any():
        raise RuntimeError("PIT history contamination: target game appears in training history.")
    if len(hist) < 100:
        raise RuntimeError(f"Insufficient PIT-safe NPB history: {len(hist)} games.")
    hist_total = pd.to_numeric(hist["home_score"], errors="coerce") + pd.to_numeric(hist["away_score"], errors="coerce")
    hist_total = hist_total.replace([np.inf, -np.inf], np.nan).dropna()
    if len(hist_total) < 100:
        raise RuntimeError("Historical score labels are insufficient for production.")
    hist_score_mean = float(hist_total.mean())
    hist_score_zero_rate = float((hist_total <= 0).mean())
    if hist_score_mean < 3.0 or hist_score_zero_rate > 0.08 or hist_total.nunique() < 5:
        raise RuntimeError(
            f"Historical score data quality failed: mean_total={hist_score_mean:.3f}, "
            f"zero_rate={hist_score_zero_rate:.3f}, unique_totals={hist_total.nunique()}."
        )

    # Build chronological state from historical games only.
    # Production training must replay chronology; never build target rows into history.
    # The existing feature builder is the canonical chronological state constructor.
    X,y,meta=bt.build_features(hist)
    if len(X) != len(hist) or len(y) != len(hist):
        raise RuntimeError("Chronological feature contract failed: feature/label row count mismatch.")
    fitted, validation_scores, _=bt.fit_ensemble(X,y,"NPB")
    if not fitted:
        raise RuntimeError("Production ensemble fitting failed.")

    score_fit=bt.fit_score_ensemble(X,hist["home_score"].astype(float).values,hist["away_score"].astype(float).values,"NPB")
    outputs=[]
    for _,r in games.iterrows():
        xrow=pd.DataFrame([bt.match_features(r)]).replace([float("inf"),float("-inf")],float("nan"))
        if xrow.isna().any().any() or not np.isfinite(xrow.to_numpy(dtype=float)).all():
            raise RuntimeError("Production target feature vector contains undefined/non-finite values; refusing implicit imputation.")
        xrow=xrow.astype(float)
        p=bt.ensemble_proba(fitted,xrow,"NPB")[0]
        regime_label = str(bt._regime_router.labels(xrow)[0]) if getattr(bt, "_regime_router", None) is not None else "global"
        regime_model_weights = dict(getattr(bt, "_regime_weights", {}).get(regime_label, {}))
        score_regime_label = "global"
        score_regime_weights = {}
        if score_fit is not None and score_fit.get("regime_router") is not None:
            score_regime_label = str(score_fit["regime_router"].labels(xrow)[0])
            score_regime_weights = dict(score_fit.get("regime_weights", {}).get(score_regime_label, {}))
        lh,la,shared=bt.predict_scores(score_fit,xrow,"NPB")
        fallback_used = score_fit is None or (abs(lh-la)<1e-12 and abs(lh-2.35)<1e-12)
        if fallback_used:
            lh,la,shared=direct_pit_safe_lambdas(hist,r,bt)
            model_label="Production ML ensemble + PIT-safe direct run-rate fallback (degeneracy recovery)"
        else:
            model_label="BaseballBacktest.fit_ensemble + fit_score_ensemble + NPB extra-inning result calibration"
        lh,la=blend_classifier_run_share(lh,la,p,weight=0.25)
        scores=score_candidates(lh,la,shared,4)
        low,high=low_high_probs(lh,la,shared)
        # NPB final-result probabilities must distinguish a 9-inning tie from
        # a final draw. A tied regulation score can still be decided in innings
        # 10-12, so replace the raw 3-class classifier draw probability with the
        # coherent score/extra-inning result model while retaining the ensemble
        # probability difference as a small run-expectation adjustment above.
        home_final, draw_final, away_final = npb_final_outcomes(lh,la,shared)
        outputs.append({
          "game_id":r.game_id,"datetime_jst":pd.Timestamp(r.datetime).tz_convert("Asia/Tokyo").isoformat(),
          "home":r.home,"away":r.away,"home_starter":r.home_starter,"away_starter":r.away_starter,
          "starter_evidence_status":r.starter_evidence_status,
          "starter_evidence_observed_at_utc":r.starter_evidence_observed_at_utc,
          "prediction_cutoff_utc":r.prediction_cutoff_utc,
          "home_win_pct":round(float(home_final)*100,4),"draw_pct":round(float(draw_final)*100,4),"away_win_pct":round(float(away_final)*100,4),
          "low_pct":round(float(low)*100,4),"high_pct":round(float(high)*100,4),
          "top4_exact_scores":[{"score":s,"prob_pct":round(float(v)*100,4)} for s,v in scores],
          "lambda_home":float(lh),"lambda_away":float(la),"shared_lambda":float(shared),
          "model":model_label,
          "regime":regime_label,
          "classification_regime_model_weights":regime_model_weights,
          "score_regime":score_regime_label,
          "score_regime_model_weights":score_regime_weights,
          "situation_tags":_situation_tags(xrow, regime_label),
          "validation_scores":validation_scores,
          "historical_games_used":int(len(hist)),
          "pit_status":"PASS",
          "prediction_generated_at":datetime.now(timezone.utc).isoformat(),
        })
    result={"schema_version":"npb-production-v1","target_date":target_date,"execution_status":"EXECUTED",
            "pit_status":"PASS","starter_gate":"PASS","model_status":"FITTED_ON_PIT_SAFE_HISTORY",
            "git_commit":__import__("os").environ.get("GITHUB_SHA","unknown"),
            "data_quality_status":"PASS",
            "historical_score_mean_total":round(hist_score_mean,6),
            "historical_score_zero_rate":round(hist_score_zero_rate,6),
            "predictions":outputs}
    # Recover from silent cross-target model collapse instead of emitting a
    # misleadingly uniform forecast. The recovery remains PIT-safe because it
    # recomputes every target from historical games strictly before the target set.
    if len(outputs) >= 2:
        sig={(round(o["lambda_home"],6),round(o["lambda_away"],6),round(o["home_win_pct"],4),round(o["away_win_pct"],4)) for o in outputs}
        if len(sig) < 3:
            recovered=[]
            for idx, (_, r) in enumerate(games.iterrows()):
                xrow=pd.DataFrame([bt.match_features(r)]).replace([float("inf"),float("-inf")],float("nan"))
                if xrow.isna().any().any() or not np.isfinite(xrow.to_numpy(dtype=float)).all():
                    raise RuntimeError("Production target feature vector contains undefined/non-finite values; refusing implicit imputation.")
                xrow=xrow.astype(float)
                p=bt.ensemble_proba(fitted,xrow,"NPB")[0]
                recovery_regime_label = str(bt._regime_router.labels(xrow)[0]) if getattr(bt, "_regime_router", None) is not None else "global"
                recovery_regime_weights = dict(getattr(bt, "_regime_weights", {}).get(recovery_regime_label, {}))
                # Prefer the raw historical run-rate recovery here. It is
                # independent of target state and therefore remains PIT-safe, while
                # also avoiding the failure mode where a sparse team-state feature
                # vector collapses every target to the same lower clamp.
                lh,la,shared=direct_pit_safe_lambdas(hist,r,bt)
                recovery_model="Production ML ensemble + PIT-safe chronological direct run-rate recovery"
                if abs(lh-la) < 1e-6 and abs(lh-0.65) < 1e-6:
                    lh,la,shared=robust_target_lambdas(bt,hist,r)
                    recovery_model="Production ML ensemble + PIT-safe chronological team-state recovery"
                lh,la=blend_classifier_run_share(lh,la,p,weight=0.25)
                scores=score_candidates(lh,la,shared,4)
                low,high=low_high_probs(lh,la,shared)
                home_final,draw_final,away_final=npb_final_outcomes(lh,la,shared)
                o=outputs[idx].copy()
                o.update({
                    "home_win_pct":round(float(home_final)*100,4),
                    "draw_pct":round(float(draw_final)*100,4),
                    "away_win_pct":round(float(away_final)*100,4),
                    "low_pct":round(float(low)*100,4),
                    "high_pct":round(float(high)*100,4),
                    "top4_exact_scores":[{"score":s,"prob_pct":round(float(v)*100,4)} for s,v in scores],
                    "lambda_home":float(lh),"lambda_away":float(la),"shared_lambda":float(shared),
                    "model":recovery_model,
                    "regime":recovery_regime_label,
                    "classification_regime_model_weights":recovery_regime_weights,
                    "score_regime":"recovery",
                    "score_regime_model_weights":{},
                    "situation_tags":_situation_tags(xrow, recovery_regime_label),
                })
                recovered.append(o)
            outputs=recovered
            result["predictions"]=outputs
            sig={(round(o["lambda_home"],6),round(o["lambda_away"],6),round(o["home_win_pct"],4),round(o["away_win_pct"],4)) for o in outputs}
            if len(sig) < 2:
                debug = [
                    {
                        "home": o["home"], "away": o["away"],
                        "lambda_home": o["lambda_home"], "lambda_away": o["lambda_away"],
                        "home_win_pct": o["home_win_pct"], "away_win_pct": o["away_win_pct"],
                    }
                    for o in outputs
                ]
                teams = sorted(set(hist["home"].map(lambda x: norm_team(x, "NPB"))) | set(hist["away"].map(lambda x: norm_team(x, "NPB"))))
                print("PRODUCTION_DEGENERACY_DEBUG", json.dumps({"predictions": debug, "historical_team_count": len(teams), "historical_teams": teams}, ensure_ascii=False))
                raise RuntimeError("Production degeneracy guard: PIT-safe recovery remained insufficiently differentiated.")
    # Output validation: probabilities are finite, win probabilities sum to 100,
    # Low/High sum to 100, and exactly four score candidates exist.
    for o in outputs:
        probs = [float(o[k]) for k in ("home_win_pct", "draw_pct", "away_win_pct", "low_pct", "high_pct")]
        if not all(np.isfinite(v) and 0.0 <= v <= 100.0 for v in probs):
            raise RuntimeError("Production output validation failed: non-finite or out-of-range probability.")
        if abs(o["home_win_pct"]+o["draw_pct"]+o["away_win_pct"]-100) >= 0.05:
            raise RuntimeError("Production output validation failed: final-result probabilities do not sum to 100%.")
        if abs(o["low_pct"]+o["high_pct"]-100) >= 0.05:
            raise RuntimeError("Production output validation failed: Low/High probabilities do not sum to 100%.")
        exact = o["top4_exact_scores"]
        if len(exact) != 4 or len({s.get("score") for s in exact}) != 4:
            raise RuntimeError("Production output validation failed: Top4 exact scores are not four unique candidates.")
        exact_probs = [float(s.get("prob_pct", float("nan"))) for s in exact]
        if not all(np.isfinite(v) and 0.0 <= v <= 100.0 for v in exact_probs):
            raise RuntimeError("Production output validation failed: exact-score probability is invalid.")
    out=ROOT/"results"/f"npb_production_{target_date}.json"; out.parent.mkdir(exist_ok=True)
    atomic_write_json(out, result)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--date",required=True,help="YYYY-MM-DD, JST")
    ap.add_argument("--data-dir",default="data")
    args=ap.parse_args()
    print(json.dumps(predict(args.date,args.data_dir),ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

# Production execution trigger: use current JST date for manual verification.
