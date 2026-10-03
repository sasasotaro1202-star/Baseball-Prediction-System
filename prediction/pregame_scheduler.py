"""Lightweight, dependency-free pregame scheduler for production baseball predictions.

The automatic slot targets roughly 60 minutes before first pitch, using a
bounded 50-60 minute window. It never predicts anything itself.
Unknown/ambiguous schedule evidence fails closed. Manual/current-production
prediction calls are independent of this scheduler and may be requested at
other times.

Currently the checked-in production runtime is NPB. When additional runtimes are
formally adopted, this scheduler can discover them from the runtime registry
without changing the cron policy.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
import http.client
import time
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "current_production_runtime.json"
PRED_DIR = ROOT / "data" / "experience" / "predictions"
SHADOW_PRED_DIR = ROOT / "data" / "experience" / "research_shadow" / "predictions"
JST = ZoneInfo("Asia/Tokyo")
NPB_DAY_URL = "https://npb.jp/bis/eng/{year}/games/gm{date}.html"
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start}&endDate={end}&hydrate=probablePitcher"
TEAM_ALIASES = {
    "Yomiuri": "読売ジャイアンツ",
    "Yakult": "東京ヤクルトスワローズ",
    "Chunichi": "中日ドラゴンズ",
    "Hiroshima": "広島東洋カープ",
    "Hanshin": "阪神タイガース",
    "DeNA": "横浜DeNAベイスターズ",
    "Nippon-Ham": "北海道日本ハムファイターズ",
    "ORIX": "オリックス・バファローズ",
    "Rakuten": "東北楽天ゴールデンイーグルス",
    "SoftBank": "福岡ソフトバンクホークス",
    "Lotte": "千葉ロッテマリーンズ",
    "Seibu": "埼玉西武ライオンズ",
    "広島": "広島東洋カープ",
    "巨人": "読売ジャイアンツ",
    "読売": "読売ジャイアンツ",
    "ヤクルト": "東京ヤクルトスワローズ",
    "中日": "中日ドラゴンズ",
    "阪神": "阪神タイガース",
    "日本ハム": "北海道日本ハムファイターズ",
    "オリックス": "オリックス・バファローズ",
    "楽天": "東北楽天ゴールデンイーグルス",
    "ソフトバンク": "福岡ソフトバンクホークス",
    "ロッテ": "千葉ロッテマリーンズ",
    "西武": "埼玉西武ライオンズ",
}
TEAM_NAMES = set(TEAM_ALIASES) | set(TEAM_ALIASES.values())


class _ScheduleParser(HTMLParser):
    """Parse the official schedule section for team labels and clock tokens.

    NPB's official pages contain unrelated team links in navigation/footer
    regions. The parser therefore starts only after an explicit schedule
    heading and then accepts both image-alt and visible-text team labels.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tokens: list[tuple[str, str]] = []
        self._hidden = 0
        self._heading_depth: int | None = None
        self._heading_parts: list[str] = []
        self._schedule_started = False

    def _append_team(self, value: str) -> None:
        value = " ".join(str(value).replace("　", " ").split())
        compact = value.replace(" ", "")
        canonical = TEAM_ALIASES.get(
            value,
            TEAM_ALIASES.get(compact, compact),
        )
        if canonical not in TEAM_NAMES:
            return
        token = ("team", canonical)
        # Responsive DOMs may expose the same label through both img[alt] and
        # adjacent visible text. Collapse only adjacent identical labels.
        if self.tokens and self.tokens[-1] == token:
            return
        self.tokens.append(token)

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        attrs_map = {k: v or "" for k, v in attrs}
        if tag in {"script", "style", "noscript", "template"}:
            self._hidden += 1
            return
        if self._hidden:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and not self._schedule_started:
            self._heading_depth = 1
            self._heading_parts = []
            return
        if self._heading_depth is not None:
            self._heading_depth += 1
            return
        if not self._schedule_started:
            return
        if tag == "img":
            self._append_team(attrs_map.get("alt") or "")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template"}:
            self._hidden = max(0, self._hidden - 1)
            return
        if self._heading_depth is None or self._hidden:
            return
        self._heading_depth -= 1
        if self._heading_depth == 0:
            heading = " ".join(self._heading_parts).strip()
            normalized = heading.replace("　", " ")
            if (
                "Regular Season (Schedules)" in normalized
                or "公式戦" in normalized and any(
                    marker in normalized for marker in ("試合予定", "試合結果", "試合日程")
                )
            ):
                self._schedule_started = True
            self._heading_parts = []
            self._heading_depth = None

    def handle_data(self, data: str) -> None:
        if self._hidden:
            return
        value = " ".join(str(data).replace("　", " ").split())
        if self._heading_depth is not None:
            if value:
                self._heading_parts.append(value)
            return
        if not self._schedule_started:
            return
        compact = value.replace(" ", "")
        if value in TEAM_NAMES or compact in TEAM_NAMES:
            self._append_team(value)
            return
        if len(value) == 5 and value[2] == ":" and value[:2].isdigit() and value[3:].isdigit():
            hh = int(value[:2])
            mm = int(value[3:])
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                self.tokens.append(("time", value))



def load_runtimes() -> dict[str, dict]:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "current-production-runtime-v1":
        raise RuntimeError("unsupported current production runtime schema")
    runtimes = payload.get("runtimes")
    if not isinstance(runtimes, dict):
        raise RuntimeError("production runtime registry has no runtimes")
    return runtimes


def _fetch(url: str) -> str:
    """Fetch schedule data with stdlib-only HTTP and bounded retry/backoff."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"unsupported schedule URL: {url}")
    path_with_query = parsed.path or "/"
    if parsed.query:
        path_with_query += "?" + parsed.query
    last_error = None
    for attempt in range(1, 5):
        connection = http.client.HTTPSConnection(parsed.netloc, timeout=30)
        try:
            connection.request(
                "GET",
                path_with_query,
                headers={"User-Agent": "Baseball-Prediction-System/pregame-scheduler"},
            )
            response = connection.getresponse()
            payload = response.read()
            if 200 <= response.status < 300:
                return payload.decode("utf-8", errors="strict")
            last_error = RuntimeError(f"HTTP {response.status} for {url}")
        except (OSError, UnicodeError) as exc:
            last_error = exc
        finally:
            connection.close()
        if attempt < 4:
            time.sleep(float(attempt))
    raise RuntimeError(f"schedule fetch failed after 4 attempts: {url}") from last_error

def _schedule_for_date(target_date: str) -> list[dict]:
    html = _fetch(NPB_DAY_URL.format(year=target_date[:4], date=target_date.replace("-", "")))
    parser = _ScheduleParser()
    parser.feed(html)
    teams: list[str] = []
    games: list[dict] = []
    for index, (kind, value) in enumerate(parser.tokens):
        if kind != "team":
            continue
        if teams and value == teams[-1]:
            continue
        teams.append(value)
        if len(teams) >= 2:
            # Search only the tokens between the latest two teams for exactly one
            # clock. Anything ambiguous is rejected rather than guessed.
            prior_team_positions = [
                i for i, tok in enumerate(parser.tokens[:index]) if tok[0] == "team"
            ]
            prev_index = prior_team_positions[-1] if prior_team_positions else None
            if prev_index is None:
                continue
            clocks = [v for k, v in parser.tokens[prev_index + 1 : index] if k == "time"]
            if len(clocks) != 1:
                continue
            games.append({
                "home": teams[-2],
                "away": teams[-1],
                "official_start_time": clocks[0],
            })
    # Remove duplicate DOM presentations only when the entire game identity and
    # time are identical. Conflicting evidence is never silently collapsed.
    keyed: dict[tuple[str, str, str], dict] = {}
    for game in games:
        key = (game["home"], game["away"], game["official_start_time"])
        keyed[key] = game
    out = list(keyed.values())
    if not out:
        raise RuntimeError(f"official NPB daily schedule yielded no deterministic games: {target_date}")
    return out




def load_research_active_competitions() -> set[str]:
    """Return active implemented research competitions without changing production state."""
    from prediction.scope_router import build_scope

    scope = build_scope()
    return {str(x).strip().upper() for x in scope.get("research_active", []) if str(x).strip()}


def _mlb_schedule_for_date(target_date: str) -> list[dict]:
    """Discover MLB games whose first pitch falls on the requested JST date.

    MLB schedule discovery is research-only. Probable pitchers are not treated
    as official starter evidence and cannot unlock production.
    """
    target = datetime.fromisoformat(f"{target_date}T00:00:00+09:00")
    start = (target - timedelta(days=1)).date().isoformat()
    end = (target + timedelta(days=1)).date().isoformat()
    raw = _fetch(MLB_SCHEDULE_URL.format(start=start, end=end))
    payload = json.loads(raw)
    games: list[dict] = []
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            game_date = str(game.get("gameDate") or "").strip()
            if not game_date:
                continue
            try:
                start_utc = datetime.fromisoformat(game_date.replace("Z", "+00:00"))
            except ValueError:
                continue
            start_jst = start_utc.astimezone(JST)
            if start_jst.date().isoformat() != target_date:
                continue
            teams = game.get("teams") or {}
            home = teams.get("home") or {}
            away = teams.get("away") or {}
            home_team = str((home.get("team") or {}).get("name") or "").strip()
            away_team = str((away.get("team") or {}).get("name") or "").strip()
            if not home_team or not away_team:
                continue
            hp = str((home.get("probablePitcher") or {}).get("fullName") or "").strip()
            ap = str((away.get("probablePitcher") or {}).get("fullName") or "").strip()
            game_id = str(game.get("gamePk") or "").strip()
            if not game_id:
                raise RuntimeError(f"MLB research schedule contains game without stable gamePk: {target_date}")
            games.append({
                "game_id": game_id,
                "home": home_team,
                "away": away_team,
                "official_start_time": start_jst.strftime("%H:%M"),
                "scheduled_start_utc": start_utc.isoformat(),
                "home_starter": hp,
                "away_starter": ap,
                "starter_evidence_status": "official_probable_only" if hp and ap else "missing",
                "starter_source": "MLB Stats API schedule",
                "pit_status": "NOT_ELIGIBLE_OFFICIAL_STARTER_REQUIRED",
            })
    return sorted(
        games,
        key=lambda g: (g["scheduled_start_utc"], g["game_id"]),
    )

def _archived_prediction_keys(target_date: str) -> set[tuple[str, str, str]]:
    path = PRED_DIR / f"{target_date}.jsonl"
    if not path.exists():
        return set()
    out: set[tuple[str, str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            out.add((
                str(row["home"]),
                str(row["away"]),
                str(row["prediction_cutoff_utc"]),
            ))
        except (KeyError, TypeError):
            continue
    return out


def _archived_prediction_sources(target_date: str) -> set[tuple[str, str, str]]:
    """Return labeled automatic sources already archived in either experience lane."""
    out: set[tuple[str, str, str]] = set()
    for base in (PRED_DIR, SHADOW_PRED_DIR):
        path = base / f"{target_date}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            source = str(row.get("prediction_source") or "").strip()
            if not source:
                continue
            try:
                out.add((str(row["home"]), str(row["away"]), source))
            except (KeyError, TypeError):
                continue
    return out


def due_games(
    *,
    now_utc: datetime | None = None,
    min_lead_minutes: float = 0.0,
    scan_ahead_minutes: float = 60.0,
    preferred_lead_minutes: float = 30.0,
    prediction_source: str | None = None,
) -> dict:
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    runtimes = load_runtimes()
    enabled = [
        (league, runtime)
        for league, runtime in runtimes.items()
        if str(runtime.get("formal_adoption_status", "")).upper() == "CURRENT_PRODUCTION"
    ]
    blocked: list[dict] = []
    if not enabled:
        # Production being fully blocked is a valid governed state, not a
        # scheduler crash. Preserve the fail-closed production boundary while
        # allowing research-only discovery to continue independently.
        blocked.extend(
            {
                "league": league,
                "status": "BLOCKED_NO_CURRENT_PRODUCTION_RUNTIME",
                "entrypoint": runtime.get("entrypoint"),
            }
            for league, runtime in runtimes.items()
        )

    due: list[dict] = []
    research_due: list[dict] = []
    research_errors: list[dict] = []
    research_scope_error = None
    try:
        research_active = load_research_active_competitions()
    except Exception as exc:
        # Research scope discovery is isolated from current-production scheduling.
        research_active = set()
        research_scope_error = f"{type(exc).__name__}: {exc}"
    # Current-production policy requires the call-time JST target date.
    # Do not precompute tomorrow's production forecast through this dispatcher.
    dates: set[str] = {now.astimezone(JST).date().isoformat()}
    for league, runtime in enabled:
        if league != "NPB":
            blocked.append({
                "league": league,
                "status": "BLOCKED_UNSUPPORTED_PREGAME_SCHEDULER",
                "entrypoint": runtime.get("entrypoint"),
            })
            continue
        for target_date in sorted(dates):
            games = _schedule_for_date(target_date)
            archived = _archived_prediction_keys(target_date)
            archived_sources = _archived_prediction_sources(target_date)
            for game_index, game in enumerate(games, start=1):
                start = datetime.fromisoformat(
                    f"{target_date}T{game['official_start_time']}:00+09:00"
                ).astimezone(timezone.utc)
                lead = (start - now).total_seconds() / 60.0
                preferred_cutoff = start - timedelta(minutes=float(preferred_lead_minutes))
                prediction_cutoff = now
                cutoff_iso = prediction_cutoff.isoformat()
                source_key = str(prediction_source or "").strip()
                already_sourced = bool(
                    source_key
                    and (game["home"], game["away"], source_key) in archived_sources
                )
                if (
                    float(min_lead_minutes) < lead <= float(scan_ahead_minutes)
                    and (game["home"], game["away"], cutoff_iso) not in archived
                    and not already_sourced
                ):
                    due.append({
                        "league": league,
                        "target_date": target_date,
                        "game_index": game_index,
                        "home": game["home"],
                        "away": game["away"],
                        "official_start_time": game["official_start_time"],
                        "prediction_cutoff_utc": cutoff_iso,
                        "preferred_prediction_cutoff_utc": preferred_cutoff.isoformat(),
                        # Keep the legacy field for schema compatibility, but
                        # expose the semantically correct preferred-target field.
                        "preferred_target_met": bool(now <= preferred_cutoff),
                        "preferred_30m_met": bool(now <= preferred_cutoff),
                        "lead_minutes": round(lead, 3),
                        "prediction_source": source_key or None,
                        "prediction_eligibility": "CURRENT_PRODUCTION",
                        "status": "DUE",
                    })
    research_shadow_due: list[dict] = []
    if "NPB" not in {league for league, _ in enabled}:
        for target_date in sorted(dates):
            try:
                games = _schedule_for_date(target_date)
            except Exception as exc:
                research_errors.append({
                    "league": "NPB",
                    "target_date": target_date,
                    "mode": "research_shadow",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                games = []
            archived_sources = _archived_prediction_sources(target_date)
            for game_index, game in enumerate(games, start=1):
                start = datetime.fromisoformat(
                    f"{target_date}T{game['official_start_time']}:00+09:00"
                ).astimezone(timezone.utc)
                lead = (start - now).total_seconds() / 60.0
                source = "RESEARCH_SHADOW_AUTO_60M"
                if (
                    float(min_lead_minutes) < lead <= min(float(scan_ahead_minutes), 60.0)
                    and (game["home"], game["away"], source) not in archived_sources
                ):
                    preferred_cutoff = start - timedelta(minutes=60.0)
                    research_shadow_due.append({
                        "league": "NPB",
                        "target_date": target_date,
                        "game_index": game_index,
                        "home": game["home"],
                        "away": game["away"],
                        "official_start_time": game["official_start_time"],
                        "prediction_cutoff_utc": now.isoformat(),
                        "preferred_prediction_cutoff_utc": preferred_cutoff.isoformat(),
                        "preferred_prediction_target_lead_minutes": 60.0,
                        "preferred_60m_met": bool(now <= preferred_cutoff),
                        "lead_minutes": round(lead, 3),
                        "prediction_source": source,
                        "prediction_eligibility": "RESEARCH_SHADOW_PIT_SAFE_STARTERS_REQUIRED",
                        "status": "RESEARCH_SHADOW_DUE",
                    })

    # Research discovery runs alongside production scanning but can never add
    # a row to due_games/due_dates. MLB remains research-only until the separate
    # official-starter PIT gate and production adoption requirements are satisfied.
    if "MLB" in research_active and "MLB" not in {league for league, _ in enabled}:
        for target_date in sorted(dates):
            try:
                games = _mlb_schedule_for_date(target_date)
            except Exception as exc:
                research_errors.append({
                    "league": "MLB",
                    "target_date": target_date,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            for game_index, game in enumerate(games, start=1):
                start = datetime.fromisoformat(str(game["scheduled_start_utc"])).astimezone(timezone.utc)
                lead = (start - now).total_seconds() / 60.0
                if float(min_lead_minutes) < lead <= float(scan_ahead_minutes):
                    preferred_cutoff = start - timedelta(minutes=float(preferred_lead_minutes))
                    research_due.append({
                        **game,
                        "league": "MLB",
                        "target_date": target_date,
                        "game_index": game_index,
                        "prediction_cutoff_utc": now.isoformat(),
                        "preferred_prediction_cutoff_utc": preferred_cutoff.isoformat(),
                        "preferred_30m_met": bool(now <= preferred_cutoff),
                        "lead_minutes": round(lead, 3),
                        "prediction_eligibility": "RESEARCH_ONLY_BLOCKED_UNTIL_OFFICIAL_STARTERS",
                        "status": "RESEARCH_DUE",
                    })

    return {
        "schema_version": "baseball-pregame-scheduler-v1",
        "checked_at_utc": now.isoformat(),
        "min_lead_minutes": float(min_lead_minutes),
        "preferred_lead_minutes": float(preferred_lead_minutes),
        "scan_ahead_minutes": float(scan_ahead_minutes),
        "prediction_source": str(prediction_source or "") or None,
        "runtimes": sorted([league for league, _ in enabled]),
        "research_active": sorted(research_active),
        "due_games": due,
        "due_dates": sorted({row["target_date"] for row in due}),
        "research_due_games": research_due,
        "research_due_dates": sorted({row["target_date"] for row in research_due}),
        "research_shadow_due_games": research_shadow_due,
        "research_shadow_due_dates": sorted({row["target_date"] for row in research_shadow_due}),
        "research_errors": research_errors,
        "research_scope_error": research_scope_error,
        "blocked_runtimes": blocked,
        "research_status": (
            "RESEARCH_DUE"
            if research_due
            else "RESEARCH_ERROR"
            if research_errors or research_scope_error
            else "NO_RESEARCH_DUE"
        ),
        "status": "DUE" if due else "NO_DUE_GAMES",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates-only", action="store_true")
    parser.add_argument("--min-lead-minutes", type=float, default=0.0)
    parser.add_argument("--preferred-lead-minutes", type=float, default=60.0)
    parser.add_argument("--scan-ahead-minutes", type=float, default=70.0)
    parser.add_argument(
        "--prediction-source",
        default=None,
        help="Optional stable source label used to prevent duplicate scheduled slots.",
    )
    args = parser.parse_args(argv)
    result = due_games(
        min_lead_minutes=args.min_lead_minutes,
        preferred_lead_minutes=args.preferred_lead_minutes,
        scan_ahead_minutes=args.scan_ahead_minutes,
        prediction_source=args.prediction_source,
    )
    if args.dates_only:
        print(",".join(result["due_dates"]))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
