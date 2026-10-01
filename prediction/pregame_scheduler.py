"""Lightweight, dependency-free pregame scheduler for production baseball predictions.

The scheduler only decides whether a currently production-enabled target has a
game inside the upcoming pregame window. Thirty minutes is the preferred target,
not a hard eligibility boundary. It never predicts anything
itself. Unknown/ambiguous schedule evidence fails closed.

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
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "current_production_runtime.json"
PRED_DIR = ROOT / "data" / "experience" / "predictions"
JST = ZoneInfo("Asia/Tokyo")
NPB_DAY_URL = "https://npb.jp/bis/eng/{year}/games/gm{date}.html"\nMLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start}&endDate={end}&hydrate=probablePitcher"
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
    """Collect only team image alts and visible clock tokens in document order."""

    def __init__(self) -> None:
        super().__init__()
        self.tokens: list[tuple[str, str]] = []
        self._hidden = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        attrs_map = {k: v or "" for k, v in attrs}
        if tag in {"script", "style", "noscript", "template"}:
            self._hidden += 1
            return
        if self._hidden:
            return
        if tag == "img":
            alt = (attrs_map.get("alt") or "").strip()
            if alt in TEAM_NAMES:
                self.tokens.append(("team", TEAM_ALIASES.get(alt, alt)))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "template"}:
            self._hidden = max(0, self._hidden - 1)

    def handle_data(self, data: str) -> None:
        if self._hidden:
            return
        value = " ".join(str(data).replace("　", " ").split())
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
    req = Request(url, headers={"User-Agent": "Baseball-Prediction-System/pregame-scheduler"})
    with urlopen(req, timeout=20) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="strict")


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


def due_games(*, now_utc: datetime | None = None, min_lead_minutes: float = 0.0, scan_ahead_minutes: float = 60.0, preferred_lead_minutes: float = 30.0) -> dict:
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    runtimes = load_runtimes()
    enabled = [
        (league, runtime)
        for league, runtime in runtimes.items()
        if str(runtime.get("formal_adoption_status", "")).upper() == "CURRENT_PRODUCTION"
    ]
    if not enabled:
        raise RuntimeError("no current production runtime is registered")

    due: list[dict] = []
    research_due: list[dict] = []
    blocked: list[dict] = []
    research_active = load_research_active_competitions()
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
            for game_index, game in enumerate(games, start=1):
                start = datetime.fromisoformat(
                    f"{target_date}T{game['official_start_time']}:00+09:00"
                ).astimezone(timezone.utc)
                lead = (start - now).total_seconds() / 60.0
                preferred_cutoff = start - timedelta(minutes=float(preferred_lead_minutes))
                prediction_cutoff = now
                cutoff_iso = prediction_cutoff.isoformat()
                if (
                    float(min_lead_minutes) < lead <= float(scan_ahead_minutes)
                    and (game["home"], game["away"], cutoff_iso) not in archived
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
                        "preferred_30m_met": bool(now <= preferred_cutoff),
                        "lead_minutes": round(lead, 3),
                        "prediction_eligibility": "CURRENT_PRODUCTION",
                        "status": "DUE",
                    })
    # Research discovery runs alongside production scanning but can never add
    # a row to due_games/due_dates. MLB remains research-only until the separate
    # official-starter PIT gate and production adoption requirements are satisfied.
    if "MLB" in research_active and "MLB" not in {league for league, _ in enabled}:
        for target_date in sorted(dates):
            games = _mlb_schedule_for_date(target_date)
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
        "runtimes": sorted([league for league, _ in enabled]),
        "due_games": due,
        "due_dates": sorted({row["target_date"] for row in due}),
        "blocked_runtimes": blocked,
        "status": "DUE" if due else "NO_DUE_GAMES",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates-only", action="store_true")
    parser.add_argument("--min-lead-minutes", type=float, default=0.0)
    parser.add_argument("--preferred-lead-minutes", type=float, default=30.0)
    parser.add_argument("--scan-ahead-minutes", type=float, default=60.0)
    args = parser.parse_args(argv)
    result = due_games(
        min_lead_minutes=args.min_lead_minutes,
        preferred_lead_minutes=args.preferred_lead_minutes,
        scan_ahead_minutes=args.scan_ahead_minutes,
    )
    if args.dates_only:
        print(",".join(result["due_dates"]))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
