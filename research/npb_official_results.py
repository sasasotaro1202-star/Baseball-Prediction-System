"""Fail-closed NPB realized-score repair.

Realized labels are repaired from NPB.jp year-specific official schedule pages.
Parsing is deliberately resilient to table/cell markup changes while remaining
strict about team identity, date, and final score integrity.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
import re
import time
from typing import Iterable

import pandas as pd
import requests

DETAIL_URL = "https://npb.jp/games/{year}/schedule_{month:02d}_detail.html"
TEAM = {
    "巨人": "読売ジャイアンツ", "読売": "読売ジャイアンツ", "読売ジャイアンツ": "読売ジャイアンツ",
    "阪神": "阪神タイガース", "阪神タイガース": "阪神タイガース",
    "中日": "中日ドラゴンズ", "中日ドラゴンズ": "中日ドラゴンズ",
    "広島": "広島東洋カープ", "広島東洋": "広島東洋カープ", "広島東洋カープ": "広島東洋カープ",
    "ヤクルト": "東京ヤクルトスワローズ", "東京ヤクルト": "東京ヤクルトスワローズ", "東京ヤクルトスワローズ": "東京ヤクルトスワローズ",
    "DeNA": "横浜DeNAベイスターズ", "ＤｅＮＡ": "横浜DeNAベイスターズ", "横浜": "横浜DeNAベイスターズ", "横浜DeNA": "横浜DeNAベイスターズ", "横浜DeNAベイスターズ": "横浜DeNAベイスターズ",
    "ソフトバンク": "福岡ソフトバンクホークス", "福岡ソフトバンク": "福岡ソフトバンクホークス", "福岡ソフトバンクホークス": "福岡ソフトバンクホークス",
    "西武": "埼玉西武ライオンズ", "埼玉西武": "埼玉西武ライオンズ", "埼玉西武ライオンズ": "埼玉西武ライオンズ",
    "日本ハム": "北海道日本ハムファイターズ", "日ハム": "北海道日本ハムファイターズ", "北海道日本ハム": "北海道日本ハムファイターズ", "北海道日本ハムファイターズ": "北海道日本ハムファイターズ",
    "ロッテ": "千葉ロッテマリーンズ", "千葉ロッテ": "千葉ロッテマリーンズ", "千葉ロッテマリーンズ": "千葉ロッテマリーンズ",
    "楽天": "東北楽天ゴールデンイーグルス", "東北楽天": "東北楽天ゴールデンイーグルス", "東北楽天ゴールデンイーグルス": "東北楽天ゴールデンイーグルス",
    "オリックス": "オリックス・バファローズ", "オリックス・バファローズ": "オリックス・バファローズ",
}
TEAM_VALUES = frozenset(TEAM.values())
# Prefer long aliases first so substring matches such as 横浜 do not shadow DeNA.
ALIASES = sorted(TEAM, key=len, reverse=True)


def canon_team(value: object) -> str:
    s = "" if value is None else str(value).strip().replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s)
    return TEAM.get(s, s)


class _Rows(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append("\n")

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self.row is not None and self.cell is not None:
            text = "".join(self.cell).replace("\u00a0", " ").replace("\u3000", " ")
            text = re.sub(r"[ \t\r\f\v]+", " ", text)
            text = "\n".join(x.strip() for x in text.splitlines() if x.strip())
            self.row.append(text.strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None
            self.cell = None


def months_for_year(year: int) -> tuple[int, ...]:
    return tuple(range(6, 12)) if year == 2020 else tuple(range(3, 11))


def _request_rows(url: str, retries: int = 4) -> list[list[str]]:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30, headers={
                "User-Agent": "Baseball-Prediction-System/6.0",
                "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            })
            if r.status_code == 404:
                return []
            r.raise_for_status()
            # NPB is Japanese HTML; apparent_encoding is preferable to an ASCII
            # default when the server omits a useful charset declaration.
            if not r.encoding or r.encoding.lower() == "iso-8859-1":
                r.encoding = r.apparent_encoding or "utf-8"
            parser = _Rows()
            parser.feed(r.text)
            return parser.rows
        except Exception as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 6))
    raise RuntimeError(f"official NPB source failed: {url}: {last}")


def _teams_around_score(line: str, score_start: int, score_end: int) -> tuple[str | None, str | None]:
    """Find the nearest recognized NPB team before/after the score token."""
    before = line[:score_start]
    after = line[score_end:]
    candidates_before: list[tuple[int, str]] = []
    candidates_after: list[tuple[int, str]] = []
    for alias in ALIASES:
        pos = before.rfind(alias)
        if pos >= 0:
            candidates_before.append((pos, alias))
        pos = after.find(alias)
        if pos >= 0:
            candidates_after.append((pos, alias))
    if not candidates_before or not candidates_after:
        return None, None
    _, home_alias = max(candidates_before, key=lambda x: (x[0], len(x[1])))
    _, away_alias = min(candidates_after, key=lambda x: (x[0], -len(x[1])))
    home = canon_team(home_alias)
    away = canon_team(away_alias)
    if home not in TEAM_VALUES or away not in TEAM_VALUES or home == away:
        return None, None
    return home, away


def _parse_score_lines(cells: list[str], current_date: str, url: str) -> list[dict]:
    out: list[dict] = []
    # Do not assume spaces around the score or a fixed matchup column. The live
    # NPB pages have historically varied in nested tags and whitespace.
    score_re = re.compile(r"(?<!\d)(\d{1,2})\s*[-−–]\s*(\d{1,2})(?!\d)")
    for cell in cells:
        for raw_line in cell.splitlines():
            line = re.sub(r"\s+", " ", raw_line.replace("\u00a0", " ").replace("\u3000", " ")).strip()
            if not line:
                continue
            for match in score_re.finditer(line):
                home, away = _teams_around_score(line, match.start(), match.end())
                if home is None or away is None:
                    continue
                out.append({
                    "date": current_date,
                    "home": home,
                    "away": away,
                    "home_score": int(match.group(1)),
                    "away_score": int(match.group(2)),
                    "source_url": url,
                })
                break
    return out


def _fetch_month(year: int, month: int) -> list[dict]:
    url = DETAIL_URL.format(year=year, month=month)
    rows = _request_rows(url)
    date_re = re.compile(r"(?:^|\s)(\d{1,2})/(\d{1,2})(?:\s|$|[（(])")
    current_date: str | None = None
    out: list[dict] = []
    for row in rows:
        if not row:
            continue
        # Date can be in the first or second cell depending on NPB markup.
        for cell in row[:3]:
            dm = date_re.search(cell.strip())
            if dm:
                current_date = f"{year:04d}-{int(dm.group(1)):02d}-{int(dm.group(2)):02d}"
                break
        if current_date is not None:
            out.extend(_parse_score_lines(row, current_date, url))
    return out


def official_schedule(years: Iterable[int], cache_dir: str | Path) -> pd.DataFrame:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    years = sorted(set(int(y) for y in years))
    jobs = [(y, m) for y in years for m in months_for_year(y)]
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(12, max(1, len(jobs)))) as ex:
        pending = {}
        for year, month in jobs:
            path = cache / f"{year}-{month:02d}_official_v6.csv"
            if path.exists() and path.stat().st_size > 0:
                try:
                    rows.extend(pd.read_csv(path).to_dict("records"))
                    continue
                except Exception:
                    path.unlink(missing_ok=True)
            pending[ex.submit(_fetch_month, year, month)] = (year, month, path)
        for future in as_completed(pending):
            year, month, path = pending[future]
            got = future.result()
            if got:
                pd.DataFrame(got).to_csv(path, index=False)
                rows.extend(got)
    if not rows:
        return pd.DataFrame(columns=["date", "home", "away", "home_score", "away_score", "source_url"])
    schedule = pd.DataFrame(rows)
    schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    schedule["home"] = schedule["home"].map(canon_team)
    schedule["away"] = schedule["away"].map(canon_team)
    schedule["home_score"] = pd.to_numeric(schedule["home_score"], errors="coerce")
    schedule["away_score"] = pd.to_numeric(schedule["away_score"], errors="coerce")
    schedule = schedule.dropna(subset=["date", "home", "away", "home_score", "away_score"])
    schedule = schedule.drop_duplicates(["date", "home", "away"], keep="last").reset_index(drop=True)
    return schedule


def _game_score_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"game_id", "date", "home", "away", "home_score", "away_score"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=["game_id", "date_key", "home_key", "away_key", "home_score", "away_score"])
    x = frame.copy()
    x["game_id"] = x["game_id"].astype(str).str.strip()
    x["date_key"] = pd.to_datetime(x["date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    x["home_key"] = x["home"].map(canon_team)
    x["away_key"] = x["away"].map(canon_team)
    x["home_score"] = pd.to_numeric(x["home_score"], errors="coerce")
    x["away_score"] = pd.to_numeric(x["away_score"], errors="coerce")
    x = x.dropna(subset=["game_id", "date_key", "home_key", "away_key"])
    scored = x.dropna(subset=["home_score", "away_score"])
    if scored.empty:
        return pd.DataFrame(columns=["game_id", "date_key", "home_key", "away_key", "home_score", "away_score"])
    conflicts = scored.groupby("game_id").agg(hn=("home_score", "nunique"), an=("away_score", "nunique"))
    if ((conflicts["hn"] > 1) | (conflicts["an"] > 1)).any():
        bad = int(((conflicts["hn"] > 1) | (conflicts["an"] > 1)).sum())
        raise RuntimeError(f"NPB target integrity failure: conflicting PBP scores in {bad} games")
    return scored.sort_values(["game_id"]).drop_duplicates("game_id")[["game_id", "date_key", "home_key", "away_key", "home_score", "away_score"]]


def repair_scores(frame: pd.DataFrame, data_dir: str | Path) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["home_score"] = pd.to_numeric(out["home_score"], errors="coerce")
    out["away_score"] = pd.to_numeric(out["away_score"], errors="coerce")
    if out[["home_score", "away_score"]].notna().all(axis=1).all():
        return out

    games = out[["game_id", "date", "home", "away", "home_score", "away_score"]].drop_duplicates("game_id").copy()
    games["date_key"] = pd.to_datetime(games["date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    games["home_key"] = games["home"].map(canon_team)
    games["away_key"] = games["away"].map(canon_team)

    local = _game_score_from_frame(out)
    if not local.empty:
        by_id = local[["game_id", "home_score", "away_score"]].rename(columns={"home_score": "local_h", "away_score": "local_a"})
        games = games.merge(by_id, on="game_id", how="left", validate="one_to_one")
        games["home_score"] = games["home_score"].fillna(games["local_h"])
        games["away_score"] = games["away_score"].fillna(games["local_a"])
        games = games.drop(columns=["local_h", "local_a"])

    unresolved = games[["home_score", "away_score"]].isna().any(axis=1)
    if unresolved.any():
        years = pd.to_datetime(games.loc[unresolved, "date"], errors="coerce", utc=True).dt.year.dropna().astype(int).unique().tolist()
        schedule = official_schedule(years, Path(data_dir) / ".npb_official_schedule_cache")
        if schedule.empty:
            raise RuntimeError("NPB official result source returned no completed score rows")
        schedule["date_key"] = schedule["date"].astype(str)
        schedule["home_key"] = schedule["home"].map(canon_team)
        schedule["away_key"] = schedule["away"].map(canon_team)
        schedule = schedule.drop_duplicates(["date_key", "home_key", "away_key"], keep="last")
        repair = schedule[["date_key", "home_key", "away_key", "home_score", "away_score"]].rename(columns={"home_score": "official_h", "away_score": "official_a"})
        games = games.merge(repair, on=["date_key", "home_key", "away_key"], how="left", validate="one_to_one")
        games["home_score"] = games["home_score"].fillna(games["official_h"])
        games["away_score"] = games["away_score"].fillna(games["official_a"])
        games = games.drop(columns=["official_h", "official_a"])

    unresolved = games[["home_score", "away_score"]].isna().any(axis=1)
    if unresolved.any():
        sample = games.loc[unresolved, ["game_id", "date_key", "home_key", "away_key"]].head(10).to_dict("records")
        raise RuntimeError(f"NPB score repair left {int(unresolved.sum())} unresolved games; sample={sample}")

    score_map = games.set_index("game_id")[["home_score", "away_score"]]
    out = out.drop(columns=["home_score", "away_score"]).merge(score_map, left_on="game_id", right_index=True, how="left", validate="many_to_one")
    if out[["home_score", "away_score"]].isna().any().any():
        raise RuntimeError("NPB score repair failed to map verified game scores back to PBP rows")
    return out


def load_repaired_pbp(data_dir: str | Path) -> pd.DataFrame:
    from data.npb_pbp_adapter import normalize_pbp_frame

    root = Path(data_dir)
    files = sorted(root.glob("*_pbp.csv"))
    if not files:
        raise FileNotFoundError("No NPB PBP files found in data directory")
    frames = []
    for path in files:
        normalized = normalize_pbp_frame(pd.read_csv(path, low_memory=False), data_dir=None)
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        raise RuntimeError("NPB PBP files were found but none contained usable game rows")
    return repair_scores(pd.concat(frames, ignore_index=True, sort=False), root)
