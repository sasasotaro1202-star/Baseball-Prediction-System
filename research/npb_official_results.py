"""Fast, fail-closed NPB realized-score repair.

Historical realized labels are read from versioned repository result tables first;
the official NPB result pages are used only for unresolved rows (for example the
current season). This keeps historical OOS runs deterministic and avoids a large
network scrape on every run. No realized-label source is used for pre-game features.
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
TEAM_RESULT_URL = "https://npb.jp/bis/teams/results_{code}_{month:02d}.html"
TEAM_RESULT_CODES = {
    "g": "読売ジャイアンツ", "t": "阪神タイガース", "d": "中日ドラゴンズ",
    "c": "広島東洋カープ", "s": "東京ヤクルトスワローズ", "db": "横浜DeNAベイスターズ",
    "h": "福岡ソフトバンクホークス", "f": "北海道日本ハムファイターズ",
    "e": "東北楽天ゴールデンイーグルス", "m": "千葉ロッテマリーンズ",
    "l": "埼玉西武ライオンズ", "b": "オリックス・バファローズ",
}
TEAM = {
    "巨人": "読売ジャイアンツ", "読売": "読売ジャイアンツ", "読売ジャイアンツ": "読売ジャイアンツ",
    "阪神": "阪神タイガース", "阪神タイガース": "阪神タイガース",
    "中日": "中日ドラゴンズ", "中日ドラゴンズ": "中日ドラゴンズ",
    "広島": "広島東洋カープ", "広島東洋カープ": "広島東洋カープ", "広島東洋": "広島東洋カープ",
    "ヤクルト": "東京ヤクルトスワローズ", "東京ヤクルト": "東京ヤクルトスワローズ", "東京ヤクルトスワローズ": "東京ヤクルトスワローズ",
    "DeNA": "横浜DeNAベイスターズ", "ＤｅＮＡ": "横浜DeNAベイスターズ", "横浜": "横浜DeNAベイスターズ", "横浜DeNA": "横浜DeNAベイスターズ", "横浜DeNAベイスターズ": "横浜DeNAベイスターズ",
    "ソフトバンク": "福岡ソフトバンクホークス", "福岡ソフトバンク": "福岡ソフトバンクホークス", "福岡ソフトバンクホークス": "福岡ソフトバンクホークス",
    "西武": "埼玉西武ライオンズ", "埼玉西武": "埼玉西武ライオンズ", "埼玉西武ライオンズ": "埼玉西武ライオンズ",
    "日本ハム": "北海道日本ハムファイターズ", "日ハム": "北海道日本ハムファイターズ", "北海道日本ハム": "北海道日本ハムファイターズ", "北海道日本ハムファイターズ": "北海道日本ハムファイターズ",
    "ロッテ": "千葉ロッテマリーンズ", "千葉ロッテ": "千葉ロッテマリーンズ", "千葉ロッテマリーンズ": "千葉ロッテマリーンズ",
    "楽天": "東北楽天ゴールデンイーグルス", "東北楽天": "東北楽天ゴールデンイーグルス", "東北楽天ゴールデンイーグルス": "東北楽天ゴールデンイーグルス",
    "オリックス": "オリックス・バファローズ", "オリックス・バファローズ": "オリックス・バファローズ",
}

def canon_team(value: object) -> str:
    s = "" if value is None else str(value).strip()
    return TEAM.get(s, s)

class _Rows(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr": self.row = []
        elif tag in {"td", "th"} and self.row is not None: self.cell = []
        elif tag == "br" and self.cell is not None: self.cell.append("\n")
    def handle_data(self, data: str) -> None:
        if self.cell is not None: self.cell.append(data)
    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.row is not None and self.cell is not None:
            text = re.sub(r"[ \t\r\f\v]+", " ", "".join(self.cell))
            text = "\n".join(x.strip() for x in text.splitlines() if x.strip())
            self.row.append(text.strip()); self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row: self.rows.append(self.row)
            self.row = None

def months_for_year(year: int) -> tuple[int, ...]:
    return tuple(range(6, 11)) if year == 2020 else tuple(range(3, 11))

def _request_rows(url: str, retries: int = 3) -> list[list[str]]:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Baseball-Prediction-System/4.0"})
            if r.status_code == 404: return []
            r.raise_for_status()
            parser = _Rows(); parser.feed(r.text)
            return parser.rows
        except Exception as exc:
            last = exc
            if attempt + 1 < retries: time.sleep(attempt + 1)
    raise RuntimeError(f"official NPB source failed: {url}: {last}")

def _fetch_month(year: int, month: int, retries: int = 3) -> list[dict]:
    url = DETAIL_URL.format(year=year, month=month)
    rows = _request_rows(url, retries=retries)
    date_re = re.compile(r"^(\d{1,2})/(\d{1,2})")
    score_re = re.compile(r"^(.+?)\s+(\d+)\s*-\s*(\d+)\s+(.+?)$")
    current_date = None; out = []
    for row in rows:
        if len(row) < 2: continue
        dm = date_re.search(row[0])
        if dm: current_date = f"{year:04d}-{int(dm.group(1)):02d}-{int(dm.group(2)):02d}"
        if current_date is None: continue
        for line in row[1].splitlines():
            m = score_re.match(line.strip())
            if not m: continue
            home, hs, aws, away = m.groups(); home, away = canon_team(home), canon_team(away)
            if home not in set(TEAM.values()) or away not in set(TEAM.values()): continue
            out.append({"date": current_date, "home": home, "away": away, "home_score": int(hs), "away_score": int(aws), "source_url": url})
    return out

def _fetch_team_month(year: int, month: int, code: str, team: str, cache: Path) -> list[dict]:
    url = TEAM_RESULT_URL.format(code=code, month=month)
    cache_path = cache / f"{year}-{month:02d}_{code}_team_v1.csv"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        try: return pd.read_csv(cache_path).to_dict("records")
        except Exception: cache_path.unlink(missing_ok=True)
    rows = _request_rows(url)
    date_re = re.compile(r"^(\d{1,2})(?:/(\d{1,2}))?$")
    score_re = re.compile(r"^(\d+)\s*-\s*(\d+)$")
    current_month = month
    out = []
    for row in rows:
        if len(row) < 7: continue
        d = row[0].strip()
        dm = date_re.match(d)
        if dm:
            day = int(dm.group(2) or dm.group(1)) if dm.group(2) else int(dm.group(1))
            if "/" in d: current_month = int(dm.group(1)); day = int(dm.group(2))
            current_date = f"{year:04d}-{current_month:02d}-{day:02d}"
        else:
            continue
        opponent = canon_team(row[1])
        score = row[6].strip()
        m = score_re.match(score)
        if not m or opponent not in set(TEAM.values()): continue
        if "中止" in " ".join(row): continue
        out.append({"date": current_date, "team": team, "opponent": opponent, "team_score": int(m.group(1)), "opponent_score": int(m.group(2)), "source_url": url})
    if out: pd.DataFrame(out).to_csv(cache_path, index=False)
    return out

def _team_result_fallback(years: Iterable[int], cache_dir: str | Path) -> pd.DataFrame:
    cache = Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)
    jobs = [(int(y), m, code, team) for y in sorted(set(int(y) for y in years)) for m in months_for_year(int(y)) for code, team in TEAM_RESULT_CODES.items()]
    rows = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        pending = {ex.submit(_fetch_team_month, y, m, code, team, cache): (y, m, code, team) for y, m, code, team in jobs}
        for fut in as_completed(pending):
            rows.extend(fut.result())
    if not rows: return pd.DataFrame(columns=["date","team","opponent","team_score","opponent_score","source_url"])
    return pd.DataFrame(rows).drop_duplicates(["date","team","opponent"], keep="last").reset_index(drop=True)

def official_schedule(years: Iterable[int], cache_dir: str | Path) -> pd.DataFrame:
    cache = Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)
    years = sorted(set(int(y) for y in years))
    jobs = [(y, m) for y in years for m in months_for_year(y)]
    rows = []
    with ThreadPoolExecutor(max_workers=min(12, max(1, len(jobs)))) as ex:
        pending = {}
        for year, month in jobs:
            path = cache / f"{year}-{month:02d}_official_v3.csv"
            if path.exists() and path.stat().st_size > 0:
                try: rows.extend(pd.read_csv(path).to_dict("records")); continue
                except Exception: path.unlink(missing_ok=True)
            pending[ex.submit(_fetch_month, year, month)] = (year, month, path)
        for fut in as_completed(pending):
            year, month, path = pending[fut]; got = fut.result()
            if got: pd.DataFrame(got).to_csv(path, index=False); rows.extend(got)
    if rows:
        schedule = pd.DataFrame(rows).drop_duplicates(["date","home","away"]).reset_index(drop=True)
    else:
        schedule = pd.DataFrame(columns=["date","home","away","home_score","away_score","source_url"])
    # NPB's schedule-detail endpoint can expose upcoming games without realized
    # scores depending on server/cache state. Fall back to the official team
    # result tables, which are realized-result pages and carry each team's score.
    fallback_years = []
    for year in years:
        if schedule.empty or not ((pd.to_datetime(schedule["date"], errors="coerce", utc=True).dt.year == year) & schedule["home_score"].notna() & schedule["away_score"].notna()).any():
            fallback_years.append(year)
    if fallback_years:
        team_rows = _team_result_fallback(fallback_years, cache)
        if not team_rows.empty:
            repaired = schedule.copy()
            if not repaired.empty:
                repaired["date_key"] = repaired["date"].astype(str); repaired["home_key"] = repaired["home"].map(canon_team); repaired["away_key"] = repaired["away"].map(canon_team)
            pairs = team_rows.copy(); pairs["date_key"] = pairs["date"].astype(str); pairs["team_key"] = pairs["team"].map(canon_team); pairs["opp_key"] = pairs["opponent"].map(canon_team)
            home = pairs.rename(columns={"team_key":"home_key","opp_key":"away_key","team_score":"home_score","opponent_score":"away_score"})[["date_key","home_key","away_key","home_score","away_score","source_url"]]
            if repaired.empty:
                repaired = home.rename(columns={"date_key":"date","home_key":"home","away_key":"away"})
            else:
                repaired = repaired.drop(columns=[c for c in ["home_score","away_score","source_url"] if c in repaired.columns]).merge(home, on=["date_key","home_key","away_key"], how="outer")
                repaired["date"] = repaired["date"].fillna(repaired["date_key"]); repaired["home"] = repaired["home"].fillna(repaired["home_key"]); repaired["away"] = repaired["away"].fillna(repaired["away_key"])
            schedule = repaired.drop(columns=[c for c in ["date_key","home_key","away_key"] if c in repaired.columns])
    return schedule.drop_duplicates(["date","home","away"]).reset_index(drop=True)

def _local_score_index(data_dir: str | Path, years: Iterable[int]) -> pd.DataFrame:
    root = Path(data_dir); frames = []
    for year in sorted(set(int(y) for y in years)):
        path = root / "npb" / f"npb_games_{year}_RAW_UNNORMALIZED.csv"
        if not path.exists() or path.stat().st_size == 0: continue
        raw = pd.read_csv(path, low_memory=False)
        needed = {"game_id","game_date","home_score","away_score","home_team_short_name","away_team_short_name"}
        if not needed.issubset(raw.columns): continue
        f = raw[["game_id","game_date","home_score","away_score","home_team_short_name","away_team_short_name"]].copy()
        f["game_id"] = f["game_id"].astype(str).str.strip()
        f["date_key"] = pd.to_datetime(f["game_date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
        f["home_key"] = f["home_team_short_name"].map(canon_team); f["away_key"] = f["away_team_short_name"].map(canon_team)
        f["home_score"] = pd.to_numeric(f["home_score"], errors="coerce"); f["away_score"] = pd.to_numeric(f["away_score"], errors="coerce")
        frames.append(f.dropna(subset=["date_key","home_score","away_score"])[["game_id","date_key","home_key","away_key","home_score","away_score"]])
    if not frames: return pd.DataFrame(columns=["game_id","date_key","home_key","away_key","home_score","away_score"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["game_id"], keep="last")

def repair_scores(frame: pd.DataFrame, data_dir: str | Path) -> pd.DataFrame:
    if frame.empty: return frame
    out = frame.copy(); out["home_score"] = pd.to_numeric(out["home_score"], errors="coerce"); out["away_score"] = pd.to_numeric(out["away_score"], errors="coerce")
    if not out[["home_score","away_score"]].isna().any(axis=1).any(): return out
    root = Path(data_dir); years = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.year.dropna().astype(int).unique(); local = _local_score_index(root, years)
    left = out.copy(); left["game_id_key"] = left["game_id"].astype(str).str.strip() if "game_id" in left.columns else ""; left["date_key"] = pd.to_datetime(left["date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d"); left["home_key"] = left["home"].map(canon_team); left["away_key"] = left["away"].map(canon_team)
    if not local.empty:
        by_id = local[["game_id","home_score","away_score"]].rename(columns={"home_score":"home_score_local","away_score":"away_score_local"})
        left = left.merge(by_id, left_on="game_id_key", right_on="game_id", how="left", suffixes=("","_idx")); left["home_score"] = left["home_score"].fillna(left["home_score_local"]); left["away_score"] = left["away_score"].fillna(left["away_score_local"])
        left = left.drop(columns=[c for c in ["game_id_idx","home_score_local","away_score_local"] if c in left.columns])
        if left[["home_score","away_score"]].isna().any(axis=1).any():
            by_key = local[["date_key","home_key","away_key","home_score","away_score"]].rename(columns={"home_score":"home_score_local","away_score":"away_score_local"})
            left = left.merge(by_key, on=["date_key","home_key","away_key"], how="left", suffixes=("","_keyidx"), validate="many_to_one"); left["home_score"] = left["home_score"].fillna(left["home_score_local"]); left["away_score"] = left["away_score"].fillna(left["away_score_local"])
            left = left.drop(columns=[c for c in ["home_score_local","away_score_local"] if c in left.columns])
    unresolved_mask = left[["home_score","away_score"]].isna().any(axis=1)
    if unresolved_mask.any():
        unresolved_years = pd.to_datetime(left.loc[unresolved_mask,"date"], errors="coerce", utc=True).dt.year.dropna().astype(int).unique(); schedule = official_schedule(unresolved_years, root / ".npb_official_schedule_cache")
        if schedule.empty: raise RuntimeError("NPB official result source returned no completed score rows for unresolved targets")
        schedule["date_key"] = schedule["date"].astype(str); schedule["home_key"] = schedule["home"].map(canon_team); schedule["away_key"] = schedule["away"].map(canon_team); schedule = schedule.drop_duplicates(["date_key","home_key","away_key"])
        repair = schedule[["date_key","home_key","away_key","home_score","away_score"]].rename(columns={"home_score":"home_score_official","away_score":"away_score_official"})
        left = left.merge(repair, on=["date_key","home_key","away_key"], how="left", suffixes=("","_official_idx"), validate="many_to_one"); left["home_score"] = left["home_score"].fillna(left["home_score_official"]); left["away_score"] = left["away_score"].fillna(left["away_score_official"])
        left = left.drop(columns=[c for c in ["home_score_official","away_score_official"] if c in left.columns])
    unresolved = int(left[["home_score","away_score"]].isna().any(axis=1).sum())
    if unresolved: raise RuntimeError(f"NPB score repair left {unresolved} unresolved games")
    return left.drop(columns=[c for c in ["game_id_key","date_key","home_key","away_key"] if c in left.columns])

def load_repaired_pbp(data_dir: str | Path) -> pd.DataFrame:
    from data.npb_pbp_adapter import normalize_pbp_frame
    root = Path(data_dir); files = sorted(root.glob("*_pbp.csv"))
    if not files: raise FileNotFoundError("No NPB PBP files found in data directory")
    frames = []
    for path in files:
        normalized = normalize_pbp_frame(pd.read_csv(path, low_memory=False), data_dir=None)
        if not normalized.empty: frames.append(normalized)
    if not frames: raise RuntimeError("NPB PBP files were found but none contained usable game rows")
    return repair_scores(pd.concat(frames, ignore_index=True, sort=False), root)
