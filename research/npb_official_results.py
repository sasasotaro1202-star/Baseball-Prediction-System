"""Fast, fail-closed NPB realized-score repair from official NPB result pages.

This module deliberately uses official NPB result pages only for realized labels.
It is isolated from feature generation so post-game information cannot leak into
pre-game features. It also avoids probing months that cannot exist in the regular
season (a major source of wasted CI time in the previous implementation).
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
        if tag in {"td", "th"} and self.row is not None and self.cell is not None:
            text = re.sub(r"[ \t\r\f\v]+", " ", "".join(self.cell))
            text = "\n".join(x.strip() for x in text.splitlines() if x.strip())
            self.row.append(text.strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None


def months_for_year(year: int) -> tuple[int, ...]:
    # 2020 regular season began in June and ended in October.
    # From 2021 onward the regular season runs March/April through October.
    return tuple(range(6, 11)) if year == 2020 else tuple(range(3, 11))


def _fetch_month(year: int, month: int, retries: int = 3) -> list[dict]:
    url = DETAIL_URL.format(year=year, month=month)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Baseball-Prediction-System/2.0"})
            if r.status_code == 404:
                # A nonexistent month is not a transient failure. Never retry it.
                return []
            r.raise_for_status()
            parser = _Rows()
            parser.feed(r.text)
            break
        except Exception as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(1.0 * (attempt + 1))
    else:
        raise RuntimeError(f"official NPB schedule failed: {url}: {last}")

    date_re = re.compile(r"^(\d{1,2})/(\d{1,2})")
    score_re = re.compile(r"^(.+?)\s+(\d+)\s*-\s*(\d+)\s+(.+?)$")
    current_date: str | None = None
    out: list[dict] = []
    for row in parser.rows:
        if len(row) < 2:
            continue
        dm = date_re.search(row[0])
        if dm:
            current_date = f"{year:04d}-{int(dm.group(1)):02d}-{int(dm.group(2)):02d}"
        if current_date is None:
            continue
        for line in row[1].splitlines():
            text = line.strip()
            m = score_re.match(text)
            if not m:
                continue
            home, hs, aws, away = m.groups()
            # Exclude accidental header-like matches and normalize official names.
            home, away = canon_team(home), canon_team(away)
            if home not in set(TEAM.values()) or away not in set(TEAM.values()):
                continue
            out.append({
                "date": current_date,
                "home": home,
                "away": away,
                "home_score": int(hs),
                "away_score": int(aws),
                "source_url": url,
            })
    return out


def official_schedule(years: Iterable[int], cache_dir: str | Path) -> pd.DataFrame:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(int(y), m) for y in sorted(set(years)) for m in months_for_year(int(y))]
    rows: list[dict] = []
    # Parallel network fetches make the first cold run much faster. Cached months
    # never hit the network, so subsequent runs are effectively local reads.
    with ThreadPoolExecutor(max_workers=min(12, max(1, len(jobs)))) as ex:
        pending = {}
        for year, month in jobs:
            path = cache / f"{year}-{month:02d}_official_v2.csv"
            if path.exists() and path.stat().st_size > 0:
                try:
                    rows.extend(pd.read_csv(path).to_dict("records"))
                    continue
                except Exception:
                    path.unlink(missing_ok=True)
            pending[ex.submit(_fetch_month, year, month)] = (year, month, path)
        for fut in as_completed(pending):
            year, month, path = pending[fut]
            got = fut.result()
            if got:
                pd.DataFrame(got).to_csv(path, index=False)
                rows.extend(got)
    if not rows:
        return pd.DataFrame(columns=["date", "home", "away", "home_score", "away_score", "source_url"])
    return pd.DataFrame(rows).drop_duplicates(["date", "home", "away"]).reset_index(drop=True)


def repair_scores(frame: pd.DataFrame, data_dir: str | Path) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["home_score"] = pd.to_numeric(out["home_score"], errors="coerce")
    out["away_score"] = pd.to_numeric(out["away_score"], errors="coerce")
    missing = out[["home_score", "away_score"]].isna().any(axis=1)
    if not missing.any():
        return out

    years = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.year.dropna().astype(int).unique()
    schedule = official_schedule(years, Path(data_dir) / ".npb_official_schedule_cache")
    if schedule.empty:
        raise RuntimeError("NPB official result source returned no completed score rows")

    left = out.copy()
    left["date_key"] = pd.to_datetime(left["date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    left["home_key"] = left["home"].map(canon_team)
    left["away_key"] = left["away"].map(canon_team)
    schedule["date_key"] = schedule["date"].astype(str)
    schedule["home_key"] = schedule["home"].map(canon_team)
    schedule["away_key"] = schedule["away"].map(canon_team)
    schedule = schedule.drop_duplicates(["date_key", "home_key", "away_key"])

    merged = left.merge(
        schedule[["date_key", "home_key", "away_key", "home_score", "away_score"]],
        on=["date_key", "home_key", "away_key"],
        how="left",
        suffixes=("", "_official"),
        validate="many_to_one",
    )
    merged["home_score"] = merged["home_score"].fillna(merged["home_score_official"])
    merged["away_score"] = merged["away_score"].fillna(merged["away_score_official"])
    unresolved = int(merged[["home_score", "away_score"]].isna().any(axis=1).sum())
    if unresolved:
        raise RuntimeError(f"NPB official score repair left {unresolved} unresolved games")
    return merged.drop(columns=["date_key", "home_key", "away_key", "home_score_official", "away_score_official"])


def load_repaired_pbp(data_dir: str | Path) -> pd.DataFrame:
    from data.npb_pbp_adapter import normalize_pbp_frame
    root = Path(data_dir)
    files = sorted(root.glob("*_pbp.csv"))
    if not files:
        raise FileNotFoundError("No NPB PBP files found in data directory")
    frames = []
    for path in files:
        raw = pd.read_csv(path, low_memory=False)
        normalized = normalize_pbp_frame(raw, data_dir=None)
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        raise RuntimeError("NPB PBP files were found but none contained usable game rows")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return repair_scores(combined, root)
