"""Adapters for public NPB PBP releases used by the Baseball research runner.

The public release stores pre-game starter announcements in the textual
``description_jap`` field rather than in the pitch-level ``pitcher`` field.
The adapter therefore extracts starters only from explicit pre-game
announcements. It never uses winner/loser pitcher fields to infer starters,
and it never uses post-game information as a feature.

The upstream public release can expose home_total_runs/away_total_runs
columns that are structurally present but null. Completed targets are repaired
from NPB.jp's official schedule/result pages. Those official results are used
only as realized labels, never as predictive features.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
import re
from pathlib import Path
from typing import Iterable
import time

import numpy as np
import pandas as pd
import requests

_MISSING_TEXT = {"", "nan", "none", "nat"}
_STARTER_JP = "先発ピッチャー"
_STARTER_EN = "starting pitcher"
_NPB_OFFICIAL_MONTHS = tuple(range(3, 12))
_NPB_OFFICIAL_URL = "https://npb.jp/games/{year}/schedule_{month:02d}_detail.html"
_NPB_TEAM_ALIASES = {
    "巨人": "読売ジャイアンツ", "読売": "読売ジャイアンツ", "読売ジャイアンツ": "読売ジャイアンツ",
    "阪神": "阪神タイガース", "阪神タイガース": "阪神タイガース",
    "中日": "中日ドラゴンズ", "中日ドラゴンズ": "中日ドラゴンズ",
    "広島": "広島東洋カープ", "広島東洋カープ": "広島東洋カープ",
    "ヤクルト": "東京ヤクルトスワローズ", "東京ヤクルト": "東京ヤクルトスワローズ", "東京ヤクルトスワローズ": "東京ヤクルトスワローズ",
    "DeNA": "横浜DeNAベイスターズ", "ＤｅＮＡ": "横浜DeNAベイスターズ", "横浜": "横浜DeNAベイスターズ", "横浜DeNA": "横浜DeNAベイスターズ", "横浜DeNAベイスターズ": "横浜DeNAベイスターズ",
    "ソフトバンク": "福岡ソフトバンクホークス", "福岡ソフトバンク": "福岡ソフトバンクホークス", "福岡ソフトバンクホークス": "福岡ソフトバンクホークス",
    "西武": "埼玉西武ライオンズ", "埼玉西武": "埼玉西武ライオンズ", "埼玉西武ライオンズ": "埼玉西武ライオンズ",
    "日本ハム": "北海道日本ハムファイターズ", "日ハム": "北海道日本ハムファイターズ", "北海道日本ハム": "北海道日本ハムファイターズ", "北海道日本ハムファイターズ": "北海道日本ハムファイターズ",
    "ロッテ": "千葉ロッテマリーンズ", "千葉ロッテ": "千葉ロッテマリーンズ", "千葉ロッテマリーンズ": "千葉ロッテマリーンズ",
    "楽天": "東北楽天ゴールデンイーグルス", "東北楽天": "東北楽天ゴールデンイーグルス", "東北楽天ゴールデンイーグルス": "東北楽天ゴールデンイーグルス",
    "オリックス": "オリックス・バファローズ", "オリックス・バファローズ": "オリックス・バファローズ",
}


def _first_existing(df: pd.DataFrame, names: Iterable[str], default=None):
    for name in names:
        if name in df.columns:
            return df[name]
    if default is None:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.Series([default] * len(df), index=df.index)


def _clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in _MISSING_TEXT else text


def _canon_team(value: str) -> str:
    s = _clean_text(value)
    return _NPB_TEAM_ALIASES.get(s, s)


def _extract_en_starters(text: str) -> tuple[str, str]:
    text = _clean_text(text)
    if _STARTER_EN not in text.lower():
        return "", ""
    return "", ""


def _starter_from_descriptions(raw: pd.DataFrame) -> pd.DataFrame:
    desc_jp = _first_existing(raw, ["description_jap"])
    desc_en = _first_existing(raw, ["description"])
    home = _first_existing(raw, ["home_team_name", "H_NameS"]).map(_clean_text)
    away = _first_existing(raw, ["away_team_name", "V_NameS"]).map(_clean_text)
    gid_raw = _first_existing(raw, ["game_id", "GameID"])
    records = []
    for game_id, gidx in raw.groupby(gid_raw.astype(str), sort=False).groups.items():
        hp = ap = ""
        for idx in gidx:
            j = _clean_text(desc_jp.loc[idx])
            hteam = _clean_text(home.loc[idx])
            ateam = _clean_text(away.loc[idx])
            if _STARTER_JP in j:
                clause = re.sub(r"^[は:：\s]+", "", j.split(_STARTER_JP, 1)[1])
                assignments = []
                for part in [p.strip(" 、,\t") for p in re.split(r"[、,]", clause) if p.strip()]:
                    m = re.match(r"^(.+?)が(.+)$", part)
                    if m:
                        assignments.append((m.group(1).strip(), m.group(2).strip()))
                if len(assignments) >= 2:
                    for team, pitcher in assignments:
                        if _canon_team(team) == _canon_team(hteam):
                            hp = pitcher
                        elif _canon_team(team) == _canon_team(ateam):
                            ap = pitcher
                    if not hp and not ap:
                        hp, ap = assignments[0][1], assignments[1][1]
                    if hp and ap:
                        break
            eh, ea = _extract_en_starters(_clean_text(desc_en.loc[idx]))
            if eh and ea and not hp and not ap:
                hp, ap = eh, ea
        records.append((game_id, hp, ap))
    return pd.DataFrame(records, columns=["game_id", "home_pitcher", "away_pitcher"])


class _ScheduleParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append("\n")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.row is not None and self.cell is not None:
            text = re.sub(r"[ \t\r\f\v]+", " ", "".join(self.cell))
            text = "\n".join(x.strip() for x in text.split("\n") if x.strip())
            self.row.append(text.strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def _fetch_official_month(year: int, month: int) -> list[dict]:
    url = _NPB_OFFICIAL_URL.format(year=year, month=month)
    last = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Baseball-Prediction-System/1.0"})
            r.raise_for_status()
            enc = (r.apparent_encoding or r.encoding or "utf-8").lower().replace("-", "_")
            if "shift_jis" in enc or "cp932" in enc or "shiftjis" in enc:
                page_text = r.content.decode("cp932", errors="strict")
            else:
                page_text = r.content.decode(r.apparent_encoding or r.encoding or "utf-8", errors="strict")
            parser = _ScheduleParser()
            parser.feed(page_text)
            break
        except Exception as exc:
            last = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    else:
        raise RuntimeError(f"official NPB schedule failed: {url}: {last}")

    date_re = re.compile(r"^(\d{1,2})/(\d{1,2})")
    score_re = re.compile(r"^(.+?)\s+(\d+)\s*-\s*(\d+)\s+(.+?)$")
    time_re = re.compile(r"(\d{1,2}):(\d{2})")
    current_date = None
    rows = []
    for row in parser.rows:
        if len(row) < 2:
            continue
        dm = date_re.search(row[0])
        if dm:
            current_date = f"{year:04d}-{int(dm.group(1)):02d}-{int(dm.group(2)):02d}"
        if not current_date:
            continue
        times = time_re.findall(row[2] if len(row) > 2 else "")
        start_time = times[0] if times else ""
        for line in row[1].split("\n"):
            m = score_re.match(line.strip())
            if not m:
                continue
            home, hs, aws, away = m.groups()
            rows.append({"date": current_date, "home": _canon_team(home), "away": _canon_team(away), "home_score": int(hs), "away_score": int(aws), "start_time": start_time, "source_url": url})
    return rows


def _spaia_schedule_scores(year: int) -> pd.DataFrame:
    """Secondary realized-label source for NPB final scores, keyed by game_id.
    
    This endpoint is used only to repair historical labels when the public PBP
    score columns or the NPB HTML schedule parser are incomplete. It is never
    used for target-time predictive features.
    """
    url=f"https://spaia.jp/baseball/npb/api/schedules?Year={int(year)}"
    r=requests.get(url,timeout=20,headers={"User-Agent":"Baseball-Prediction-System/1.0"})
    r.raise_for_status()
    data=r.json()
    if not data:
        return pd.DataFrame(columns=["game_id","home_score","away_score"])
    df=pd.DataFrame(data)
    ren={"GameID":"game_id","HScore":"home_score","VScore":"away_score"}
    df=df.rename(columns=ren)
    for col in ("game_id","home_score","away_score"):
        if col not in df.columns:
            return pd.DataFrame(columns=["game_id","home_score","away_score"])
    df["game_id"]=df["game_id"].astype(str)
    df["home_score"]=pd.to_numeric(df["home_score"],errors="coerce")
    df["away_score"]=pd.to_numeric(df["away_score"],errors="coerce")
    return df[["game_id","home_score","away_score"]].dropna().drop_duplicates("game_id")

def _official_schedule(years: Iterable[int], cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    jobs = [(int(y), m) for y in sorted(set(years)) for m in _NPB_OFFICIAL_MONTHS]
    with ThreadPoolExecutor(max_workers=8) as ex:
        future_map = {}
        for year, month in jobs:
            path = cache_dir / f"{year}-{month:02d}_official.csv"
            if path.exists() and path.stat().st_size > 0:
                try:
                    rows.extend(pd.read_csv(path).to_dict("records"))
                    continue
                except Exception:
                    path.unlink(missing_ok=True)
            future_map[ex.submit(_fetch_official_month, year, month)] = (year, month, path)
        for future in as_completed(future_map):
            year, month, path = future_map[future]
            try:
                got = future.result()
                pd.DataFrame(got).to_csv(path, index=False)
                rows.extend(got)
            except Exception as exc:
                print(f"[NPB OFFICIAL] {year}-{month:02d} unavailable: {exc}")
    if not rows:
        return pd.DataFrame(columns=["date", "home", "away", "home_score", "away_score", "start_time", "source_url"])
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def _repair_scores_from_official(out: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    if out.empty:
        return out
    hs = pd.to_numeric(out["home_score"], errors="coerce")
    aws = pd.to_numeric(out["away_score"], errors="coerce")
    # Some public PBP releases expose structurally present score columns filled
    # with zeros rather than NaN. Treat a game whose entire observed score path
    # is 0-0 as missing and repair it from the official schedule/result page.
    tmp = out.copy()
    tmp["_score_sum"] = hs.fillna(0.0) + aws.fillna(0.0)
    zero_games = set(
        tmp.groupby("game_id", dropna=False)["_score_sum"].max()
        .loc[lambda s: s <= 0]
        .index.astype(str)
    )
    missing_any = hs.isna() | aws.isna()
    if not missing_any.any() and not zero_games:
        return out
    years = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.year.dropna().astype(int).unique().tolist()
    schedule = _official_schedule(years, data_dir / ".npb_official_schedule_cache")
    # If the NPB HTML schedule cannot be parsed on a runner, use the
    # repository's secondary schedule endpoint keyed by the exact game_id.
    # This is a realized-score repair only and therefore does not create PIT
    # leakage.
    spaia_frames=[]
    for year in years:
        try:
            spaia_frames.append(_spaia_schedule_scores(int(year)))
        except Exception as exc:
            print(f"[NPB SECONDARY SCORE] SPAIA schedule unavailable for {year}: {exc}")
    spaia=pd.concat(spaia_frames,ignore_index=True) if spaia_frames else pd.DataFrame(columns=["game_id","home_score","away_score"])
    if schedule.empty and spaia.empty:
        return out
    left = out.copy()
    left["date_key"] = pd.to_datetime(left["date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    left["home_key"] = left["home"].map(_canon_team)
    left["away_key"] = left["away"].map(_canon_team)
    schedule["date_key"] = schedule["date"].astype(str)
    schedule["home_key"] = schedule["home"].map(_canon_team)
    schedule["away_key"] = schedule["away"].map(_canon_team)
    schedule = schedule.drop_duplicates(["date_key", "home_key", "away_key"])
    if schedule.empty:
        merged = left.copy()
        merged["home_score_official"] = np.nan
        merged["away_score_official"] = np.nan
    else:
        merged = left.merge(schedule[["date_key", "home_key", "away_key", "home_score", "away_score"]], on=["date_key", "home_key", "away_key"], how="left", suffixes=("", "_official"))
    if not spaia.empty:
        merged=merged.merge(spaia.rename(columns={"home_score":"home_score_spaia","away_score":"away_score_spaia"}),on="game_id",how="left")
    else:
        merged["home_score_spaia"]=np.nan
        merged["away_score_spaia"]=np.nan
    official_h = pd.to_numeric(merged["home_score_official"], errors="coerce")
    official_a = pd.to_numeric(merged["away_score_official"], errors="coerce")
    spaia_h = pd.to_numeric(merged["home_score_spaia"], errors="coerce")
    spaia_a = pd.to_numeric(merged["away_score_spaia"], errors="coerce")
    current_h = pd.to_numeric(merged["home_score"], errors="coerce")
    current_a = pd.to_numeric(merged["away_score"], errors="coerce")
    needs_official = current_h.isna() | current_a.isna() | merged["game_id"].astype(str).isin(zero_games)
    # Official NPB final scores are the authoritative realized labels. When the
    # date/team key resolves, prefer them even if the upstream PBP exposes a
    # non-null but structurally unreliable score column.
    resolved_official = official_h.notna() & official_a.notna()
    chosen_h = official_h.fillna(spaia_h)
    chosen_a = official_a.fillna(spaia_a)
    resolved_any = chosen_h.notna() & chosen_a.notna()
    merged["home_score"] = current_h.where(~(needs_official | resolved_official), chosen_h).fillna(chosen_h)
    merged["away_score"] = current_a.where(~(needs_official | resolved_official), chosen_a).fillna(chosen_a)
    unresolved = int(merged[["home_score", "away_score"]].isna().any(axis=1).sum())
    if unresolved:
        print(f"[NPB OFFICIAL] unresolved score rows after official repair: {unresolved}")
    return merged.drop(columns=["date_key", "home_key", "away_key", "home_score_official", "away_score_official", "home_score_spaia", "away_score_spaia", "_score_sum"], errors="ignore")


def normalize_pbp_frame(raw: pd.DataFrame, *, data_dir: str | Path | None = None) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["game_id", "row_order", "date", "home", "away", "home_score", "away_score", "game_type", "home_pitcher", "away_pitcher"])
    out = pd.DataFrame(index=raw.index)
    out["game_id"] = _first_existing(raw, ["game_id", "GameID"]).astype(str)
    out["row_order"] = pd.to_numeric(_first_existing(raw, ["PlayInfo_SeqNo", "play_id", "ID", "page"], 0), errors="coerce").fillna(0)
    out["date"] = pd.to_datetime(_first_existing(raw, ["game_date", "GameDate"]), errors="coerce", utc=True)
    out["home"] = _first_existing(raw, ["home_team_name", "H_NameS"]).astype(str)
    out["away"] = _first_existing(raw, ["away_team_name", "V_NameS"]).astype(str)
    out["home_score"] = pd.to_numeric(_first_existing(raw, ["home_total_runs", "H_R"]), errors="coerce")
    out["away_score"] = pd.to_numeric(_first_existing(raw, ["away_total_runs", "V_R"]), errors="coerce")
    out["game_type"] = _first_existing(raw, ["game_type_name", "GameKindName"], "").astype(str)
    starter_df = _starter_from_descriptions(raw)
    out = out.merge(starter_df, on="game_id", how="left")
    out["home_pitcher"] = out["home_pitcher"].fillna("").astype(str)
    out["away_pitcher"] = out["away_pitcher"].fillna("").astype(str)
    out = out.dropna(subset=["game_id", "date"]).reset_index(drop=True)
    if data_dir is not None:
        out = _repair_scores_from_official(out, Path(data_dir))
    return out


def load_public_pbp(data_dir: str | Path) -> pd.DataFrame:
    root = Path(data_dir)
    files = sorted(root.glob("*_pbp.csv"))
    if not files:
        raise FileNotFoundError("No NPB PBP files found in data directory.")
    frames = []
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        normalized = normalize_pbp_frame(frame, data_dir=root)
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        raise RuntimeError("NPB PBP files were found but none contained usable game rows.")
    return pd.concat(frames, ignore_index=True, sort=False)
