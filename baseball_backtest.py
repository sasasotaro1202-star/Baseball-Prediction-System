#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BASEBALL BACKTEST SYSTEM - NPB + MLB
====================================
Past-only chronological baseball prediction / backtest engine.

Data:
  NPB: local *_pbp.csv files produced from NPB PBP repositories.
  MLB: MLB Stats API (schedule + game feed) for historical games.

Design goals:
  - No target-game leakage.
  - Same-day games are ordered by game start time when available.
  - Expanding walk-forward evaluation.
  - Model comparison is genuinely out-of-sample.
  - Team form, home/away form, Elo, run environment, starter metrics,
    bullpen workload and park effects when available.
  - Classification: home/draw/away for NPB; home/away for MLB.
  - Score model: independent Poisson with tail bucket for 7+ internally.
  - Low/High internally for both leagues.
  - MLB prediction output requires BOTH starters to be confirmed.

This is intentionally self-contained so it can run in GitHub Actions.
It is not a claim of 100% accuracy: the objective is to maximize validated
out-of-sample performance and expose model error honestly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, VotingClassifier

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"

MLB_API = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 30

# Backtest controls
MIN_TRAIN = 100
RETRAIN_EVERY = int(os.getenv("NPB_RETRAIN_EVERY", "150"))
VALIDATION_RATIO = 0.20
MIN_VALIDATION = 45
MAX_FORM = 60
ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME = 25.0
ELO_REGRESSION = 0.20

# NPB game-type strings seen in common NPB PBP exports.
NPB_OFFICIAL_KEYWORDS = ("公式戦", "交流戦")
NPB_EXCLUDE_KEYWORDS = ("オープン戦", "オールスター", "ファーム", "二軍", "教育")

# Common NPB team aliases -> canonical names.
NPB_ALIASES = {
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


def norm_team(x: Any, league: str) -> str:
    s = str(x).strip()
    if league == "NPB":
        return NPB_ALIASES.get(s, s)
    return s


def num(x: Any, default=np.nan) -> float:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return default
        return float(x)
    except Exception:
        return default


def clip_prob(p: Sequence[float]) -> np.ndarray:
    a = np.asarray(p, dtype=float)
    a = np.nan_to_num(a, nan=1/len(a), posinf=1/len(a), neginf=1/len(a))
    a = np.maximum(a, 1e-9)
    return a / a.sum()


def parse_dt(x: Any) -> pd.Timestamp:
    return pd.to_datetime(x, errors="coerce", utc=True)


def poisson_pmf(k: int, lam: float) -> float:
    lam = max(float(lam), 1e-6)
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def poisson_grid(lam_h: float, lam_a: float, max_runs: int = 14) -> np.ndarray:
    """Joint score matrix. Tail beyond max_runs is deliberately preserved by a bucket."""
    ph = np.array([poisson_pmf(k, lam_h) for k in range(max_runs + 1)])
    pa = np.array([poisson_pmf(k, lam_a) for k in range(max_runs + 1)])
    m = np.outer(ph, pa)
    return m / m.sum()


def score_candidates(lam_h: float, lam_a: float, n: int = 4) -> List[Tuple[str, float]]:
    """Top score candidates with every 7+ outcome aggregated as その他."""
    lam_h = max(float(lam_h), 1e-6)
    lam_a = max(float(lam_a), 1e-6)
    cells = []
    p_low_h = sum(poisson_pmf(k, lam_h) for k in range(7))
    p_low_a = sum(poisson_pmf(k, lam_a) for k in range(7))
    low_mass = p_low_h * p_low_a
    for h in range(7):
        for a in range(7):
            cells.append((f"{h}-{a}", poisson_pmf(h, lam_h) * poisson_pmf(a, lam_a)))
    cells.sort(key=lambda z: z[1], reverse=True)
    tail = max(0.0, 1.0 - low_mass)
    # The display contract requires exactly four candidates. If その他 is
    # not naturally in the top four, it is still included and the weakest
    # ordinary candidate is removed.
    out = cells[:max(0, n-1)]
    out.append(("その他", tail))
    out.sort(key=lambda z: z[1], reverse=True)
    return out[:n]


def low_high_probs(lam_h: float, lam_a: float) -> Tuple[float, float]:
    # Low = both teams 0..6. High = complement.
    low = (sum(poisson_pmf(k, lam_h) for k in range(7)) *
           sum(poisson_pmf(k, lam_a) for k in range(7)))
    low = float(np.clip(low, 0, 1))
    return low, 1.0 - low


def result_from_score(h: float, a: float, league: str) -> int:
    if league == "NPB":
        return 0 if h > a else 1 if h == a else 2
    return 0 if h > a else 1  # MLB binary home win


@dataclass
class TeamState:
    results: deque = field(default_factory=lambda: deque(maxlen=MAX_FORM))
    gf: deque = field(default_factory=lambda: deque(maxlen=MAX_FORM))
    ga: deque = field(default_factory=lambda: deque(maxlen=MAX_FORM))
    home_results: deque = field(default_factory=lambda: deque(maxlen=MAX_FORM))
    away_results: deque = field(default_factory=lambda: deque(maxlen=MAX_FORM))
    home_gf: deque = field(default_factory=lambda: deque(maxlen=MAX_FORM))
    away_gf: deque = field(default_factory=lambda: deque(maxlen=MAX_FORM))
    home_ga: deque = field(default_factory=lambda: deque(maxlen=MAX_FORM))
    away_ga: deque = field(default_factory=lambda: deque(maxlen=MAX_FORM))
    total_matches: int = 0
    points: float = 0.0
    total_gf: float = 0.0
    total_ga: float = 0.0
    home_matches: int = 0
    away_matches: int = 0
    home_points: float = 0.0
    away_points: float = 0.0
    last_dt: Optional[pd.Timestamp] = None
    bullpen_ip_3: float = 0.0
    bullpen_ip_7: float = 0.0
    starter_history: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=12)))
    # Batting process history (game-level, shifted automatically because
    # match_features() is called before update_after_game()).
    pa: deque = field(default_factory=lambda: deque(maxlen=30))
    ab: deque = field(default_factory=lambda: deque(maxlen=30))
    h: deque = field(default_factory=lambda: deque(maxlen=30))
    hr: deque = field(default_factory=lambda: deque(maxlen=30))
    bb: deque = field(default_factory=lambda: deque(maxlen=30))
    so: deque = field(default_factory=lambda: deque(maxlen=30))
    doubles: deque = field(default_factory=lambda: deque(maxlen=30))
    triples: deque = field(default_factory=lambda: deque(maxlen=30))
    sb: deque = field(default_factory=lambda: deque(maxlen=30))
    cs: deque = field(default_factory=lambda: deque(maxlen=30))
    bullpen_er: deque = field(default_factory=lambda: deque(maxlen=20))
    bullpen_runs: deque = field(default_factory=lambda: deque(maxlen=20))
    bullpen_appearances: deque = field(default_factory=lambda: deque(maxlen=20))


class BaseballBacktest:
    def __init__(self, data_dir: Path = DATA):
        self.data_dir = Path(data_dir)
        self.states: Dict[Tuple[str, str], TeamState] = {}
        self.elo_ratings: Dict[Tuple[str, str], float] = {}
        self.results: List[Dict[str, Any]] = []
        self.model_scores: List[Dict[str, Any]] = []
        self.started_at = time.time()
        self.time_budget_sec = min(float(os.getenv("BASEBALL_TIME_BUDGET_SEC", "1500")), 1500.0)  # hard cap: 29:00
        self.audit: List[Dict[str, Any]] = []
        self.checkpoint_dir = RESULTS / "checkpoints"
        self.checkpoint_version = "npb-massive-resume-v4-100target"
        self._last_temperature = 1.0
        self.player_game = pd.DataFrame()
        self.player_history = defaultdict(list)
        self.player_index = {}
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # NPB loader
    # ------------------------------------------------------------------
    def load_npb_pbp(self) -> pd.DataFrame:
        """Load all available NPB seasons from the massive multi-season collector.

        Preferred input is data/npb_multi_source_games_all.csv. If it is absent,
        season partitions under data/npb_games/*.csv are combined. Legacy monthly
        PBP files remain a fallback, but are never combined with an overlapping
        multi-source season file.
        """
        aggregate = self.data_dir / "npb_multi_source_games_all.csv"
        season_files = sorted((self.data_dir / "npb_games").glob("*_multi_source_pbp.csv")) if (self.data_dir / "npb_games").exists() else []
        if aggregate.exists():
            files = [aggregate]
        elif season_files:
            files = season_files
        else:
            files = sorted(self.data_dir.glob("*_multi_source_pbp.csv"))
            files += sorted(self.data_dir.glob("*_pbp.csv"))
            files += sorted(ROOT.glob("*_pbp.csv"))
            files = [f for f in dict.fromkeys(files) if f.name not in {aggregate.name, "2026_multi_source_pbp.csv"}]
        if not files:
            raise FileNotFoundError("NPB data not found. Run npb_multi_source_massive_resumable.py first.")
        chunks=[]
        for f in files:
            if time.time()-self.started_at >= self.time_budget_sec:
                raise TimeoutError("time budget reached during NPB loading")
            try:
                df=pd.read_csv(f,low_memory=False)
                df.columns=[str(c).strip() for c in df.columns]
                if "game_id" not in df.columns:
                    continue
                chunks.append(df)
                print(f"[NPB LOAD] {f} rows={len(df)}")
            except Exception as e:
                self.audit.append({"type":"npb_load_error","file":str(f),"error":str(e)})
                print(f"[NPB SKIP] {f}: {e}")
        if not chunks: raise RuntimeError("No readable NPB data files.")
        raw=pd.concat(chunks,ignore_index=True,sort=False)
        out=self._normalize_npb_pbp(raw)
        if "game_id" in out:
            out=out.sort_values(["date","game_id","row_order"],na_position="last").drop_duplicates(["game_id"],keep="last").reset_index(drop=True)
        return out

    def load_npb_player_features(self) -> pd.DataFrame:
        """Load player-game micro-features produced by the collector.

        These are consumed strictly chronologically: target-game player stats are
        added only after that game is scored, so player-level features cannot leak.
        """
        f = self.data_dir / "npb_player_game_features_all.csv"
        if not f.exists():
            print("[PLAYER LOAD] no player-level corpus yet; using team-only features")
            return pd.DataFrame()
        try:
            d = pd.read_csv(f, low_memory=False)
            if 'game_id' not in d or 'player_id' not in d:
                return pd.DataFrame()
            d['game_id'] = d['game_id'].astype(str)
            d['player_id'] = d['player_id'].astype(str)
            if 'side' not in d: d['side'] = ''
            print(f"[PLAYER LOAD] rows={len(d)} players={d.player_id.nunique()} games={d.game_id.nunique()}")
            return d
        except Exception as e:
            self.audit.append({'type':'player_load_error','error':str(e)})
            print(f"[PLAYER LOAD ERROR] {e}")
            return pd.DataFrame()

    def aggregate_npb_games(self, pbp: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for gid, g in pbp.groupby("game_id", sort=False):
            g = g.sort_values("row_order")
            home, away = g["home"].iloc[0], g["away"].iloc[0]
            dt = g["date"].iloc[0]
            # Prefer explicit final score columns, otherwise reconstruct from last valid value.
            hs = g["home_score"].dropna()
            aas = g["away_score"].dropna()
            if len(hs) and len(aas):
                hscore, ascore = float(hs.iloc[-1]), float(aas.iloc[-1])
            else:
                hscore, ascore = self._reconstruct_npb_score(g)
            if np.isnan(hscore) or np.isnan(ascore):
                continue
            # Official-game filter: exclude exhibition/farm/all-star.
            gt = " ".join(g["game_type"].dropna().astype(str).tolist())
            if any(k in gt for k in NPB_EXCLUDE_KEYWORDS):
                continue
            # If game type exists and contains neither official keyword nor is empty, keep only likely official.
            if gt and not any(k in gt for k in NPB_OFFICIAL_KEYWORDS):
                # Known PBP exports sometimes omit a clean type label; don't reject unless it is clearly non-official.
                if any(k in gt.lower() for k in ("open", "spring", "farm", "allstar")):
                    continue
            hp = self._first_pitcher(g, "home")
            ap = self._first_pitcher(g, "away")
            rows.append({
                "league": "NPB", "game_id": str(gid), "datetime": dt,
                "home": home, "away": away, "home_score": hscore, "away_score": ascore,
                "home_starter": hp, "away_starter": ap,
                "venue": "unknown", "confirmed_starters": bool(hp and ap),
            })
        out = pd.DataFrame(rows)
        if out.empty:
            raise RuntimeError("No NPB games could be reconstructed.")
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        return out.sort_values(["datetime", "game_id"]).drop_duplicates("game_id").reset_index(drop=True)

    def _first_pitcher(self, g: pd.DataFrame, side: str) -> str:
        col = f"{side}_pitcher"
        if col in g:
            vals = [str(x).strip() for x in g[col].tolist() if str(x).strip() not in ("", "nan", "None")]
            if vals:
                return vals[0]
        # Fallback: common event text patterns are intentionally conservative.
        return ""

    def _reconstruct_npb_score(self, g: pd.DataFrame) -> Tuple[float, float]:
        # Supports PBP files where each half-inning has addedRuns / run fields.
        cols = {str(c).lower(): c for c in g.columns}
        added = cols.get("added_runs")
        if added is None:
            for k in ("addedruns", "added_runs", "runs_scored", "run", "runs"):
                if k in cols:
                    added = cols[k]
                    break
        if added is not None:
            h, a = 0.0, 0.0
            for _, r in g.iterrows():
                x = num(r[added], 0.0)
                if not np.isfinite(x):
                    continue
                half = str(r.get("half", "")).lower()
                inning = str(r.get("inning", "")).lower()
                text = half + " " + inning
                if any(t in text for t in ("top", "表", "visitor", "away", "先攻")):
                    a += max(0, x)
                elif any(t in text for t in ("bottom", "裏", "home", "後攻")):
                    h += max(0, x)
            return h, a
        # Last-resort event parser. Only recognizes explicit run totals, not arbitrary text.
        h = a = 0.0
        for _, r in g.iterrows():
            ev = str(r.get("event", ""))
            m = re.search(r"(?:runs?|得点)[=: ]+(\d+)", ev, flags=re.I)
            if not m:
                continue
            x = float(m.group(1))
            half = str(r.get("half", "")).lower()
            if "top" in half or "表" in half:
                a = max(a, x)
            elif "bottom" in half or "裏" in half:
                h = max(h, x)
        return h, a

    # ------------------------------------------------------------------
    # MLB loader
    # ------------------------------------------------------------------
    def load_mlb(self, start_year: int = 2020, end_year: int = 2026) -> pd.DataFrame:
        cache = self.data_dir / "mlb_games.csv"
        if cache.exists():
            try:
                df = pd.read_csv(cache)
                if len(df) > 100:
                    print(f"[MLB] using cache: {cache} ({len(df)})")
                    return self._normalize_mlb_games(df)
            except Exception:
                pass
        rows = []
        for year in range(start_year, end_year + 1):
            url = f"{MLB_API}/schedule"
            params = {"sportId": 1, "startDate": f"{year}-03-01", "endDate": f"{year}-11-30", "hydrate": "probablePitcher,linescore"}
            print(f"[MLB] downloading schedule {year}")
            data = self._get_json(url, params=params)
            for date_block in data.get("dates", []):
                for game in date_block.get("games", []):
                    status = game.get("status", {}).get("abstractGameState")
                    if status != "Final":
                        continue
                    teams = game.get("teams", {})
                    home = teams.get("home", {})
                    away = teams.get("away", {})
                    hp = (home.get("probablePitcher") or {}).get("fullName", "")
                    ap = (away.get("probablePitcher") or {}).get("fullName", "")
                    # Historical schedule probablePitcher can be missing/wrong; feed will be used below.
                    rows.append({
                        "league": "MLB", "game_id": str(game.get("gamePk")),
                        "datetime": game.get("gameDate"),
                        "home": home.get("team", {}).get("name", ""),
                        "away": away.get("team", {}).get("name", ""),
                        "home_score": home.get("score", np.nan),
                        "away_score": away.get("score", np.nan),
                        "home_starter": hp, "away_starter": ap,
                        "venue": (game.get("venue") or {}).get("name", ""),
                        "confirmed_starters": bool(hp and ap),
                    })
            time.sleep(0.1)
        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("MLB Stats API returned no completed games.")
        df = self._normalize_mlb_games(df)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        return df

    def enrich_mlb_starters(self, games: pd.DataFrame) -> pd.DataFrame:
        """Use game feed to identify actual starting pitchers for completed games."""
        games = games.copy()
        for i in range(len(games)):
            gid = games.iloc[i]["game_id"]
            if not gid or str(gid) == "nan":
                continue
            try:
                feed = self._get_json(f"{MLB_API}/game/{gid}/feed/live")
                live = feed.get("liveData", {})
                box = live.get("boxscore", {}).get("teams", {})
                hp = box.get("home", {}).get("players", {})
                ap = box.get("away", {}).get("players", {})
                def find_starter(players):
                    for p in players.values():
                        pit = p.get("stats", {}).get("pitching", {})
                        if pit.get("gamesStarted", 0) == 1:
                            return p.get("person", {}).get("fullName", "")
                    # More reliable for game feed: first pitcher listed in pitchers array.
                    arr = []
                    for p in players.values():
                        pid = p.get("person", {}).get("id")
                        if pid and p.get("stats", {}).get("pitching"):
                            arr.append((p.get("gameStatus", {}).get("isStartingPitcher", False), p.get("person", {}).get("fullName", "")))
                    for flag, name in arr:
                        if flag and name:
                            return name
                    return ""
                hname, aname = find_starter(hp), find_starter(ap)
                if hname: games.at[i, "home_starter"] = hname
                if aname: games.at[i, "away_starter"] = aname
                games.at[i, "confirmed_starters"] = bool(hname and aname)
            except Exception as e:
                print(f"[MLB feed skip] {gid}: {e}")
        return games

    def _normalize_mlb_games(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in ["home_score", "away_score"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce", utc=True)
        out = out.dropna(subset=["datetime", "home_score", "away_score", "home", "away"])
        out["league"] = "MLB"
        out["game_id"] = out["game_id"].astype(str)
        return out.sort_values(["datetime", "game_id"]).drop_duplicates("game_id").reset_index(drop=True)

    def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # State and features
    # ------------------------------------------------------------------
    def state(self, league: str, team: str) -> TeamState:
        key = (league, team)
        if key not in self.states:
            self.states[key] = TeamState()
        return self.states[key]

    def elo(self, league: str, team: str) -> float:
        key = (league, team)
        return self.elo_ratings.get(key, ELO_START)

    def _team_features(self, league: str, team: str, venue: str, dt: pd.Timestamp) -> Dict[str, float]:
        s = self.state(league, team)
        f: Dict[str, float] = {}
        for w in (3, 5, 10, 20, 30, 45, 60):
            r = list(s.results)[-w:]
            gf = list(s.gf)[-w:]
            ga = list(s.ga)[-w:]
            pts = [3 if x == 0 else 1 if x == 1 else 0 for x in r]
            f[f"pts_{w}"] = float(np.mean(pts)) if pts else 1.0
            f[f"gf_{w}"] = float(np.mean(gf)) if gf else 0.0
            f[f"ga_{w}"] = float(np.mean(ga)) if ga else 0.0
            f[f"gd_{w}"] = f[f"gf_{w}"] - f[f"ga_{w}"]
            f[f"win_{w}"] = float(np.mean(np.asarray(r) == 0)) if r else 0.33
            f[f"draw_{w}"] = float(np.mean(np.asarray(r) == 1)) if r else 0.33
        if venue == "home":
            rr, gf, ga = list(s.home_results), list(s.home_gf), list(s.home_ga)
            vm, vp = s.home_matches, s.home_points
        else:
            rr, gf, ga = list(s.away_results), list(s.away_gf), list(s.away_ga)
            vm, vp = s.away_matches, s.away_points
        f["venue_n"] = float(vm)
        f["venue_pts"] = float(vp / vm) if vm else 1.0
        f["venue_gf"] = float(np.mean(gf[-10:])) if gf else 0.0
        f["venue_ga"] = float(np.mean(ga[-10:])) if ga else 0.0
        f["elo"] = self.elo(league, team)
        f["rest_days"] = float(max(0.0, (dt - s.last_dt).total_seconds() / 86400.0)) if s.last_dt is not None else 30.0
        f["matches"] = float(s.total_matches)
        # Bullpen workload is updated from completed games only; never from target game.
        f["bp3"] = float(s.bullpen_ip_3)
        f["bp7"] = float(s.bullpen_ip_7)
        # Rich rolling batting/process features. These are strictly prior-game
        # state because update_after_game() runs after match_features().
        for w in (3, 5, 10, 20, 30):
            def meanq(q, default=0.0):
                z=list(q)[-w:]
                return float(np.mean(z)) if z else default
            ab=meanq(s.ab, 0.0); h=meanq(s.h, 0.0); hr=meanq(s.hr, 0.0); bb=meanq(s.bb, 0.0); so=meanq(s.so, 0.0)
            doubles=meanq(s.doubles,0.0); triples=meanq(s.triples,0.0)
            pa=meanq(s.pa,0.0)
            f[f"bat_ab_{w}"]=ab; f[f"bat_avg_{w}"]=h/max(ab,1.0)
            f[f"bat_hr_{w}"]=hr; f[f"bat_bb_{w}"]=bb; f[f"bat_so_{w}"]=so
            f[f"bat_bb_rate_{w}"]=bb/max(pa,1.0); f[f"bat_so_rate_{w}"]=so/max(pa,1.0)
            f[f"bat_xbh_{w}"]=doubles+triples+hr
            f[f"bat_hr_rate_{w}"]=hr/max(ab,1.0)
            f[f"bat_iso_proxy_{w}"]=max(0.0, (doubles+2*triples+3*hr)/max(ab,1.0) - f[f"bat_avg_{w}"])
            f[f"bat_extra_base_rate_{w}"]=max(0.0, (doubles+2*triples+3*hr)/max(ab,1.0))
        f["bp_er_10"] = float(np.mean(list(s.bullpen_er)[-10:])) if s.bullpen_er else 0.0
        f["bp_runs_10"] = float(np.mean(list(s.bullpen_runs)[-10:])) if s.bullpen_runs else 0.0
        f["bp_app_10"] = float(np.sum(list(s.bullpen_appearances)[-10:])) if s.bullpen_appearances else 0.0
        # Volatility/trend signals: useful for separating stable teams from
        # high-variance teams without peeking at the target game.
        for name,q in (("gf",s.gf),("ga",s.ga),("hr",s.hr),("so",s.so),("bb",s.bb)):
            vals=np.asarray(list(q)[-20:],dtype=float)
            f[f"{name}_sd_20"] = float(np.std(vals)) if len(vals) >= 2 else 0.0
            if len(vals) >= 4:
                x=np.arange(len(vals),dtype=float)
                try: f[f"{name}_slope_20"] = float(np.polyfit(x,vals,1)[0])
                except Exception: f[f"{name}_slope_20"] = 0.0
            else: f[f"{name}_slope_20"] = 0.0
        return f

    def _lineup_ids(self, row: pd.Series, side: str) -> List[Dict[str, Any]]:
        raw=row.get(f'{side}_lineup_json','')
        if raw in (None,'',float('nan')): return []
        try:
            x=json.loads(raw) if isinstance(raw,str) else raw
            return x if isinstance(x,list) else []
        except Exception:
            return []

    def _player_profile(self, league: str, player_id: str, dt: pd.Timestamp) -> Dict[str,float]:
        hist=self.player_history.get((league,str(player_id)),[])
        if not hist:
            return {'pa':0.0,'avg':0.25,'obp':0.32,'slg':0.38,'iso':0.13,'bb_rate':0.08,
                    'so_rate':0.20,'sb_rate':0.02,'bunt_rate':0.02,'power_rate':0.05,'contact_rate':0.70,
                    'gidp_rate':0.03,'groundout_rate':0.20,'flyout_rate':0.20,'lineout_rate':0.08,
                    'errors_rate':0.0,'games':0.0,'recent_pa':0.0,'recent_iso':0.13,'recent_so_rate':0.20,
                    'recent_bb_rate':0.08,'recent_sb_rate':0.02}
        df=pd.DataFrame(hist).copy()
        for c in ('pa','ab','h','2b','3b','hr','bb','so','sb','cs','bunt','power_events','contact_events','gidp','groundout','flyout','lineout','errors'):
            if c not in df: df[c]=0.0
            df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0.0)
        df['avg']=df.h/df.ab.clip(lower=1); df['obp']=(df.h+df.bb)/((df.ab+df.bb).clip(lower=1))
        df['slg']=(df.h+df['2b']+2*df['3b']+3*df.hr)/df.ab.clip(lower=1); df['iso']=(df.slg-df.avg).clip(lower=0)
        df['bb_rate']=df.bb/df.pa.clip(lower=1); df['so_rate']=df.so/df.pa.clip(lower=1); df['sb_rate']=df.sb/df.pa.clip(lower=1)
        df['bunt_rate']=df.bunt/df.pa.clip(lower=1); df['power_rate']=df.power_events/df.pa.clip(lower=1); df['contact_rate']=df.contact_events/df.pa.clip(lower=1)
        df['gidp_rate']=df.gidp/df.pa.clip(lower=1); df['groundout_rate']=df.groundout/df.pa.clip(lower=1); df['flyout_rate']=df.flyout/df.pa.clip(lower=1); df['lineout_rate']=df.lineout/df.pa.clip(lower=1)
        df['errors_rate']=df.errors/df.pa.clip(lower=1)
        age=np.arange(len(df),dtype=float); w=np.exp(-(len(df)-1-age)/12.0)
        def wa(c,default):
            v=pd.to_numeric(df[c],errors='coerce'); m=v.notna()
            return float(np.average(v[m],weights=w[m])) if m.any() else default
        recent=df.tail(5)
        def rm(c,default):
            v=pd.to_numeric(recent[c],errors='coerce').dropna(); return float(v.mean()) if len(v) else default
        return {'pa':wa('pa',0.0),'avg':wa('avg',0.25),'obp':wa('obp',0.32),'slg':wa('slg',0.38),'iso':wa('iso',0.13),
                'bb_rate':wa('bb_rate',0.08),'so_rate':wa('so_rate',0.20),'sb_rate':wa('sb_rate',0.02),'bunt_rate':wa('bunt_rate',0.02),
                'power_rate':wa('power_rate',0.05),'contact_rate':wa('contact_rate',0.70),'gidp_rate':wa('gidp_rate',0.03),
                'groundout_rate':wa('groundout_rate',0.20),'flyout_rate':wa('flyout_rate',0.20),'lineout_rate':wa('lineout_rate',0.08),
                'errors_rate':wa('errors_rate',0.0),'games':float(len(df)),'recent_pa':rm('pa',0.0),'recent_iso':rm('iso',0.13),
                'recent_so_rate':rm('so_rate',0.20),'recent_bb_rate':rm('bb_rate',0.08),'recent_sb_rate':rm('sb_rate',0.02)}

    def _lineup_features(self, row: pd.Series, side: str, league: str) -> Dict[str,float]:
        lineup=self._lineup_ids(row,side)
        prof=[]; orders=[]
        for item in lineup:
            pid=str(item.get('player_id','')).strip()
            if not pid: continue
            p=self._player_profile(league,pid,pd.Timestamp(row['datetime']))
            prof.append(p); orders.append(num(item.get('batting_order')))
        out={}
        metrics=('avg','obp','slg','iso','bb_rate','so_rate','sb_rate','bunt_rate','power_rate','contact_rate','gidp_rate','groundout_rate','flyout_rate','lineout_rate','errors_rate','recent_iso','recent_so_rate','recent_bb_rate','recent_sb_rate')
        # PA-weighted lineup skill plus order-weighted top-of-lineup emphasis.
        weights=np.asarray([max(1.0,p['pa']) for p in prof],dtype=float) if prof else np.array([])
        if len(weights): weights=weights/weights.sum()
        for m in metrics:
            vals=np.asarray([p[m] for p in prof],dtype=float) if prof else np.array([])
            out[f'lineup_{m}']=float(np.average(vals,weights=weights)) if len(vals) else {'avg':0.25,'obp':0.32,'slg':0.38,'iso':0.13,'bb_rate':0.08,'so_rate':0.20,'sb_rate':0.02,'bunt_rate':0.02,'power_rate':0.05,'contact_rate':0.70,'gidp_rate':0.03,'groundout_rate':0.20,'flyout_rate':0.20,'lineout_rate':0.08,'errors_rate':0.0,'recent_iso':0.13,'recent_so_rate':0.20,'recent_bb_rate':0.08,'recent_sb_rate':0.02}[m]
        out['lineup_n']=float(len(prof)); out['lineup_known_n']=float(sum(p['games']>0 for p in prof)); out['lineup_history_coverage']=out['lineup_known_n']/max(out['lineup_n'],1.0)
        # Sum of complementary skills across the whole lineup.
        out['lineup_power_sum']=float(sum(p['power_rate'] for p in prof)); out['lineup_speed_sum']=float(sum(p['sb_rate'] for p in prof)); out['lineup_bunt_sum']=float(sum(p['bunt_rate'] for p in prof))
        out['lineup_contact_sum']=float(sum(p['contact_rate'] for p in prof)); out['lineup_gdp_sum']=float(sum(p['gidp_rate'] for p in prof))
        return out

    def _update_player_history(self, row: pd.Series):
        if self.player_game.empty: return
        gid=str(row.get('game_id',''))
        sub=self.player_game[self.player_game.game_id.astype(str)==gid]
        for _,r in sub.iterrows():
            pid=str(r.get('player_id','')); side=str(r.get('side',''))
            if pid: self.player_history[(row['league'],pid)].append(r.to_dict())

    def match_features(self, row: pd.Series) -> Dict[str, float]:
        league = row["league"]
        dt = pd.Timestamp(row["datetime"])
        h, a = norm_team(row["home"], league), norm_team(row["away"], league)
        hf = self._team_features(league, h, "home", dt)
        af = self._team_features(league, a, "away", dt)
        out: Dict[str, float] = {"home_adv": 1.0}
        for k, v in hf.items(): out[f"h_{k}"] = v
        for k, v in af.items(): out[f"a_{k}"] = v
        for k in set(hf) & set(af): out[f"d_{k}"] = hf[k] - af[k]
        # Starter pregame information comes only from historical starter profiles.
        hs = str(row.get("home_starter", "") or "")
        ass = str(row.get("away_starter", "") or "")
        out.update(self.starter_features(league, hs, dt, prefix="hs_"))
        out.update(self.starter_features(league, ass, dt, prefix="as_"))
        # Player-by-player lineup micro-features. These are built from each
        # hitter's own prior games and are therefore available before the target.
        hpf=self._lineup_features(row,"home",league); apf=self._lineup_features(row,"away",league)
        for k,v in hpf.items(): out[f"h_{k}"]=v
        for k,v in apf.items(): out[f"a_{k}"]=v
        for k in set(hpf)&set(apf): out[f"d_{k}"]=hpf[k]-apf[k]
        # IMPORTANT: never use the target game's own pitching line as a
        # pregame feature. The data-prep layer may carry *_starter_* columns
        # for audit, but those values are consumed only AFTER this feature row
        # is created by _update_pitcher_history().
        # Weather is a feature only when a historical observation was available.
        # Missing weather stays neutral rather than being imputed from future data.
        for c in ("weather_temp_c", "weather_humidity_pct", "weather_wind_kmh", "weather_precip_mm"):
            if c in row:
                try:
                    out[c] = float(row.get(c)) if pd.notna(row.get(c)) else 0.0
                except Exception:
                    out[c] = 0.0
        # Market-neutral run environment from historical team scoring.
        out["expected_env"] = max(0.5, min(12.0, 0.5 * (hf["gf_10"] + af["gf_10"] + hf["ga_10"] + af["ga_10"])))
        # Nonlinear matchup signals: offense vs opposing starter skill, bullpen
        # fatigue asymmetry, and weather/run-environment interactions.
        out["matchup_home_bat_vs_away_fip"] = hf.get("bat_avg_10",0.0) - out.get("as_fip",4.0)/20.0
        out["matchup_away_bat_vs_home_fip"] = af.get("bat_avg_10",0.0) - out.get("hs_fip",4.0)/20.0
        out["bullpen_fatigue_diff"] = hf.get("bp_app_10",0.0) - af.get("bp_app_10",0.0)
        out["weather_run_signal"] = (out.get("weather_temp_c",0.0)-20.0)/10.0 + out.get("weather_wind_kmh",0.0)/30.0 - out.get("weather_precip_mm",0.0)/5.0
        out["starter_x_quality_proxy"] = (out.get("hs_k9",7.5)-out.get("hs_bb9",3.0)-out.get("hs_hr9",1.0)) - (out.get("as_k9",7.5)-out.get("as_bb9",3.0)-out.get("as_hr9",1.0))
        out["starter_recency_gap"] = out.get("hs_recent_era",4.0)-out.get("as_recent_era",4.0)
        out["starter_experience_gap"] = out.get("hs_starts",0.0)-out.get("as_starts",0.0)
        out["starter_kbb_gap"] = (out.get("hs_k9",7.5)-out.get("hs_bb9",3.0)) - (out.get("as_k9",7.5)-out.get("as_bb9",3.0))
        out["starter_hr_gap"] = out.get("as_hr9",1.0)-out.get("hs_hr9",1.0)
        out["starter_recent_form_gap"] = out.get("as_recent_era",4.0)-out.get("hs_recent_era",4.0)
        out["offense_power_gap_10"] = hf.get("bat_hr_rate_10",0.0)-af.get("bat_hr_rate_10",0.0)
        out["offense_walk_gap_10"] = hf.get("bat_bb_rate_10",0.0)-af.get("bat_bb_rate_10",0.0)
        out["offense_contact_gap_10"] = (hf.get("bat_avg_10",0.0)-hf.get("bat_so_rate_10",0.0))-(af.get("bat_avg_10",0.0)-af.get("bat_so_rate_10",0.0))
        out["run_volatility_gap_20"] = hf.get("gf_sd_20",0.0)-af.get("gf_sd_20",0.0)
        out["run_trend_gap_20"] = hf.get("gf_slope_20",0.0)-af.get("gf_slope_20",0.0)
        out["starter_known"] = float(bool(hs and ass))
        return out

    def starter_features(self, league: str, pitcher: str, dt: pd.Timestamp, prefix: str) -> Dict[str, float]:
        # Historical pitcher metrics stored in TeamState-like global dictionaries.
        hist = self.pitcher_history.get((league, pitcher), []) if pitcher else []
        if not hist:
            # Conservative league-neutral priors. Unknown pitchers are not
            # treated as elite or terrible merely because history is missing.
            return {
                prefix + "era": 4.00, prefix + "whip": 1.30,
                prefix + "k9": 7.5, prefix + "bb9": 3.0,
                prefix + "hr9": 1.0, prefix + "fip": 4.00,
                prefix + "starts": 0.0, prefix + "recent_era": 4.00,
                prefix + "recent_k9": 7.5, prefix + "recent_k_rate": 0.20, prefix + "recent_bb_rate": 0.08,
                prefix + "recent_pitches": 0.0, prefix + "recent_ip": 0.0, prefix + "era_slope": 0.0,
                prefix + "k9_slope": 0.0, prefix + "fip_slope": 0.0, prefix + "recent_era_sd": 0.0,
                prefix + "recent_k9_sd": 0.0, prefix + "recent_pitches_sd": 0.0,
            }
        df = pd.DataFrame(hist).copy()
        df["_age_idx"] = np.arange(len(df), dtype=float)
        # Exponentially decay old starts instead of giving a 2012 start the same
        # weight as yesterday's start. The large historical corpus remains useful
        # while recent form dominates.
        decay = np.exp(-(len(df)-1-df["_age_idx"])/18.0)
        def wavg(col, default=0.0):
            if col not in df: return default
            v=pd.to_numeric(df[col],errors="coerce")
            mask=v.notna()
            if not mask.any(): return default
            return float(np.average(v[mask],weights=decay[mask]))
        recent=df.tail(5)
        def ravg(col, default=0.0):
            if col not in recent: return default
            v=pd.to_numeric(recent[col],errors="coerce").dropna()
            return float(v.mean()) if len(v) else default
        def slope(col, default=0.0):
            if col not in df or len(df) < 3: return default
            v=pd.to_numeric(df[col],errors="coerce").dropna().tail(8)
            if len(v) < 3: return default
            x=np.arange(len(v),dtype=float)
            try: return float(np.polyfit(x,v.values,1)[0])
            except Exception: return default
        def std(col, default=0.0):
            if col not in recent: return default
            v=pd.to_numeric(recent[col],errors="coerce").dropna()
            return float(v.std(ddof=0)) if len(v) else default
        return {
            prefix + "era": wavg("era",4.00), prefix + "whip": wavg("whip",1.30), prefix + "k9": wavg("k9",7.5),
            prefix + "bb9": wavg("bb9",3.0), prefix + "hr9": wavg("hr9",1.0), prefix + "fip": wavg("fip",4.00),
            prefix + "starts": float(len(df)), prefix + "recent_era": ravg("era",4.00),
            prefix + "recent_k9": ravg("k9",7.5), prefix + "recent_k_rate": ravg("k_rate",0.20),
            prefix + "recent_bb_rate": ravg("bb_rate",0.08), prefix + "recent_pitches": ravg("pitches",0.0), prefix + "recent_ip": ravg("ip",0.0),
            prefix + "era_slope": slope("era",0.0), prefix + "k9_slope": slope("k9",0.0),
            prefix + "fip_slope": slope("fip",0.0), prefix + "recent_era_sd": std("era",0.0),
            prefix + "recent_k9_sd": std("k9",0.0), prefix + "recent_pitches_sd": std("pitches",0.0),
        }

    def build_features(self, games: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        self.states.clear(); self.elo_ratings.clear(); self.pitcher_history = defaultdict(list); self.player_history = defaultdict(list)
        if self.player_game.empty:
            self.player_game = self.load_npb_player_features()
        self.player_index = {}
        if not self.player_game.empty:
            for (gid,pid,side),g in self.player_game.groupby(['game_id','player_id','side'],sort=False):
                self.player_index[(str(gid),str(pid),str(side))] = g.iloc[-1].to_dict()
        Xrows, y, meta = [], [], []
        # Deterministic chronological order: datetime then game_id. This handles doubleheaders better than date-only logic.
        games = games.sort_values(["datetime", "game_id"]).reset_index(drop=True)
        last_season=None
        for _, row in games.iterrows():
            cur_season=int(pd.Timestamp(row["datetime"]).year)
            if last_season is not None and cur_season != last_season:
                # Regress Elo at each season boundary; rolling team form naturally
                # retains only recent games through bounded deques.
                for key,val in list(self.elo_ratings.items()):
                    self.elo_ratings[key] = ELO_START + ELO_REGRESSION*(val-ELO_START)
            last_season=cur_season
            feat = self.match_features(row)
            Xrows.append(feat)
            league = row["league"]
            hscore, ascore = float(row["home_score"]), float(row["away_score"])
            if league == "NPB":
                target = 0 if hscore > ascore else 1 if hscore == ascore else 2
            else:
                target = 0 if hscore > ascore else 1
            y.append(target)
            meta.append(row.to_dict())
            self.update_after_game(row)
        X = pd.DataFrame(Xrows).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        return X, np.asarray(y, dtype=int), pd.DataFrame(meta)

    def update_after_game(self, row: pd.Series):
        league = row["league"]
        dt = pd.Timestamp(row["datetime"])
        h, a = norm_team(row["home"], league), norm_team(row["away"], league)
        hs, ass = float(row["home_score"]), float(row["away_score"])
        sh, sa = self.state(league, h), self.state(league, a)
        if hs > ass: hr, ar, hp, ap = 0, 2, 3, 0
        elif hs < ass: hr, ar, hp, ap = 2, 0, 0, 3
        else: hr = ar = 1; hp = ap = 1
        self._update_team(sh, hr, hs, ass, True, hp, dt, row)
        self._update_team(sa, ar, ass, hs, False, ap, dt, row)
        self._update_elo(league, h, a, hs, ass)
        self._update_pitcher_history(row)
        self._update_player_history(row)

    def _update_team(self, s: TeamState, result: int, gf: float, ga: float, home: bool, pts: float, dt: pd.Timestamp, row: pd.Series):
        s.results.append(result); s.gf.append(gf); s.ga.append(ga)
        s.total_matches += 1; s.points += pts; s.total_gf += gf; s.total_ga += ga
        if home:
            s.home_matches += 1; s.home_points += pts; s.home_results.append(result); s.home_gf.append(gf); s.home_ga.append(ga)
        else:
            s.away_matches += 1; s.away_points += pts; s.away_results.append(result); s.away_gf.append(gf); s.away_ga.append(ga)
        # Batting process fields may be supplied by the multi-source collector.
        # Missing values remain missing/neutral; no target-game information is synthesized.
        def fv(name, default=0.0):
            v=row.get(name, default)
            try: return float(v) if pd.notna(v) else default
            except Exception: return default
        prefix="home" if home else "away"
        s.pa.append(fv(f"{prefix}_bat_pa",0.0))
        s.ab.append(fv(f"{prefix}_bat_ab",0.0))
        s.h.append(fv(f"{prefix}_bat_h",0.0))
        s.hr.append(fv(f"{prefix}_bat_hr",0.0))
        s.bb.append(fv(f"{prefix}_bat_bb",0.0))
        s.so.append(fv(f"{prefix}_bat_so",0.0))
        s.doubles.append(fv(f"{prefix}_bat_2b",0.0))
        s.triples.append(fv(f"{prefix}_bat_3b",0.0))
        s.sb.append(fv(f"{prefix}_bat_sb",0.0))
        s.cs.append(fv(f"{prefix}_bat_cs",0.0))
        # A conservative bullpen proxy remains available when detailed bullpen
        # lines are absent. Use the team's own allowed runs, never the opponent
        # team's batting fields, and keep it strictly post-game.
        bp_er=fv(f"{prefix}_bullpen_er", max(0.0,ga-3.0))
        bp_runs=fv(f"{prefix}_bullpen_runs", bp_er)
        bp_app=fv(f"{prefix}_bullpen_apps", 0.0)
        s.bullpen_er.append(bp_er); s.bullpen_runs.append(bp_runs); s.bullpen_appearances.append(bp_app)
        bp = max(0.0, ga - 3.0)
        s.bullpen_ip_3 = max(0.0, s.bullpen_ip_3 * 0.65 + bp * 0.45)
        s.bullpen_ip_7 = max(0.0, s.bullpen_ip_7 * 0.88 + bp * 0.20)
        s.last_dt = dt

    def _update_elo(self, league: str, home: str, away: str, hs: float, aas: float):
        eh = self.elo(league, home); ea = self.elo(league, away)
        expected_h = 1.0 / (1.0 + 10 ** (-(eh + ELO_HOME - ea) / 400.0))
        actual_h = 1.0 if hs > aas else 0.0 if hs < aas else 0.5
        margin = math.log1p(abs(hs - aas))
        k = ELO_K * (1.0 + 0.35 * margin)
        self.elo_ratings[(league, home)] = eh + k * (actual_h - expected_h)
        self.elo_ratings[(league, away)] = ea - k * (actual_h - expected_h)
        # Gentle seasonal/competition regression is handled when a team first appears in a new dataset;
        # no future information is injected here.

    def _update_pitcher_history(self, row: pd.Series):
        # Consume per-game starter lines supplied by the multi-source data layer.
        # Crucially, this is called AFTER match_features() for the target game, so
        # the target game's own pitching line cannot leak into its prediction.
        for side in ("home", "away"):
            p = str(row.get(f"{side}_starter", "") or "")
            if not p:
                continue
            metrics = row.get(f"{side}_starter_metrics")
            if isinstance(metrics, dict):
                self.pitcher_history[(row["league"], p)].append(metrics)
                continue
            cols = {}
            for m in ("era", "whip", "k9", "bb9", "hr9", "fip", "pitches", "k_rate", "bb_rate"):
                v = row.get(f"{side}_starter_{m}")
                if pd.notna(v):
                    try: cols[m] = float(v)
                    except Exception: pass
            if cols:
                cols["starts"] = 1.0
                self.pitcher_history[(row["league"], p)].append(cols)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    def models(self, league: str) -> Dict[str, Any]:
        """High-diversity model pool. Every candidate is trained only on past data."""
        k = 3 if league == "NPB" else 2
        m: Dict[str, Any] = {
            "Logistic": Pipeline([("scale", StandardScaler()), ("m", LogisticRegression(C=0.5, max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE))]),
            "HistGB": HistGradientBoostingClassifier(max_iter=280, learning_rate=0.035, max_leaf_nodes=15, min_samples_leaf=12, l2_regularization=2.0, random_state=RANDOM_STATE),
            "RandomForest": RandomForestClassifier(n_estimators=320, max_depth=10, min_samples_leaf=6, max_features=0.55, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=320, max_depth=12, min_samples_leaf=5, max_features=0.65, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        }
        if LGBMClassifier is not None:
            m["LightGBM"] = LGBMClassifier(n_estimators=300, learning_rate=0.025, num_leaves=15, max_depth=6, min_child_samples=18, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=2.0, objective="multiclass" if k==3 else "binary", num_class=k if k==3 else None, verbosity=-1, random_state=RANDOM_STATE, n_jobs=-1)
        if XGBClassifier is not None:
            m["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.025, min_child_weight=8, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=3.0, objective="multi:softprob" if k==3 else "binary:logistic", num_class=k if k==3 else None, eval_metric="mlogloss" if k==3 else "logloss", tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1)
        if CatBoostClassifier is not None:
            m["CatBoost"] = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.03, loss_function="MultiClass" if k==3 else "Logloss", verbose=False, random_seed=RANDOM_STATE, thread_count=-1, l2_leaf_reg=5.0)
        return m

    def _validation_splits(self, n: int) -> List[Tuple[int,int]]:
        """Several chronological validation windows; no random CV and no future leakage."""
        if n < 120: return []
        windows=[]
        for frac in (0.70, 0.86):
            cut=max(60, int(n*frac))
            val=max(25, min(MIN_VALIDATION, n-cut))
            if cut+val <= n and cut >= 60:
                windows.append((cut, val))
        return list(dict.fromkeys(windows))

    def _sample_weights(self, n: int) -> np.ndarray:
        """Recency weighting: old games remain useful but recent league/team context dominates."""
        if n <= 1: return np.ones(n, dtype=float)
        half_life = float(os.getenv("NPB_RECENCY_HALF_LIFE_GAMES", "1800"))
        age = np.arange(n-1, -1, -1, dtype=float)
        w = np.exp(-np.log(2.0) * age / max(100.0, half_life))
        return np.clip(w, 0.20, 1.0)

    def _fit_model(self, model, X, y, weights=None, league="NPB"):
        if weights is None:
            return model.fit(X, y)
        try:
            if isinstance(model, Pipeline):
                return model.fit(X, y, m__sample_weight=weights)
            return model.fit(X, y, sample_weight=weights)
        except TypeError:
            return model.fit(X, y)

    def _temperature_from_probs(self, p: np.ndarray, y: np.ndarray) -> float:
        """Fit a single temperature on a chronological holdout; T>1 softens overconfidence."""
        if len(p) < 25: return 1.0
        labels=np.asarray(y,dtype=int)
        best_t=1.0; best_ll=float("inf")
        for t in np.linspace(0.65,2.20,32):
            q=np.clip(p,1e-7,1.0) ** (1.0/t)
            q=q/q.sum(axis=1,keepdims=True)
            ll=log_loss(labels,q,labels=list(range(p.shape[1])))
            if ll < best_ll: best_ll=float(ll); best_t=float(t)
        return best_t

    def fit_best(self, X: pd.DataFrame, y: np.ndarray, league: str) -> Tuple[str, Any, Dict[str, float]]:
        if len(X) < MIN_TRAIN or len(np.unique(y)) < 2:
            raise ValueError("Insufficient training data")
        scores=[]
        models=self.models(league)
        splits=self._validation_splits(len(X))
        for name, model in models.items():
            losses=[]
            for cut,val in splits:
                Xfit,Xval=X.iloc[:cut],X.iloc[cut:cut+val]
                yfit,yval=y[:cut],y[cut:cut+val]
                if len(np.unique(yfit)) < (3 if league=="NPB" else 2): continue
                try:
                    self._fit_model(model,Xfit,yfit,self._sample_weights(len(Xfit)),league)
                    p=self.align_proba(model.predict_proba(Xval),model.classes_,league)
                    losses.append(log_loss(yval,p,labels=list(range(3 if league=="NPB" else 2))))
                except Exception as e:
                    self.audit.append({"type":"model_error","model":name,"error":str(e)})
                    break
            if losses:
                scores.append((float(np.mean(losses)),name))
        if not scores: raise RuntimeError("All models failed")
        scores.sort()
        # Fit the best single model for fallback/reporting.
        best_name=scores[0][1]
        best=models[best_name]
        self._fit_model(best,X,y,self._sample_weights(len(X)),league)
        # Store a compact validation leaderboard for the caller.
        return best_name,best,{name:float(sc) for sc,name in scores}

    def fit_ensemble(self, X: pd.DataFrame, y: np.ndarray, league: str):
        """Fit all robust candidates and weight them by inverse chronological validation loss."""
        models=self.models(league); k=3 if league=="NPB" else 2
        splits=self._validation_splits(len(X))
        scored=[]
        for name,model in models.items():
            losses=[]
            for cut,val in splits:
                try:
                    self._fit_model(model,X.iloc[:cut],y[:cut],self._sample_weights(cut),league)
                    p=self.align_proba(model.predict_proba(X.iloc[cut:cut+val]),model.classes_,league)
                    losses.append(log_loss(y[cut:cut+val],p,labels=list(range(k))))
                except Exception:
                    losses=[]; break
            if losses: scored.append((name,float(np.mean(losses))))
        if not scored: return None,{},None
        scored.sort(key=lambda z:z[1])
        top=scored[:5]
        inv=np.array([1/max(x[1],1e-6) for x in top]); inv/=inv.sum()
        fitted=[]
        for (name,loss),w in zip(top,inv):
            model=models[name]
            self._fit_model(model,X,y,self._sample_weights(len(X)),league)
            fitted.append((model,float(w),name))
        # Calibrate the ensemble temperature on the latest chronological validation window.
        temperature=1.0
        if splits and fitted:
            cut,val=splits[-1]
            try:
                raw=np.zeros((val,k))
                inv=np.array([1/max(loss,1e-6) for _,loss in top]); inv/=inv.sum()
                for (name,_loss),w in zip(top,inv):
                    mm=models[name]
                    self._fit_model(mm,X.iloc[:cut],y[:cut],self._sample_weights(cut),league)
                    raw += float(w)*self.align_proba(mm.predict_proba(X.iloc[cut:cut+val]),mm.classes_,league)
                temperature=self._temperature_from_probs(raw,y[cut:cut+val])
            except Exception as e:
                self.audit.append({"type":"calibration_error","error":str(e)})
        # Refit the final ensemble on ALL available past data after calibration.
        # The validation models above are temporary calibration models and must not
        # replace the final full-history estimators.
        for model,w,name in fitted:
            self._fit_model(model,X,y,self._sample_weights(len(X)),league)
        self._last_temperature = temperature
        return fitted,{n:float(l) for n,l in scored},top[0][0]

    def ensemble_proba(self, fitted, X: pd.DataFrame, league: str) -> np.ndarray:
        k=3 if league=="NPB" else 2
        p=np.zeros((len(X),k))
        for model,w,_ in fitted:
            p += float(w)*self.align_proba(model.predict_proba(X),model.classes_,league)
        t=float(getattr(self,"_last_temperature",1.0))
        if abs(t-1.0)>1e-9:
            p=np.clip(p,1e-7,1.0) ** (1.0/t)
            p=p/p.sum(axis=1,keepdims=True)
        return np.apply_along_axis(clip_prob,1,p)

    def align_proba(self, raw: np.ndarray, classes: np.ndarray, league: str) -> np.ndarray:
        k = 3 if league == "NPB" else 2
        out = np.zeros((len(raw), k))
        for j, c in enumerate(classes):
            if int(c) < k: out[:, int(c)] = raw[:, j]
        out = np.apply_along_axis(clip_prob, 1, out)
        return out

    # ------------------------------------------------------------------
    # Walk-forward
    # ------------------------------------------------------------------
    def fit_score_ensemble(self, X: pd.DataFrame, y_home: np.ndarray, y_away: np.ndarray, league: str):
        """Chronological OOS ensemble for run scoring. Uses only pregame X/y history."""
        if len(X) < max(80, MIN_TRAIN // 2):
            return None
        splits = self._validation_splits(len(X))
        specs = [
            ("Poisson", lambda: PoissonRegressor(alpha=0.15, max_iter=1000)),
            ("HistPoisson", lambda: HistGradientBoostingRegressor(loss="poisson", max_iter=180, learning_rate=0.035, max_leaf_nodes=15, l2_regularization=1.5, random_state=42)),
            ("RFReg", lambda: RandomForestRegressor(n_estimators=180, min_samples_leaf=5, max_features=0.75, random_state=42, n_jobs=-1)),
            ("ExtraTreesReg", lambda: ExtraTreesRegressor(n_estimators=180, min_samples_leaf=4, max_features=0.8, random_state=42, n_jobs=-1)),
        ]
        scored=[]
        for name, factory in specs:
            losses=[]
            for tr, va in splits:
                if time.time() - self.started_at >= self.time_budget_sec:
                    break
                try:
                    mh=factory(); ma=factory()
                    self._fit_model(mh, X.iloc[tr], y_home[tr], self._sample_weights(tr), league); self._fit_model(ma, X.iloc[tr], y_away[tr], self._sample_weights(tr), league)
                    ph=np.clip(mh.predict(X.iloc[va]), 0.05, 15)
                    pa=np.clip(ma.predict(X.iloc[va]), 0.05, 15)
                    # Poisson deviance-like NLL; robustly defined for integer/float observed runs.
                    nll_h=np.mean(ph - y_home[va]*np.log(ph) + np.array([math.lgamma(v+1) for v in y_home[va]]))
                    nll_a=np.mean(pa - y_away[va]*np.log(pa) + np.array([math.lgamma(v+1) for v in y_away[va]]))
                    losses.append(float((nll_h+nll_a)/2))
                except Exception:
                    continue
            if losses: scored.append((float(np.mean(losses)), name, factory))
        if not scored: return None
        scored.sort(key=lambda z:z[0])
        top=scored[:3]
        fitted=[]
        weights=[]
        for loss,name,factory in top:
            mh=factory(); ma=factory()
            self._fit_model(mh, X, y_home, self._sample_weights(len(X)), league); self._fit_model(ma, X, y_away, self._sample_weights(len(X)), league)
            w=1.0/max(loss,1e-6)
            fitted.append((name,mh,ma)); weights.append(w)
        weights=np.asarray(weights,float); weights/=weights.sum()
        return {"models":fitted,"weights":weights,"scores":{n:float(l) for l,n,_ in scored}}

    def predict_scores(self, fitted, xrow: pd.DataFrame, league: str) -> Tuple[float,float]:
        if fitted is None:
            return 2.35 if league=="NPB" else 4.55, 2.35 if league=="NPB" else 4.55
        lh=la=0.0
        for w,(name,mh,ma) in zip(fitted["weights"], fitted["models"]):
            lh += float(w)*float(np.clip(mh.predict(xrow)[0],0.05,15.0))
            la += float(w)*float(np.clip(ma.predict(xrow)[0],0.05,15.0))
        return lh,la

    def run_walkforward(self, games: pd.DataFrame, league: str) -> pd.DataFrame:
        games = games.copy()
        games = games[games["league"] == league].sort_values(["datetime", "game_id"]).reset_index(drop=True)
        if len(games) <= MIN_TRAIN + 1:
            print(f"[{league}] insufficient games: {len(games)}")
            return pd.DataFrame()
        # Data-quality gates: the backtest must not silently run on a tiny
        # or starter-free sample.
        if league == "NPB":
            starter_rate = float(
                ((games["home_starter"].fillna("").astype(str).str.len() > 0) &
                 (games["away_starter"].fillna("").astype(str).str.len() > 0)).mean()
            )
            self.audit.append({
                "type": "npb_starter_coverage",
                "games": int(len(games)),
                "both_starter_rate": starter_rate,
            })
            print(f"[NPB AUDIT] both-starter coverage={starter_rate:.1%}")
            if starter_rate < 0.70:
                raise RuntimeError(
                    f"NPB starter coverage too low: {starter_rate:.1%}; "
                    "refusing to run a misleading backtest."
                )

        X, y, meta = self.build_features(games)
        # Resume support: completed OOS predictions are persisted after every
        # retraining block. On a later run, completed game IDs are skipped.
        ck = self.checkpoint_dir / f"{league.lower()}_walkforward.csv"
        existing = pd.DataFrame()
        version_file = ck.with_suffix(".version")
        if ck.exists() and version_file.exists() and version_file.read_text(encoding="utf-8").strip() == self.checkpoint_version:
            try: existing = pd.read_csv(ck)
            except Exception: existing = pd.DataFrame()
        elif ck.exists():
            print(f"[{league}] ignoring stale checkpoint (version mismatch)")
            existing = pd.DataFrame()
        completed_ids = set(existing.get("game_id", pd.Series(dtype=str)).astype(str)) if not existing.empty else set()
        all_rows = existing.to_dict("records") if not existing.empty else []
        start = max(MIN_TRAIN, int(len(X) * 0.25))
        for bstart in range(start, len(X), RETRAIN_EVERY):
            bend = min(len(X), bstart + RETRAIN_EVERY)
            block_ids = set(meta.iloc[bstart:bend]["game_id"].astype(str))
            if block_ids and block_ids.issubset(completed_ids):
                print(f"[{league}] resume skip block {bstart}:{bend} ({len(block_ids)} games already checkpointed)")
                continue
            if time.time() - self.started_at >= self.time_budget_sec:
                self.audit.append({"type":"time_budget","league":league,"bstart":int(bstart),"budget_sec":self.time_budget_sec})
                print(f"[{league}] time budget reached; stopping walk-forward cleanly")
                break
            try:
                fitted, val_scores, best_name = self.fit_ensemble(X.iloc[:bstart], y[:bstart], league)
                if not fitted: raise RuntimeError("ensemble fitting failed")
                name = "Ensemble(" + "+".join(x[2] for x in fitted) + ")"
                p = self.ensemble_proba(fitted, X.iloc[bstart:bend], league)
                score_fit = self.fit_score_ensemble(X.iloc[:bstart], games.iloc[:bstart]["home_score"].astype(float).values, games.iloc[:bstart]["away_score"].astype(float).values, league)
            except Exception as e:
                print(f"[{league}] block {bstart}: model failure {e}")
                continue
            block_rows = []
            for j, idx in enumerate(range(bstart, bend)):
                r = meta.iloc[idx]
                if str(r["game_id"]) in completed_ids:
                    continue
                prob = p[j]
                pred = int(np.argmax(prob))
                actual = int(y[idx])
                target = np.zeros(len(prob)); target[actual] = 1
                ll = float(-math.log(max(prob[actual], 1e-12)))
                br = float(np.sum((prob-target)**2))
                # Dedicated chronological run model.
                fx = X.iloc[idx]
                lam_h, lam_a = self.predict_scores(score_fit, X.iloc[[idx]], league)
                # Small, bounded win-probability consistency adjustment.
                if league == "NPB":
                    split = float(np.clip(prob[0] - prob[2], -0.35, 0.35))
                else:
                    split = float(np.clip(prob[0] - 0.5, -0.35, 0.35))
                lam_h *= (1.0 + 0.08 * split)
                lam_a *= (1.0 - 0.08 * split)
                scores = score_candidates(lam_h, lam_a, 4)
                while len(scores) < 4:
                    scores.append(("その他", 0.0))
                low, high = low_high_probs(lam_h, lam_a)
                block_rows.append({
                    "league": league, "game_id": r["game_id"], "datetime": r["datetime"],
                    "home": r["home"], "away": r["away"], "home_starter": r.get("home_starter", ""), "away_starter": r.get("away_starter", ""),
                    "pred_home": float(prob[0]), "pred_draw": float(prob[1]) if league == "NPB" else np.nan,
                    "pred_away": float(prob[2]) if league == "NPB" else float(prob[1]),
                    "prediction": pred, "actual": actual, "correct": int(pred == actual),
                    "logloss": ll, "brier": br, "model": name,
                    "validation_logloss": json.dumps(val_scores, ensure_ascii=False),
                    "lambda_home": lam_h, "lambda_away": lam_a,
                    "score1": scores[0][0], "score1_prob": scores[0][1], "score2": scores[1][0], "score2_prob": scores[1][1],
                    "score3": scores[2][0], "score3_prob": scores[2][1], "score4": scores[3][0], "score4_prob": scores[3][1],
                    "low": low, "high": high,
                    "actual_home_score": float(r["home_score"]), "actual_away_score": float(r["away_score"]),
                })
            if block_rows:
                all_rows.extend(block_rows)
                completed_ids.update(str(x["game_id"]) for x in block_rows)
                try:
                    pd.DataFrame(all_rows).drop_duplicates(["game_id","model"], keep="last").to_csv(ck, index=False)
                    version_file.write_text(self.checkpoint_version, encoding="utf-8")
                    print(f"[{league}] checkpoint saved: {len(completed_ids)} games")
                except Exception as e:
                    self.audit.append({"type":"checkpoint_write_error","league":league,"error":str(e)})
        return pd.DataFrame(all_rows)

    # ------------------------------------------------------------------
    # Evaluation / reports
    # ------------------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, league: str) -> Dict[str, Any]:
        if df.empty: return {}
        out = {
            "League": league, "Predictions": len(df), "Accuracy": float(df.correct.mean()),
            "LogLoss": float(df.logloss.mean()), "Brier": float(df.brier.mean()),
            "MeanAbsoluteScoreError": float((abs(df.actual_home_score-df.lambda_home)+abs(df.actual_away_score-df.lambda_away)).mean()/2),
            "HighActualRate": float(((df.actual_home_score >= 7) | (df.actual_away_score >= 7)).mean()),
            "LowHighAccuracy": float((((df.high >= 0.5).astype(int)) == (((df.actual_home_score >= 7) | (df.actual_away_score >= 7)).astype(int))).mean()),
            "Top4ScoreHitRate": float(df.apply(lambda r: (("その他" in {str(r.score1),str(r.score2),str(r.score3),str(r.score4)}) if (r.actual_home_score >= 7 or r.actual_away_score >= 7) else (f"{int(r.actual_home_score)}-{int(r.actual_away_score)}" in {str(r.score1),str(r.score2),str(r.score3),str(r.score4)})), axis=1).mean()),
        }
        if league == "MLB":
            try:
                out["AUC"] = float(roc_auc_score(df.actual, df.pred_home))
            except Exception: out["AUC"] = np.nan
        return out

    def save_reports(self, df: pd.DataFrame, league: str):
        RESULTS.mkdir(exist_ok=True)
        if df.empty: return
        df.to_csv(RESULTS / f"{league.lower()}_backtest_results.csv", index=False)
        summary = pd.DataFrame([self.evaluate(df, league)])
        summary.to_csv(RESULTS / f"{league.lower()}_backtest_summary.csv", index=False)
        model = df.groupby("model").agg(Predictions=("correct", "size"), Accuracy=("correct", "mean"), LogLoss=("logloss", "mean"), Brier=("brier", "mean")).reset_index()
        model.to_csv(RESULTS / f"{league.lower()}_model_comparison.csv", index=False)
        # Calibration bins are useful for diagnosing overconfidence.
        if league == "MLB":
            tmp = df.copy(); tmp["bin"] = pd.cut(tmp.pred_home, np.linspace(0,1,11), include_lowest=True)
            cal = tmp.groupby("bin", observed=False).agg(n=("actual","size"), predicted=("pred_home","mean"), actual=("actual","mean")).reset_index()
            cal.to_csv(RESULTS / "mlb_calibration.csv", index=False)

    # ------------------------------------------------------------------
    # Current/future prediction helpers
    # ------------------------------------------------------------------
    def current_mlb_schedule(self, date: str) -> pd.DataFrame:
        data = self._get_json(f"{MLB_API}/schedule", params={"sportId":1, "date":date, "hydrate":"probablePitcher"})
        rows=[]
        for d in data.get("dates", []):
            for g in d.get("games", []):
                t=g.get("teams",{}); h=t.get("home",{}); a=t.get("away",{})
                hp=(h.get("probablePitcher") or {}).get("fullName",""); ap=(a.get("probablePitcher") or {}).get("fullName","")
                # "probable" is not equivalent to officially confirmed. Only mark confirmed when status/game data says it.
                confirmed=bool(hp and ap)
                rows.append({"game_id":g.get("gamePk"),"datetime":g.get("gameDate"),"home":h.get("team",{}).get("name",""),"away":a.get("team",{}).get("name",""),"home_starter":hp,"away_starter":ap,"confirmed_starters":confirmed})
        return pd.DataFrame(rows)

    def build_future_mlb_predictions(self, schedule: pd.DataFrame) -> pd.DataFrame:
        # This method deliberately does NOT guess missing starters.
        if schedule.empty: return schedule
        out=[]
        for _,r in schedule.iterrows():
            if not bool(r.get("confirmed_starters")):
                out.append({**r.to_dict(), "status":"保留", "reason":"両先発の公式確認が揃っていない"})
            else:
                out.append({**r.to_dict(), "status":"予測対象"})
        return pd.DataFrame(out)

    def run(self, npb: bool = True, mlb: bool = True, mlb_start: int = 2020, mlb_end: int = 2026):
        RESULTS.mkdir(exist_ok=True)
        print("="*72); print("BASEBALL BACKTEST SYSTEM / NPB + MLB"); print("="*72)
        if time.time() - self.started_at >= self.time_budget_sec:
            print("[HARD STOP] 30-minute limit reached before processing")
            return
        if npb:
            try:
                npb_raw = self.load_npb_pbp()
                npb_games = self.aggregate_npb_games(npb_raw)
                print(f"NPB games: {len(npb_games)}")
                r = self.run_walkforward(npb_games, "NPB")
                self.save_reports(r, "NPB")
                if not r.empty: self.results.extend(r.to_dict("records"))
            except TimeoutError as e:
                print(f"[NPB STOP] {e}")
            except Exception as e:
                print(f"[NPB ERROR] {type(e).__name__}: {e}")
        if time.time() - self.started_at >= self.time_budget_sec:
            print("[HARD STOP] 30-minute limit reached; skipping remaining leagues")
        elif mlb:
            try:
                mlb_games = self.load_mlb(mlb_start, mlb_end)
                # Actual starters are obtained from completed game feeds where possible.
                # This can be slow for many seasons, so only refresh when explicitly requested.
                if os.getenv("MLB_ENRICH_STARTERS", "0") == "1":
                    mlb_games = self.enrich_mlb_starters(mlb_games)
                    mlb_games.to_csv(self.data_dir / "mlb_games.csv", index=False)
                print(f"MLB games: {len(mlb_games)}")
                r = self.run_walkforward(mlb_games, "MLB")
                self.save_reports(r, "MLB")
                if not r.empty: self.results.extend(r.to_dict("records"))
            except TimeoutError as e:
                print(f"[MLB STOP] {e}")
            except Exception as e:
                print(f"[MLB ERROR] {type(e).__name__}: {e}")
        if self.results:
            pd.DataFrame(self.results).to_csv(RESULTS / "combined_backtest_results.csv", index=False)
        if time.time() - self.started_at >= self.time_budget_sec:
            print("[HARD STOP] computation budget exhausted; writing emergency summary")
        audit = pd.DataFrame(self.audit)
        audit.to_csv(RESULTS / "audit_log.csv", index=False)
        pd.DataFrame([{
            "runtime_seconds": round(time.time() - self.started_at, 2),
            "time_budget_seconds": self.time_budget_sec,
            "budget_ok": bool(time.time() - self.started_at <= self.time_budget_sec),
            "npb_predictions": int(sum(1 for x in self.results if x.get("league") == "NPB")),
            "mlb_predictions": int(sum(1 for x in self.results if x.get("league") == "MLB")),
        }]).to_csv(RESULTS / "runtime_summary.csv", index=False)
        print("="*72); print("COMPLETE"); print("="*72)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npb-only", action="store_true")
    p.add_argument("--mlb-only", action="store_true")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--mlb-start", type=int, default=2020)
    p.add_argument("--mlb-end", type=int, default=2026)
    args = p.parse_args()
    bt = BaseballBacktest(Path(args.data_dir))
    bt.run(npb=not args.mlb_only, mlb=not args.npb_only, mlb_start=args.mlb_start, mlb_end=args.mlb_end)


if __name__ == "__main__":
    main()
