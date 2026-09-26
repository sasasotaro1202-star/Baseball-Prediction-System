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
import hashlib
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
from sklearn.linear_model import LogisticRegression, PoissonRegressor, TweedieRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from research.regime_router import RegimeRouter
from research.competition_taxonomy import classify_mlb, classify_npb
from research.hierarchical_result_model import HierarchicalNPBClassifier
from research.correlated_score import estimate_shared_lambda, low_high as correlated_low_high, top_scores as correlated_top_scores
from evaluation.calibration import fit_temperature, TemperatureCalibration
from core.atomic_io import atomic_write_text

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"

MLB_API = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 30

# Backtest controls
MIN_TRAIN = 100
RETRAIN_EVERY = int(os.getenv("NPB_RETRAIN_EVERY", "150"))
PIT_SAFE_STARTER_DATA = os.getenv("PIT_SAFE_STARTER_DATA", "0") == "1"
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
    """Normalize a valid probability vector; fail closed on invalid values."""
    a = np.asarray(p, dtype=float)
    if a.ndim != 1 or len(a) < 2:
        raise ValueError("probability vector must be one-dimensional with >=2 classes")
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("probability vector contains non-finite or negative values")
    total = float(a.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("probability vector must have a positive finite sum")
    a = np.maximum(a, 1e-9)
    total = float(a.sum())
    return a / total


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


def score_candidates(lam_h: float, lam_a: float, shared: float = 0.0, n: int = 4) -> List[Tuple[str, float]]:
    """Return exact-score candidates from the coherent score distribution."""
    return correlated_top_scores(lam_h, lam_a, shared, n)


def low_high_probs(lam_h: float, lam_a: float, shared: float = 0.0) -> Tuple[float, float]:
    """Canonical 6.5 contract: LOW=total runs <=6, HIGH=total runs >=7."""
    return correlated_low_high(lam_h, lam_a, shared)


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
    # Explicit relief-pitching aggregates when the source provides them.
    # These are kept separate from the conservative workload proxy below.
    bullpen_ip: deque = field(default_factory=lambda: deque(maxlen=20))
    bullpen_h: deque = field(default_factory=lambda: deque(maxlen=20))
    bullpen_bb: deque = field(default_factory=lambda: deque(maxlen=20))
    bullpen_so: deque = field(default_factory=lambda: deque(maxlen=20))
    bullpen_hr: deque = field(default_factory=lambda: deque(maxlen=20))
    bullpen_actual_games: deque = field(default_factory=lambda: deque(maxlen=20))
    bullpen_actual_er: deque = field(default_factory=lambda: deque(maxlen=20))
    bullpen_actual_h: deque = field(default_factory=lambda: deque(maxlen=20))
    bullpen_actual_bb: deque = field(default_factory=lambda: deque(maxlen=20))
    bullpen_actual_so: deque = field(default_factory=lambda: deque(maxlen=20))
    bullpen_actual_hr: deque = field(default_factory=lambda: deque(maxlen=20))
    bullpen_actual_ip: deque = field(default_factory=lambda: deque(maxlen=20))


class BaseballBacktest:
    def __init__(self, data_dir: Path = DATA):
        self.data_dir = Path(data_dir)
        self.states: Dict[Tuple[str, str], TeamState] = {}
        self.elo_ratings: Dict[Tuple[str, str], float] = {}
        self.results: List[Dict[str, Any]] = []
        self.model_scores: List[Dict[str, Any]] = []
        self.started_at = time.time()
        # Honor the workflow-configured budget while keeping an explicit
        # workflow-selectable hard cap. Closed Loop stays below its default cap;
        # long candidate/holdout jobs can opt into a larger bounded cap without
        # changing production defaults.
        try:
            requested_budget = float(os.getenv("BASEBALL_TIME_BUDGET_SEC", "1500"))
            hard_cap = float(os.getenv("BASEBALL_HARD_CAP_SEC", "5400"))
        except ValueError as exc:
            raise ValueError("BASEBALL_TIME_BUDGET_SEC and BASEBALL_HARD_CAP_SEC must be numeric") from exc
        if not np.isfinite(requested_budget) or requested_budget <= 0:
            raise ValueError("BASEBALL_TIME_BUDGET_SEC must be > 0 and finite")
        if not np.isfinite(hard_cap) or hard_cap <= 0:
            raise ValueError("BASEBALL_HARD_CAP_SEC must be > 0 and finite")
        self.time_budget_sec = min(requested_budget, hard_cap)
        # Keep long OOS fits visibly alive without materially increasing compute.
        self.heartbeat_sec = max(10.0, float(os.getenv("BASEBALL_HEARTBEAT_SEC", "45")))
        # NPB and MLB run concurrently in the research wrapper. Limit inner
        # tree/boosting parallelism to avoid CPU oversubscription on free runners.
        self.inner_jobs = max(1, int(os.getenv("BASEBALL_INNER_JOBS", "2")))
        self.audit: List[Dict[str, Any]] = []
        self.checkpoint_dir = RESULTS / "checkpoints"
        self.checkpoint_version = "npb-massive-resume-v7-code-input-fingerprint"
        fingerprint_paths = (
            Path(__file__),
            ROOT / "research_runner_v6.py",
            ROOT / "research" / "regime_router.py",
            ROOT / "research" / "correlated_score.py",
            ROOT / "prediction" / "score_distribution.py",
            ROOT / "research" / "backtest_output_contract.py",
            ROOT / "data" / "npb_pbp_adapter.py",
        )
        digest = hashlib.sha256()
        for fingerprint_path in fingerprint_paths:
            if not fingerprint_path.exists():
                raise RuntimeError(
                    f"checkpoint fingerprint dependency missing: {fingerprint_path}"
                )
            digest.update(str(fingerprint_path.relative_to(ROOT)).encode("utf-8"))
            digest.update(fingerprint_path.read_bytes())
        self.checkpoint_code_fingerprint = digest.hexdigest()
        self._last_temperature = 1.0
        self._ensemble_weight_power = 1.0
        self._model_temperatures = {}
        self._calibration_mode = "ensemble"
        self._ensemble_blend_mode = "linear"
        self._regime_router = None
        self._regime_weights = {}
        # OOF-only second-level stacker. It remains disabled unless nested
        # chronological validation demonstrates a material improvement.
        self._stacking_model = None
        self._stacking_model_names = ()
        self.player_game = pd.DataFrame()
        self.player_history = defaultdict(list)
        self.player_index = {}
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # NPB loader
    # ------------------------------------------------------------------
    def _normalize_npb_pbp(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalize public NPB PBP releases through the canonical PIT-safe adapter."""
        from data.npb_pbp_adapter import normalize_pbp_frame
        return normalize_pbp_frame(raw, data_dir=self.data_dir)

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
            # Preserve the full play-by-play chronology. Collapsing to one row per
            # game here destroys score reconstruction and starter evidence because
            # aggregate_npb_games needs all plays inside each game.
            out=out.sort_values(["date","game_id","row_order"],na_position="last").drop_duplicates(["game_id","row_order"],keep="last").reset_index(drop=True)
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
            hs = g["home_score"].dropna()
            aas = g["away_score"].dropna()
            if len(hs) and len(aas):
                hscore, ascore = float(hs.iloc[-1]), float(aas.iloc[-1])
            else:
                hscore, ascore = self._reconstruct_npb_score(g)
            if np.isnan(hscore) or np.isnan(ascore):
                continue
            # Derive a strictly lagged bullpen-usage proxy from historical PBP.
            # Only pitcher identities/inning sides from the completed game are used;
            # this metadata is consumed after the game and therefore cannot leak
            # into the same game's feature row. The research flag controls whether
            # the resulting workload feature is enabled in model fitting.
            home_bullpen_apps = 0.0
            away_bullpen_apps = 0.0
            if "pitcher_id" in g.columns:
                for side, half_value in (("home", "B"), ("away", "T")):
                    vals = g.loc[
                        g["half_inning"].astype(str).str.upper().str.startswith(half_value),
                        "pitcher_id",
                    ].astype(str).str.strip()
                    vals = [v for v in vals if v and v.lower() not in {"nan", "none"}]
                    unique_pitchers = list(dict.fromkeys(vals))
                    apps = max(0, len(unique_pitchers) - 1)
                    if side == "home":
                        home_bullpen_apps = float(apps)
                    else:
                        away_bullpen_apps = float(apps)
            gt = " ".join(g["game_type"].dropna().astype(str).tolist())
            if any(k in gt for k in NPB_EXCLUDE_KEYWORDS):
                continue
            if gt and not any(k in gt for k in NPB_OFFICIAL_KEYWORDS):
                if any(k in gt.lower() for k in ("open", "spring", "farm", "allstar")):
                    continue
            # The first pitcher appearing in PBP is a realized post-game
            # identity, not announcement-time evidence. Preserve starter names
            # only when explicit publication/availability timestamps are present
            # and are at or before the prediction cutoff.
            cutoff_col = "prediction_cutoff" if "prediction_cutoff" in g.columns else None
            h_ann_col = "home_starter_announced_at" if "home_starter_announced_at" in g.columns else None
            a_ann_col = "away_starter_announced_at" if "away_starter_announced_at" in g.columns else None
            cutoff = pd.to_datetime(g[cutoff_col].iloc[0], errors="coerce", utc=True) if cutoff_col else pd.NaT
            h_ann = pd.to_datetime(g[h_ann_col].iloc[0], errors="coerce", utc=True) if h_ann_col else pd.NaT
            a_ann = pd.to_datetime(g[a_ann_col].iloc[0], errors="coerce", utc=True) if a_ann_col else pd.NaT
            hp = str(g.get("home_starter", pd.Series([""])).iloc[0] or "").strip() if "home_starter" in g.columns else ""
            ap = str(g.get("away_starter", pd.Series([""])).iloc[0] or "").strip() if "away_starter" in g.columns else ""
            h_safe = bool(hp and pd.notna(cutoff) and pd.notna(h_ann) and h_ann <= cutoff)
            a_safe = bool(ap and pd.notna(cutoff) and pd.notna(a_ann) and a_ann <= cutoff)
            rows.append({
                "league": "NPB", "game_id": str(gid), "datetime": dt,
                "home": home, "away": away, "home_score": hscore, "away_score": ascore,
                "home_starter": hp if h_safe else "", "away_starter": ap if a_safe else "",
                "venue": "unknown",
                "confirmed_starters": bool(h_safe and a_safe),
                "starter_evidence_status": "pit_safe" if (h_safe and a_safe) else "unknown",
                "home_bullpen_apps": float(home_bullpen_apps),
                "away_bullpen_apps": float(away_bullpen_apps),
                "game_type": gt,
                "series_description": "",
            })
        out = pd.DataFrame(rows)
        if out.empty:
            raise RuntimeError("No NPB games could be reconstructed.")
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        labels = [classify_npb(x) for x in out.get("game_type", pd.Series("", index=out.index)).fillna("")]
        out["competition"] = [x.competition for x in labels]
        out["competition_stage"] = [x.stage for x in labels]
        out["season_type"] = [x.season_type for x in labels]
        out["game_class"] = [x.game_class for x in labels]
        out["competition_key"] = [x.competition_key for x in labels]
        out["competition_classification_status"] = [x.status for x in labels]
        return out.sort_values(["datetime", "game_id"]).drop_duplicates("game_id").reset_index(drop=True)

    def _first_pitcher(self, g: pd.DataFrame, side: str) -> str:
        col = f"{side}_pitcher"
        if col in g:
            vals = [str(x).strip() for x in g[col].tolist() if str(x).strip() not in ("", "nan", "None")]
            if vals:
                return vals[0]
        return ""

    def _reconstruct_npb_score(self, g: pd.DataFrame) -> Tuple[float, float]:
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
                # A pre-taxonomy cache lacks gameType; do not silently reuse it,
                # otherwise postseason/all-star games become UNKNOWN forever.
                if len(df) > 100 and {"game_type", "series_description"}.issubset(df.columns):
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
                    rows.append({
                        "league": "MLB", "game_id": str(game.get("gamePk")),
                        "datetime": game.get("gameDate"),
                        "home": home.get("team", {}).get("name", ""),
                        "away": away.get("team", {}).get("name", ""),
                        "home_score": home.get("score", np.nan),
                        "away_score": away.get("score", np.nan),
                        "home_starter": hp, "away_starter": ap,
                        # MLB StatsAPI exposes gameType at the game level; retain
                        # it alongside seriesDescription so competition routing is
                        # deterministic and auditable.
                        "game_type": str(game.get("gameType") or ""),
                        "series_description": str(game.get("seriesDescription") or ""),
                        "season": str(game.get("season") or year),
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
                # Feed/live can establish who actually started, but it does not
                # provide a historical public-announcement timestamp. Preserve
                # this only as realized post-game metadata; never promote it into
                # prediction-time starter features.
                if "home_actual_starter" not in games.columns:
                    games["home_actual_starter"] = ""
                if "away_actual_starter" not in games.columns:
                    games["away_actual_starter"] = ""
                if hname: games.at[i, "home_actual_starter"] = hname
                if aname: games.at[i, "away_actual_starter"] = aname
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
        if "game_type" not in out.columns:
            out["game_type"] = ""
        if "series_description" not in out.columns:
            out["series_description"] = ""
        labels = [
            classify_mlb(gt, sd)
            for gt, sd in zip(out["game_type"].fillna(""), out["series_description"].fillna(""))
        ]
        out["competition"] = [x.competition for x in labels]
        out["competition_stage"] = [x.stage for x in labels]
        out["season_type"] = [x.season_type for x in labels]
        out["game_class"] = [x.game_class for x in labels]
        out["competition_key"] = [x.competition_key for x in labels]
        out["competition_classification_status"] = [x.status for x in labels]
        out["game_id"] = out["game_id"].astype(str)

        # A starter identity is usable in prediction-time features only when
        # publication/availability evidence exists no later than prediction time.
        # The Stats API probablePitcher field alone has no historical
        # announcement timestamp, so names from that field are never treated as
        # PIT-safe. This prevents probable/final starter hindsight leakage.
        for c in ("home_starter", "away_starter"):
            if c not in out.columns:
                out[c] = ""
        for c in ("home_starter_announced_at", "away_starter_announced_at"):
            if c not in out.columns:
                out[c] = pd.NaT
            out[c] = pd.to_datetime(out[c], errors="coerce", utc=True)

        if "prediction_cutoff" in out.columns:
            out["prediction_cutoff"] = pd.to_datetime(
                out["prediction_cutoff"], errors="coerce", utc=True
            )
        else:
            out["prediction_cutoff"] = pd.NaT

        home_name = out["home_starter"].fillna("").astype(str).str.strip()
        away_name = out["away_starter"].fillna("").astype(str).str.strip()
        # Compare normalized UTC instants as int64 nanoseconds. This avoids
        # pandas timezone-array comparison edge cases while keeping NaT guarded
        # separately by the explicit notna() predicates below.
        cutoff_ns = out["prediction_cutoff"].astype("int64", copy=False).to_numpy()
        home_ann_ns = out["home_starter_announced_at"].astype("int64", copy=False).to_numpy()
        away_ann_ns = out["away_starter_announced_at"].astype("int64", copy=False).to_numpy()
        home_ok = (
            home_name.ne("")
            & out["home_starter_announced_at"].notna()
            & out["prediction_cutoff"].notna()
            & (home_ann_ns <= cutoff_ns)
        )
        away_ok = (
            away_name.ne("")
            & out["away_starter_announced_at"].notna()
            & out["prediction_cutoff"].notna()
            & (away_ann_ns <= cutoff_ns)
        )
        out["starter_evidence_status"] = np.select(
            [home_ok & away_ok, home_ok ^ away_ok],
            ["pit_safe", "partial"],
            default="unknown",
        )
        out.loc[~home_ok, "home_starter"] = ""
        out.loc[~away_ok, "away_starter"] = ""
        out["confirmed_starters"] = home_ok & away_ok
        return out.sort_values(["datetime", "game_id"]).drop_duplicates("game_id").reset_index(drop=True)

    def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fetch JSON with bounded transient-error retries.

        Historical MLB replay depends on a public API. A single 429/502/503/504
        must not invalidate an otherwise valid OOS run. Retries are bounded and
        deterministic; non-transient HTTP errors still fail closed.
        """
        attempts = 4
        last_exc = None
        for attempt in range(attempts):
            try:
                r = requests.get(
                    url,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                    headers={"User-Agent": "Baseball-Prediction-System/production"},
                )
                if r.status_code in {429, 502, 503, 504} and attempt < attempts - 1:
                    retry_after = r.headers.get("Retry-After")
                    try:
                        delay = min(8.0, max(1.0, float(retry_after)))
                    except (TypeError, ValueError):
                        delay = float(2 ** attempt)
                    time.sleep(delay)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= attempts - 1:
                    raise
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status not in {429, 502, 503, 504}:
                    raise
                time.sleep(float(2 ** attempt))
        raise RuntimeError(f"MLB API request failed after {attempts} attempts: {url}") from last_exc

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
        # Small-sample empirical-Bayes style shrinkage features. The raw
        # rolling rates remain available; these additional features pull volatile
        # short windows toward the league prior without introducing future data.
        # This is deliberately low-dimensional and fixed rather than a tuned
        # high-capacity transform.
        for w in (3, 5, 10, 20):
            n_eff = float(min(s.total_matches, w))
            kappa = 20.0
            f[f"win_shrunk_{w}"] = (
                f[f"win_{w}"] * n_eff + 0.50 * kappa
            ) / max(n_eff + kappa, 1e-9)
            f[f"gd_shrunk_{w}"] = (
                f[f"gd_{w}"] * n_eff
            ) / max(n_eff + kappa, 1e-9)
            if league == "NPB":
                f[f"draw_shrunk_{w}"] = (
                    f[f"draw_{w}"] * n_eff + 0.24 * kappa
                ) / max(n_eff + kappa, 1e-9)

        f["venue_n"] = float(vm)
        f["venue_pts"] = float(vp / vm) if vm else 1.0
        f["venue_gf"] = float(np.mean(gf[-10:])) if gf else 0.0
        f["venue_ga"] = float(np.mean(ga[-10:])) if ga else 0.0
        f["elo"] = self.elo(league, team)
        f["rest_days"] = float(max(0.0, (dt - s.last_dt).total_seconds() / 86400.0)) if s.last_dt is not None else 30.0
        f["matches"] = float(s.total_matches)
        f["bp3"] = float(s.bullpen_ip_3)
        f["bp7"] = float(s.bullpen_ip_7)
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
        # Relief quality is calculated only from explicitly supplied relief
        # aggregates. Missing fields stay missing/zero rather than being
        # reconstructed from the target score.
        bp_ip = float(np.sum(list(s.bullpen_actual_ip)[-10:])) if s.bullpen_actual_ip else 0.0
        bp_h = float(np.sum(list(s.bullpen_actual_h)[-10:])) if s.bullpen_actual_h else 0.0
        bp_bb = float(np.sum(list(s.bullpen_actual_bb)[-10:])) if s.bullpen_actual_bb else 0.0
        bp_so = float(np.sum(list(s.bullpen_actual_so)[-10:])) if s.bullpen_actual_so else 0.0
        bp_hr = float(np.sum(list(s.bullpen_actual_hr)[-10:])) if s.bullpen_actual_hr else 0.0
        bp_er_actual = float(np.sum(list(s.bullpen_actual_er)[-10:])) if s.bullpen_actual_er else 0.0
        bp_games = float(np.sum(list(s.bullpen_actual_games)[-10:])) if s.bullpen_actual_games else 0.0
        f["bp_ip_10"] = bp_ip
        f["bp_era_10"] = 9.0 * bp_er_actual / max(bp_ip, 1e-6) if bp_ip > 0 else 0.0
        f["bp_whip_10"] = (bp_h + bp_bb) / max(bp_ip, 1e-6) if bp_ip > 0 else 0.0
        f["bp_k9_10"] = 9.0 * bp_so / max(bp_ip, 1e-6) if bp_ip > 0 else 0.0
        f["bp_bb9_10"] = 9.0 * bp_bb / max(bp_ip, 1e-6) if bp_ip > 0 else 0.0
        f["bp_hr9_10"] = 9.0 * bp_hr / max(bp_ip, 1e-6) if bp_ip > 0 else 0.0
        f["bp_actual_coverage_10"] = bp_games / max(min(10.0, s.total_matches), 1.0)
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
        weights=np.asarray([max(1.0,p['pa']) for p in prof],dtype=float) if prof else np.array([])
        if len(weights): weights=weights/weights.sum()
        for m in metrics:
            vals=np.asarray([p[m] for p in prof],dtype=float) if prof else np.array([])
            out[f'lineup_{m}']=float(np.average(vals,weights=weights)) if len(vals) else {'avg':0.25,'obp':0.32,'slg':0.38,'iso':0.13,'bb_rate':0.08,'so_rate':0.20,'sb_rate':0.02,'bunt_rate':0.02,'power_rate':0.05,'contact_rate':0.70,'gidp_rate':0.03,'groundout_rate':0.20,'flyout_rate':0.20,'lineout_rate':0.08,'errors_rate':0.0,'recent_iso':0.13,'recent_so_rate':0.20,'recent_bb_rate':0.08,'recent_sb_rate':0.02}[m]
        out['lineup_n']=float(len(prof)); out['lineup_known_n']=float(sum(p['games']>0 for p in prof)); out['lineup_history_coverage']=out['lineup_known_n']/max(out['lineup_n'],1.0)
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

    def _context_pit_safe(self, row: pd.Series) -> bool:
        """Return True only when lineup/weather context is explicitly PIT-safe.

        A historical dataset can contain final lineups or realized weather even
        though those values were not available when the prediction was made.
        Requiring an explicit availability timestamp prevents silent postgame
        enrichment from entering OOS features.
        """
        if os.getenv("PIT_SAFE_CONTEXT_DATA", "0") != "1":
            return False
        cutoff_raw = row.get("prediction_cutoff")
        if cutoff_raw in (None, "", "nan"):
            return False
        try:
            cutoff = pd.Timestamp(cutoff_raw)
            if cutoff.tzinfo is None:
                cutoff = cutoff.tz_localize("UTC")
            else:
                cutoff = cutoff.tz_convert("UTC")
        except Exception:
            return False
        timestamps = []
        for key in ("lineup_announced_at", "weather_available_at"):
            raw = row.get(key)
            if raw in (None, "", "nan"):
                return False
            try:
                ts = pd.Timestamp(raw)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                timestamps.append(ts)
            except Exception:
                return False
        return all(ts <= cutoff for ts in timestamps)

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
        hs = str(row.get("home_starter", "") or "")
        ass = str(row.get("away_starter", "") or "")
        out.update(self.starter_features(league, hs, dt, prefix="hs_"))
        out.update(self.starter_features(league, ass, dt, prefix="as_"))
        # Historical lineups/weather are high-risk PIT inputs: a completed game's
        # row may contain information that was only published after first pitch.
        # Never consume them unless the row carries an explicit pre-cutoff
        # availability timestamp and the operator explicitly enables PIT-safe
        # context data. This keeps ordinary backtests fail-closed rather than
        # silently turning postgame enrichment into predictive signal.
        context_pit_safe = self._context_pit_safe(row)
        out["context_pit_safe"] = float(context_pit_safe)
        if context_pit_safe:
            hpf=self._lineup_features(row,"home",league); apf=self._lineup_features(row,"away",league)
            for k,v in hpf.items(): out[f"h_{k}"]=v
            for k,v in apf.items(): out[f"a_{k}"]=v
            for k in set(hpf)&set(apf): out[f"d_{k}"]=hpf[k]-apf[k]
            for c in ("weather_temp_c", "weather_humidity_pct", "weather_wind_kmh", "weather_precip_mm"):
                if c in row:
                    try: out[c] = float(row.get(c)) if pd.notna(row.get(c)) else 0.0
                    except Exception: out[c] = 0.0
        out["expected_env"] = max(0.5, min(12.0, 0.5 * (hf["gf_10"] + af["gf_10"] + hf["ga_10"] + af["ga_10"])))
        out["matchup_home_bat_vs_away_fip"] = hf.get("bat_avg_10",0.0) - out.get("as_fip",4.0)/20.0
        out["matchup_away_bat_vs_home_fip"] = af.get("bat_avg_10",0.0) - out.get("hs_fip",4.0)/20.0
        out["bullpen_fatigue_diff"] = hf.get("bp_app_10",0.0) - af.get("bp_app_10",0.0)
        out["bullpen_quality_era_diff"] = hf.get("bp_era_10",0.0) - af.get("bp_era_10",0.0)
        out["bullpen_whip_diff"] = hf.get("bp_whip_10",0.0) - af.get("bp_whip_10",0.0)
        out["bullpen_k9_diff"] = hf.get("bp_k9_10",0.0) - af.get("bp_k9_10",0.0)
        out["bullpen_bb9_diff"] = hf.get("bp_bb9_10",0.0) - af.get("bp_bb9_10",0.0)
        out["bullpen_hr9_diff"] = hf.get("bp_hr9_10",0.0) - af.get("bp_hr9_10",0.0)
        out["bullpen_actual_coverage_diff"] = hf.get("bp_actual_coverage_10",0.0) - af.get("bp_actual_coverage_10",0.0)
        out["weather_run_signal"] = (
            (out.get("weather_temp_c",0.0)-20.0)/10.0
            + out.get("weather_wind_kmh",0.0)/30.0
            - out.get("weather_precip_mm",0.0)/5.0
        ) if context_pit_safe else 0.0
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

        # Low-complexity pregame interaction features. These are intentionally
        # algebraic combinations of already PIT-safe signals, so they add
        # matchup structure without introducing new data sources or target
        # dependence. They are especially useful for linear/logistic members;
        # tree models can also exploit them as explicit stability aids.
        elo_diff = float(out.get("d_elo", 0.0))
        starter_quality_gap = float(out.get("starter_x_quality_proxy", 0.0))
        starter_reliability_gap = (
            float(out.get("hs_starts", 0.0)) / (float(out.get("hs_starts", 0.0)) + 8.0)
            - float(out.get("as_starts", 0.0)) / (float(out.get("as_starts", 0.0)) + 8.0)
        )
        out["elo_x_starter_quality_gap"] = elo_diff * starter_quality_gap
        out["elo_x_starter_reliability_gap"] = elo_diff * starter_reliability_gap
        out["form_x_rest_gap"] = float(out.get("d_gd_10", 0.0)) * (
            float(out.get("h_rest_days", 0.0)) - float(out.get("a_rest_days", 0.0))
        )
        out["bullpen_fatigue_x_rest_gap"] = float(out.get("bullpen_fatigue_diff", 0.0)) * (
            1.0 + abs(float(out.get("h_rest_days", 0.0)) - float(out.get("a_rest_days", 0.0)))
        )
        out["offense_power_x_starter_quality"] = float(out.get("offense_power_gap_10", 0.0)) * starter_quality_gap
        out["environment_x_volatility_gap"] = float(out.get("expected_env", 0.0)) * float(out.get("run_volatility_gap_20", 0.0))
        out["starter_quality_reliability_gap"] = starter_quality_gap * starter_reliability_gap
        out["starter_known"] = float(bool(hs and ass))
        return out

    def starter_features(self, league: str, pitcher: str, dt: pd.Timestamp, prefix: str) -> Dict[str, float]:
        hist = self.pitcher_history.get((league, pitcher), []) if pitcher else []
        if not hist:
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

    def _validate_feature_matrix(self, X: pd.DataFrame, league: str) -> None:
        """Fail closed on structural feature corruption before OOS fitting."""
        if X.empty:
            raise RuntimeError(f"{league} feature matrix is empty")
        if X.columns.duplicated().any():
            duplicates = X.columns[X.columns.duplicated()].astype(str).tolist()
            raise RuntimeError(f"{league} feature matrix has duplicate columns: {duplicates[:20]}")
        values = X.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            bad_columns = X.columns[~np.isfinite(values).all(axis=0)].astype(str).tolist()
            raise RuntimeError(
                f"{league} feature matrix has non-finite values: {bad_columns[:20]}"
            )

        # Differential features must be exactly home-minus-away. This catches
        # silent column swaps and feature assembly regressions without using y.
        failures = []
        for col in X.columns:
            name = str(col)
            if not name.startswith("d_"):
                continue
            base = name[2:]
            hcol, acol = f"h_{base}", f"a_{base}"
            if hcol not in X.columns or acol not in X.columns:
                continue
            expected = X[hcol].to_numpy(dtype=float) - X[acol].to_numpy(dtype=float)
            actual = X[name].to_numpy(dtype=float)
            if not np.allclose(actual, expected, rtol=0.0, atol=1e-10):
                failures.append(name)
        if failures:
            raise RuntimeError(
                f"{league} differential feature invariant failed: {failures[:20]}"
            )

        if "home_adv" in X.columns and not np.allclose(
            X["home_adv"].to_numpy(dtype=float), 1.0, rtol=0.0, atol=0.0
        ):
            raise RuntimeError(f"{league} home_adv invariant failed")

        if "expected_env" in X.columns:
            env = X["expected_env"].to_numpy(dtype=float)
            if not np.isfinite(env).all():
                raise RuntimeError(f"{league} expected_env contains non-finite values")

        self.audit.append({
            "type": "feature_integrity_pass",
            "league": league,
            "rows": int(len(X)),
            "columns": int(X.shape[1]),
            "differential_columns_checked": int(sum(
                str(c).startswith("d_") and f"h_{str(c)[2:]}" in X.columns and f"a_{str(c)[2:]}" in X.columns
                for c in X.columns
            )),
        })

    def build_features(self, games: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        self.states.clear(); self.elo_ratings.clear(); self.pitcher_history = defaultdict(list); self.player_history = defaultdict(list)
        if self.player_game.empty:
            self.player_game = self.load_npb_player_features()
        self.player_index = {}
        if not self.player_game.empty:
            for (gid,pid,side),g in self.player_game.groupby(['game_id','player_id','side'],sort=False):
                self.player_index[(str(gid),str(pid),str(side))] = g.iloc[-1].to_dict()
        Xrows, y, meta = [], [], []
        games = games.sort_values(["datetime", "game_id"]).reset_index(drop=True)

        # Freeze state for every exact prediction timestamp. Results from one
        # game must never become features for another game that starts at the
        # same timestamp, because all games in that timestamp group share the
        # same pregame information horizon. State is advanced only after the
        # entire timestamp group has produced features/targets.
        same_timestamp_groups = int(
            games.groupby("datetime", dropna=False).size().gt(1).sum()
        )
        same_timestamp_rows = int(
            games.groupby("datetime", dropna=False).size()
            .loc[lambda s: s > 1].sum()
        )
        self.audit.append({
            "type": "same_timestamp_state_freeze",
            "groups": same_timestamp_groups,
            "rows": same_timestamp_rows,
        })

        last_season = None
        for _, time_group in games.groupby("datetime", sort=False, dropna=False):
            first_dt = pd.Timestamp(time_group.iloc[0]["datetime"])
            cur_season = int(first_dt.year)
            if last_season is not None and cur_season != last_season:
                for key, val in list(self.elo_ratings.items()):
                    self.elo_ratings[key] = ELO_START + ELO_REGRESSION * (val - ELO_START)
            last_season = cur_season

            pending_rows = []
            for _, row in time_group.iterrows():
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
                pending_rows.append(row)

            for row in pending_rows:
                self.update_after_game(row)

        X = pd.DataFrame(Xrows).replace([np.inf, -np.inf], np.nan)
        missing = X.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            self.audit.append({
                "type": "feature_missing_fail_closed",
                "columns": {str(k): int(v) for k, v in missing.items()},
                "rows": int(missing.max()),
            })
            raise RuntimeError(
                "Feature matrix contains undefined values; refusing implicit zero imputation: "
                + ", ".join(f"{k}={v}" for k, v in missing.items())
            )
        X = X.astype(float)
        self._validate_feature_matrix(X, str(games["league"].iloc[0]) if len(games) else "UNKNOWN")
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
        def fv_any(names, default=np.nan):
            for name in names:
                v = row.get(name, np.nan)
                try:
                    if pd.notna(v):
                        return float(v)
                except Exception:
                    pass
            return default

        # Prefer explicit relief-pitching totals from the source. The old
        # max(ga-3, 0) calculation is retained only as a workload/run proxy
        # when no explicit bullpen totals exist; it is never labeled as ERA/WHIP.
        bp_er_explicit = fv_any((f"{prefix}_bullpen_er", f"{prefix}_relief_er"))
        bp_runs_explicit = fv_any((f"{prefix}_bullpen_runs", f"{prefix}_relief_runs"))
        bp_ip_explicit = fv_any((f"{prefix}_bullpen_ip", f"{prefix}_relief_ip"))
        bp_h_explicit = fv_any((f"{prefix}_bullpen_h", f"{prefix}_relief_h"))
        bp_bb_explicit = fv_any((f"{prefix}_bullpen_bb", f"{prefix}_relief_bb"))
        bp_so_explicit = fv_any((f"{prefix}_bullpen_so", f"{prefix}_relief_so"))
        bp_hr_explicit = fv_any((f"{prefix}_bullpen_hr", f"{prefix}_relief_hr"))
        enable_safe_bp = os.getenv("BASEBALL_ENABLE_PIT_SAFE_BULLPEN_USAGE", "0") == "1"
        bp_app=fv(f"{prefix}_bullpen_apps", 0.0) if enable_safe_bp else 0.0

        bp_er = bp_er_explicit if np.isfinite(bp_er_explicit) else max(0.0, ga - 3.0)
        bp_runs = bp_runs_explicit if np.isfinite(bp_runs_explicit) else bp_er
        bp_ip = bp_ip_explicit if np.isfinite(bp_ip_explicit) else 0.0
        bp_h = bp_h_explicit if np.isfinite(bp_h_explicit) else 0.0
        bp_bb = bp_bb_explicit if np.isfinite(bp_bb_explicit) else 0.0
        bp_so = bp_so_explicit if np.isfinite(bp_so_explicit) else 0.0
        bp_hr = bp_hr_explicit if np.isfinite(bp_hr_explicit) else 0.0
        bp_actual = float(np.isfinite(bp_ip_explicit) or np.isfinite(bp_er_explicit) or np.isfinite(bp_so_explicit))
        s.bullpen_er.append(bp_er); s.bullpen_runs.append(bp_runs); s.bullpen_appearances.append(bp_app)
        s.bullpen_ip.append(bp_ip); s.bullpen_h.append(bp_h); s.bullpen_bb.append(bp_bb)
        s.bullpen_so.append(bp_so); s.bullpen_hr.append(bp_hr); s.bullpen_actual_games.append(bp_actual)
        s.bullpen_actual_er.append(bp_er_explicit if np.isfinite(bp_er_explicit) else 0.0)
        s.bullpen_actual_h.append(bp_h_explicit if np.isfinite(bp_h_explicit) else 0.0)
        s.bullpen_actual_bb.append(bp_bb_explicit if np.isfinite(bp_bb_explicit) else 0.0)
        s.bullpen_actual_so.append(bp_so_explicit if np.isfinite(bp_so_explicit) else 0.0)
        s.bullpen_actual_hr.append(bp_hr_explicit if np.isfinite(bp_hr_explicit) else 0.0)
        s.bullpen_actual_ip.append(bp_ip_explicit if np.isfinite(bp_ip_explicit) else 0.0)
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

    def _update_pitcher_history(self, row: pd.Series):
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

    @staticmethod
    def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
        raw = os.getenv(name)
        if raw in (None, ""):
            return int(default)
        try:
            value = int(raw)
        except Exception as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
        return value

    def models(self, league: str) -> Dict[str, Any]:
        k = 3 if league == "NPB" else 2
        fast = os.getenv("BASEBALL_FAST_OOS", "0") == "1"

        # The full research portfolio stays available by default. Scheduled
        # closed-loop runs opt into a bounded free-run profile so one expensive
        # booster cannot consume the complete OOS budget before predictions.
        hist_max_iter = self._env_int("BASEBALL_HISTGB_MAX_ITER", 180 if fast else 280, minimum=1)
        rf_estimators = self._env_int("BASEBALL_RF_ESTIMATORS", 180 if fast else 320, minimum=1)
        et_estimators = self._env_int("BASEBALL_ET_ESTIMATORS", 180 if fast else 320, minimum=1)
        lgbm_estimators = self._env_int("BASEBALL_LGBM_ESTIMATORS", 180 if fast else 300, minimum=1)
        xgb_estimators = self._env_int("BASEBALL_XGB_ESTIMATORS", 120 if fast else 300, minimum=1)
        cat_iterations = self._env_int("BASEBALL_CATBOOST_ITERATIONS", 120 if fast else 300, minimum=1)
        raw_cat_random_strength = os.getenv("BASEBALL_CATBOOST_RANDOM_STRENGTH", "1.0")
        try:
            cat_random_strength = float(raw_cat_random_strength)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "BASEBALL_CATBOOST_RANDOM_STRENGTH must be a finite float >= 0"
            ) from exc
        if not np.isfinite(cat_random_strength) or cat_random_strength < 0:
            raise ValueError("BASEBALL_CATBOOST_RANDOM_STRENGTH must be finite and >= 0")
        cat_deterministic = os.getenv("BASEBALL_CATBOOST_DETERMINISTIC", "0") == "1"

        m: Dict[str, Any] = {
            "Logistic": Pipeline([("scale", StandardScaler()), ("m", LogisticRegression(C=0.5, max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE))]),
            "HistGB": HistGradientBoostingClassifier(max_iter=hist_max_iter, learning_rate=0.035, max_leaf_nodes=15, min_samples_leaf=12, l2_regularization=2.0, random_state=RANDOM_STATE),
            "RandomForest": RandomForestClassifier(n_estimators=rf_estimators, max_depth=10, min_samples_leaf=6, max_features=0.55, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=self.inner_jobs),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=et_estimators, max_depth=12, min_samples_leaf=5, max_features=0.65, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=self.inner_jobs),
            # Analog/nearest-neighbor challenger: a deliberately local model that
            # can capture matchup states that tree ensembles smooth away. It is
            # evaluated only on chronological validation and therefore cannot
            # bypass the existing promotion/OOS gates.
            "KNNAnalog": Pipeline([("scale", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=45, weights="distance", p=2, leaf_size=30, n_jobs=self.inner_jobs))]),
        }
        if league == "NPB":
            # Hierarchical result challenger: separate the rare-draw event,
            # then home/away conditional on non-draw. It is selected only by
            # chronological OOS evidence and cannot bypass the promotion gate.
            m["HierarchicalDrawResult"] = HierarchicalNPBClassifier(
                hgb_max_iter=hist_max_iter,
                random_state=RANDOM_STATE,
            )
        if LGBMClassifier is not None:
            m["LightGBM"] = LGBMClassifier(n_estimators=lgbm_estimators, learning_rate=0.025, num_leaves=15, max_depth=6, min_child_samples=18, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=2.0, objective="multiclass" if k==3 else "binary", num_class=k if k==3 else None, verbosity=-1, random_state=RANDOM_STATE, n_jobs=self.inner_jobs)
        if XGBClassifier is not None:
            m["XGBoost"] = XGBClassifier(n_estimators=xgb_estimators, max_depth=4, learning_rate=0.025, min_child_weight=8, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=3.0, objective="multi:softprob" if k==3 else "binary:logistic", num_class=k if k==3 else None, eval_metric="mlogloss" if k==3 else "logloss", tree_method="hist", random_state=RANDOM_STATE, n_jobs=self.inner_jobs)
        if CatBoostClassifier is not None:
            cat_kwargs = dict(iterations=cat_iterations, depth=6, learning_rate=0.03, loss_function="MultiClass" if k==3 else "Logloss", verbose=False, random_seed=RANDOM_STATE, thread_count=self.inner_jobs, l2_leaf_reg=5.0, random_strength=cat_random_strength)
            if cat_deterministic:
                cat_kwargs["bootstrap_type"] = "No"
                cat_kwargs["random_strength"] = 0.0
                cat_kwargs["thread_count"] = 1
            m["CatBoost"] = CatBoostClassifier(**cat_kwargs)
        return m

    def _validation_splits(self, n: int) -> List[Tuple[int,int]]:
        if n < 120: return []
        windows=[]
        for frac in (0.70, 0.86):
            cut=max(60, int(n*frac))
            val=max(25, min(MIN_VALIDATION, n-cut))
            if cut+val <= n and cut >= 60:
                windows.append((cut, val))
        return list(dict.fromkeys(windows))

    def _check_time_budget(self, stage: str = "") -> None:
        """Fail early at safe Python boundaries before GitHub forces cancellation."""
        elapsed = time.time() - self.started_at
        if elapsed >= self.time_budget_sec:
            self.audit.append({
                "type": "time_budget_exceeded",
                "stage": str(stage),
                "elapsed_seconds": float(elapsed),
                "budget_seconds": float(self.time_budget_sec),
            })
            raise TimeoutError(
                f"Baseball computation budget reached during {stage or 'research'} "
                f"({elapsed:.1f}s >= {self.time_budget_sec:.1f}s)"
            )

    def _sample_weights(self, n: int) -> np.ndarray:
        if n <= 1: return np.ones(n, dtype=float)
        half_life = float(os.getenv("BASEBALL_RECENCY_HALF_LIFE_GAMES", os.getenv("NPB_RECENCY_HALF_LIFE_GAMES", "1800")))
        age = np.arange(n-1, -1, -1, dtype=float)
        w = np.exp(-np.log(2.0) * age / max(100.0, half_life))
        return np.clip(w, 0.20, 1.0)

    def _fit_model(self, model, X, y, weights=None, league="NPB"):
        # A bounded recent-prefix window controls runtime while preserving
        # chronology: every retained row is still earlier than the OOS target.
        rows_in = int(len(X))
        max_fit_rows = self._env_int("BASEBALL_MAX_FIT_ROWS", 0, minimum=0)
        if max_fit_rows and len(X) > max_fit_rows:
            start = len(X) - max_fit_rows
            X = X.iloc[start:]
            y = np.asarray(y)[start:]
            if weights is not None:
                weights = np.asarray(weights)[start:]
            self.audit.append({
                "type": "fit_window_cap",
                "league": str(league),
                "rows_in": rows_in,
                "rows_used": int(len(X)),
                "max_fit_rows": int(max_fit_rows),
            })
        if weights is None:
            return model.fit(X, y)
        try:
            if isinstance(model, Pipeline):
                return model.fit(X, y, m__sample_weight=weights)
            return model.fit(X, y, sample_weight=weights)
        except TypeError:
            return model.fit(X, y)

    def _temperature_from_probs(self, p: np.ndarray, y: np.ndarray) -> float:
        if len(p) < 25:
            return 1.0
        return float(fit_temperature(p, y).temperature)
    def fit_best(self, X: pd.DataFrame, y: np.ndarray, league: str) -> Tuple[str, Any, Dict[str, float]]:
        if len(X) < MIN_TRAIN or len(np.unique(y)) < 2:
            raise ValueError("Insufficient training data")
        scores=[]
        models=self.models(league)
        splits=self._validation_splits(len(X))
        for name, model in models.items():
            self._check_time_budget(f"fit_best:{league}:{name}:start")
            losses=[]
            for cut,val in splits:
                self._check_time_budget(f"fit_best:{league}:{name}:split")
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
        best_name=scores[0][1]
        best=models[best_name]
        self._fit_model(best,X,y,self._sample_weights(len(X)),league)
        return best_name,best,{name:float(sc) for sc,name in scores}

    def _chronological_regime_labels(self, X: pd.DataFrame, splits):
        """Build validation regime labels from each split's training prefix only."""
        labels = {}
        for cut, val in splits:
            if cut <= 0 or val <= 0 or cut + val > len(X):
                raise ValueError("invalid chronological regime split")
            router = RegimeRouter().fit(X.iloc[:cut])
            labels[(cut, val)] = router.labels(X.iloc[cut:cut + val])
        return labels

    def _validation_error_redundancy(
        self,
        validation_predictions: Dict[Tuple[int, int], Dict[str, np.ndarray]],
        y: np.ndarray,
        model_names: Iterable[str],
    ) -> Dict[str, float]:
        """Measure OOS model-error redundancy without using holdout targets."""
        names = tuple(sorted(set(str(x) for x in model_names)))
        if len(names) < 2:
            return {name: 0.0 for name in names}
        chunks: Dict[str, List[np.ndarray]] = {name: [] for name in names}
        for (cut, val), by_model in sorted(validation_predictions.items()):
            yv = np.asarray(y[cut:cut + val], dtype=int)
            for name in names:
                p = by_model.get(name)
                if p is None:
                    continue
                p = np.asarray(p, dtype=float)
                if p.ndim != 2 or len(p) != len(yv) or not np.isfinite(p).all():
                    continue
                if (p < 0).any() or np.any(p.sum(axis=1) <= 0):
                    continue
                p = p / p.sum(axis=1, keepdims=True)
                loss = -np.log(np.clip(p[np.arange(len(yv)), yv], 1e-12, 1.0))
                if np.isfinite(loss).all():
                    chunks[name].append(loss)
        usable = [name for name in names if chunks[name]]
        if len(usable) < 2:
            return {name: 0.0 for name in names}
        matrix = [np.concatenate(chunks[name]) for name in usable]
        try:
            corr = np.corrcoef(np.vstack(matrix))
        except Exception:
            return {name: 0.0 for name in names}
        if corr.ndim != 2 or corr.shape[0] != len(usable):
            return {name: 0.0 for name in names}
        out = {name: 0.0 for name in names}
        for i, name in enumerate(usable):
            vals = [float(corr[i, j]) for j in range(len(usable)) if j != i and np.isfinite(corr[i, j])]
            if vals:
                out[name] = float(np.clip(np.mean(vals), 0.0, 1.0))
        return out

    def _stack_features(
        predictions: Dict[str, np.ndarray],
        model_names: Sequence[str],
    ) -> np.ndarray:
        """Build deterministic second-level features from base-model probabilities."""
        blocks = []
        expected_n = None
        expected_k = None
        for name in model_names:
            if name not in predictions:
                raise ValueError(f"missing stacker member prediction: {name}")
            p = np.asarray(predictions[name], dtype=float)
            if p.ndim != 2 or p.shape[1] < 2:
                raise ValueError(f"invalid stacker member shape for {name}: {p.shape}")
            if not np.isfinite(p).all():
                raise ValueError(f"non-finite stacker member probabilities for {name}")
            p = np.apply_along_axis(clip_prob, 1, p)
            if expected_n is None:
                expected_n, expected_k = p.shape
            if p.shape != (expected_n, expected_k):
                raise ValueError("stacker member probability shapes differ")
            blocks.append(p)
        if not blocks:
            raise ValueError("stacker requires at least one member")
        return np.concatenate(blocks, axis=1)

    def fit_ensemble(self, X: pd.DataFrame, y: np.ndarray, league: str, *, fast_oos: bool = False):
        """Fit an ensemble with leakage-safe regime-specific routing.

        Global chronological validation remains the primary selector. A
        secondary regime router can change the mixture only when enough
        validation observations exist for that regime; otherwise it shrinks
        completely back to the global weights.
        """
        models=self.models(league); k=3 if league=="NPB" else 2
        splits=self._validation_splits(len(X))
        if fast_oos and splits:
            splits=splits[-1:]
        scored=[]
        # Fit regime boundaries only from the data available to the current
        # walk-forward training window. This router never sees the eventual
        # prediction target, and its labels are used consistently across all
        # validation models.
        validation_regime_labels = self._chronological_regime_labels(X, splits)
        regime_losses={}
        regime_counts={}
        validation_predictions = {}
        for cut, val in splits:
            labels=validation_regime_labels[(cut,val)]
            for regime in np.unique(labels):
                regime_counts[str(regime)] = regime_counts.get(str(regime),0) + int(np.sum(labels==regime))
        for name,model in models.items():
            self._check_time_budget(f"fit_ensemble:{league}:{name}:start")
            losses=[]
            for cut,val in splits:
                self._check_time_budget(f"fit_ensemble:{league}:{name}:split")
                try:
                    self._fit_model(model,X.iloc[:cut],y[:cut],self._sample_weights(cut),league)
                    p=self.align_proba(model.predict_proba(X.iloc[cut:cut+val]),model.classes_,league)
                    yv=y[cut:cut+val]
                    validation_predictions.setdefault((cut, val), {})[name] = p
                    losses.append(log_loss(yv,p,labels=list(range(k))))
                    labels=validation_regime_labels[(cut,val)]
                    row_losses=-np.log(np.clip(p[np.arange(len(yv)),yv],1e-12,1.0))
                    for regime in np.unique(labels):
                        mask=labels==regime
                        if mask.any():
                            regime_losses.setdefault(str(regime),{}).setdefault(name,[]).extend(row_losses[mask].tolist())
                except Exception as exc:
                    self.audit.append({"type":"model_error","model":name,"error":str(exc),"stage":"ensemble_validation"})
                    losses=[]; break
            if losses: scored.append((float(np.mean(losses)),name))
        if not scored: return None,{},None
        scored.sort(key=lambda z:z[0])
        top=scored[:3 if fast_oos else 5]
        top_names={name for _,name in top}
        global_losses={name:float(loss) for loss,name in top}

        # Jointly tune conservative regime routing, ensemble concentration and
        # redundancy penalty on chronological validation predictions only.
        weight_power=1.0
        diversity_lambda=0.0
        router_params={
            "min_regime_rows":35,
            "shrinkage":80.0,
            "min_relative_edge":0.03,
        }
        power_grid=(0.50,0.75,1.00,1.25,1.50,2.00)
        router_grid=(
            (25,40.0,0.02),(25,80.0,0.03),(25,120.0,0.05),
            (35,40.0,0.02),(35,80.0,0.03),(35,120.0,0.05),
            (50,40.0,0.02),(50,80.0,0.03),(50,120.0,0.05),
        )
        diversity_grid=(0.0,0.05,0.10,0.15)
        top_names_sorted=tuple(sorted(top_names))
        redundancy=self._validation_error_redundancy(
            validation_predictions, y, top_names_sorted
        ) if validation_predictions else {name:0.0 for name in top_names_sorted}

        def adjust_losses(loss_map, lam):
            return {
                name: float(loss) * (1.0 + float(lam) * float(redundancy.get(name,0.0)))
                for name, loss in loss_map.items()
            }

        if not fast_oos and validation_predictions:
            best_key=(float("inf"), float("inf"), float("inf"), float("inf"), float("inf"))
            for min_rows, shrinkage, min_edge in router_grid:
                for power in power_grid:
                    for lam in diversity_grid:
                        try:
                            adjusted_global=adjust_losses(global_losses, lam)
                            adjusted_regime={
                                str(regime): adjust_losses(
                                    {
                                        name: float(np.mean(vals))
                                        for name, vals in by_model.items()
                                        if name in top_names and vals
                                    }, lam,
                                )
                                for regime, by_model in regime_losses.items()
                            }
                            trial_router=RegimeRouter(
                                min_regime_rows=int(min_rows),
                                shrinkage=float(shrinkage),
                                min_relative_edge=float(min_edge),
                            )
                            wmap=trial_router.weights(
                                adjusted_global,
                                adjusted_regime,
                                regime_counts,
                                power=float(power),
                            )
                            for blend_candidate in ("linear", "log_pool"):
                                trial_ll=[]
                                for (cut,val), by_model_pred in sorted(validation_predictions.items()):
                                    labels=validation_regime_labels[(cut,val)]
                                    yv=np.asarray(y[cut:cut+val],dtype=int)
                                    q=np.zeros((len(yv),k))
                                    for label in np.unique(labels):
                                        idx=np.flatnonzero(labels==label)
                                        regime_w=wmap.get(str(label),{})
                                        members={
                                            name: np.asarray(by_model_pred[name])[idx]
                                            for name in top_names_sorted
                                            if by_model_pred.get(name) is not None and float(regime_w.get(name,0.0)) > 0.0
                                        }
                                        if not members:
                                            raise RuntimeError(f"regime routing produced no positive-weight members for {label}")
                                        member_weights={name: float(regime_w.get(name,0.0)) for name in members}
                                        q[idx] = self._blend_probability_members(members, member_weights, mode=blend_candidate)
                                    q=np.apply_along_axis(clip_prob,1,q)
                                    trial_ll.append(log_loss(yv,q,labels=list(range(k))))
                                if trial_ll:
                                    key=(
                                        float(np.mean(trial_ll)),
                                        0 if blend_candidate == "linear" else 1,
                                        abs(float(power)-1.0),
                                        abs(float(shrinkage)-80.0),
                                        abs(float(min_rows)-35.0)+abs(float(min_edge)-0.03)*100.0,
                                    )
                                    if key < best_key:
                                        best_key=key
                                        weight_power=float(power)
                                        diversity_lambda=float(lam)
                                        self._ensemble_blend_mode=blend_candidate
                                        router_params={
                                            "min_regime_rows":int(min_rows),
                                            "shrinkage":float(shrinkage),
                                            "min_relative_edge":float(min_edge),
                                        }
                        except Exception as exc:
                            self.audit.append({
                                "type":"ensemble_routing_search_error",
                                "min_regime_rows":int(min_rows),
                                "shrinkage":float(shrinkage),
                                "min_relative_edge":float(min_edge),
                                "weight_power":float(power),
                                "diversity_lambda":float(lam),
                                "error":f"{type(exc).__name__}: {exc}",
                            })

        effective_global_losses=adjust_losses(global_losses, diversity_lambda)
        self._ensemble_diversity_lambda=float(diversity_lambda)
        self._ensemble_model_redundancy=dict(redundancy)
        self.audit.append({
            "type":"ensemble_routing_selection",
            "router_params":dict(router_params),
            "weight_power":float(weight_power),
            "diversity_lambda":float(diversity_lambda),
            "blend_mode":str(self._ensemble_blend_mode),
            "model_redundancy":{k:float(v) for k,v in sorted(redundancy.items())},
        })

        fitted=[]
        for (loss,name) in top:
            self._check_time_budget(f"fit_ensemble:{league}:{name}:final_fit")
            model=models[name]
            self._fit_model(model,X,y,self._sample_weights(len(X)),league)
            eff_loss=float(effective_global_losses.get(name,loss))
            fitted.append((model,float(1.0/max(eff_loss,1e-6)**weight_power),name))
        inv=np.asarray([w for _,w,_ in fitted],dtype=float); inv/=max(inv.sum(),1e-12)
        fitted=[(m,float(w),n) for (m,_,n),w in zip(fitted,inv)]
        # Regime-specific routing uses only prefix-trained OOS predictions
        # collected above. Never re-evaluate validation rows with models fitted
        # on the complete X window, which would contaminate the routing signal.
        filtered_regime_losses={
            str(regime): adjust_losses(
                {
                    name: float(np.mean(vals))
                    for name, vals in by_model.items()
                    if name in top_names and vals
                },
                diversity_lambda,
            )
            for regime, by_model in regime_losses.items()
        }
        # The deployment router is fit on the complete current training prefix
        # only after validation evidence has been collected.
        router=RegimeRouter(**router_params).fit(X)
        self._regime_router=router
        self._regime_weights=router.weights(
            global_losses,
            filtered_regime_losses,
            regime_counts,
            power=float(weight_power),
        ) if filtered_regime_losses else {}
        self._ensemble_weight_power=float(weight_power)
        # Calibration is itself a candidate. Compare two low-dimensional,
        # strictly chronological contracts on the same validation folds:
        # (A) calibrate the ensemble after blending, or
        # (B) calibrate each member before regime-aware blending.
        # The lower validation LogLoss wins; the locked holdout remains the
        # independent adoption gate.
        temperature=1.0
        self._model_temperatures = {}
        self._calibration_mode = "ensemble"
        if not fast_oos and splits and fitted:
            try:
                raw_parts=[]
                y_parts=[]
                model_parts={name: [] for _,name in top}
                inv=np.array([
                    1/max(float(effective_global_losses.get(name, loss)), 1e-6)**weight_power
                    for loss, name in top
                ],dtype=float)
                inv/=max(inv.sum(),1e-12)
                for cut,val in splits:
                    self._check_time_budget(f"fit_ensemble:{league}:calibration:{cut}")
                    yv=np.asarray(y[cut:cut+val],dtype=int)
                    stored=validation_predictions.get((cut,val), {})
                    calibration_inputs={name: np.asarray(stored[name], dtype=float) for name,_loss in top if name in stored}
                    raw=self._blend_probability_members(
                        calibration_inputs,
                        {name: float(w) for (name,_loss),w in zip(top,inv)},
                        mode=self._ensemble_blend_mode,
                    )
                    for (name,_loss),w in zip(top,inv):
                        mp=stored.get(name)
                        if mp is None:
                            raise RuntimeError(
                                f"missing stored validation prediction for calibration: "
                                f"{league} {name} cut={cut} val={val}"
                            )
                        mp=np.asarray(mp,dtype=float)
                        if mp.shape != (val,k) or not np.isfinite(mp).all():
                            raise RuntimeError(
                                f"invalid stored validation prediction for calibration: "
                                f"{league} {name} cut={cut} val={val}"
                            )
                        model_parts[name].append(mp)
                    raw_parts.append(raw)
                    y_parts.append(yv)

                if raw_parts:
                    raw_all=np.vstack(raw_parts)
                    y_all=np.concatenate(y_parts)
                    ensemble_cal=self._temperature_from_probs(raw_all,y_all)
                    ensemble_q=np.clip(raw_all,1e-7,1.0) ** (1.0/ensemble_cal)
                    ensemble_q/=ensemble_q.sum(axis=1,keepdims=True)
                    ensemble_ll=float(log_loss(y_all,ensemble_q,labels=list(range(k))))

                    member_temps={}
                    member_q_parts={}
                    for (name,_loss),w in zip(top,inv):
                        mp=np.vstack(model_parts[name])
                        cal=fit_temperature(mp,y_all)
                        member_temps[name]=float(cal.temperature)
                        member_q_parts[name]=cal.transform(mp)
                    member_q=np.zeros_like(raw_all)
                    cursor=0
                    for (name,_loss),w in zip(top,inv):
                        mq=member_q_parts[name]
                        member_q += float(w)*mq
                    member_q=np.apply_along_axis(clip_prob,1,member_q)
                    member_ll=float(log_loss(y_all,member_q,labels=list(range(k))))

                    if member_ll < ensemble_ll - 1e-6:
                        self._calibration_mode="individual"
                        self._model_temperatures=member_temps
                        temperature=1.0
                        self.audit.append({
                            "type":"calibration_selection",
                            "mode":"individual",
                            "ensemble_logloss":ensemble_ll,
                            "individual_logloss":member_ll,
                            "model_temperatures":member_temps,
                            "rows":int(len(y_all)),
                        })
                    else:
                        temperature=float(ensemble_cal)
                        self.audit.append({
                            "type":"calibration_selection",
                            "mode":"ensemble",
                            "ensemble_logloss":ensemble_ll,
                            "individual_logloss":member_ll,
                            "model_temperatures":member_temps,
                            "rows":int(len(y_all)),
                        })
            except Exception as e:
                self.audit.append({"type":"calibration_error","error":str(e)})
        # Conservative second-level stacker: train/evaluate only on
        # already-generated chronological OOF probabilities. Each evaluation
        # fold trains the meta-model only on earlier OOF folds, preventing
        # same-fold meta leakage. It is selected only when it materially beats a
        # chronology-safe probability pool; otherwise the existing routed blend
        # remains authoritative.
        self._stacking_model = None
        self._stacking_model_names = ()
        if not fast_oos and self._calibration_mode == "ensemble" and len(splits) >= 2:
            try:
                ordered_splits = sorted(splits, key=lambda z: (z[0], z[1]))
                meta_scores = []
                base_scores = []
                for meta_idx in range(1, len(ordered_splits)):
                    prior = ordered_splits[:meta_idx]
                    current = ordered_splits[meta_idx]
                    train_blocks = []
                    train_y_blocks = []
                    loss_by_model = {name: [] for name in top_names_sorted}
                    usable = True
                    for key in prior:
                        stored = validation_predictions.get(key, {})
                        cut, val = key
                        y_part = np.asarray(y[cut:cut+val], dtype=int)
                        if any(name not in stored for name in top_names_sorted):
                            usable = False
                            break
                        train_blocks.append(
                            self._stack_features(
                                {name: np.asarray(stored[name], dtype=float) for name in top_names_sorted},
                                top_names_sorted,
                            )
                        )
                        train_y_blocks.append(y_part)
                        for name in top_names_sorted:
                            p_part = np.asarray(stored[name], dtype=float)
                            loss_by_model[name].extend(
                                -np.log(np.clip(p_part[np.arange(len(y_part)), y_part], 1e-12, 1.0)).tolist()
                            )
                    if not usable or not train_blocks:
                        continue
                    meta_X_train = np.vstack(train_blocks)
                    meta_y_train = np.concatenate(train_y_blocks)
                    if len(meta_y_train) < max(80, k * 30) or np.unique(meta_y_train).size < k:
                        continue
                    stored_test = validation_predictions.get(current, {})
                    cut, val = current
                    y_test = np.asarray(y[cut:cut+val], dtype=int)
                    if any(name not in stored_test for name in top_names_sorted):
                        continue
                    test_members = {name: np.asarray(stored_test[name], dtype=float) for name in top_names_sorted}

                    inverse = {}
                    for name in top_names_sorted:
                        mean_loss = float(np.mean(loss_by_model[name]))
                        inverse[name] = 1.0 / max(mean_loss, 1e-6)
                    inv_sum = max(sum(inverse.values()), 1e-12)
                    inverse = {name: weight / inv_sum for name, weight in inverse.items()}
                    base_q = self._blend_probability_members(
                        test_members, inverse, mode=self._ensemble_blend_mode
                    )
                    base_scores.append(float(log_loss(y_test, base_q, labels=list(range(k)))))

                    meta = LogisticRegression(
                        C=0.3,
                        max_iter=1000,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    )
                    meta.fit(meta_X_train, meta_y_train)
                    raw_meta = meta.predict_proba(
                        self._stack_features(test_members, top_names_sorted)
                    )
                    stack_q = np.zeros((len(y_test), k), dtype=float)
                    for j, cls in enumerate(meta.classes_):
                        if int(cls) < k:
                            stack_q[:, int(cls)] = raw_meta[:, j]
                    stack_q = np.apply_along_axis(clip_prob, 1, stack_q)
                    meta_scores.append(float(log_loss(y_test, stack_q, labels=list(range(k)))))

                if meta_scores and base_scores and len(meta_scores) == len(base_scores):
                    meta_mean = float(np.mean(meta_scores))
                    base_mean = float(np.mean(base_scores))
                    self.audit.append({
                        "type": "stacking_selection_test",
                        "rows": int(sum(s[1] for s in ordered_splits)),
                        "meta_folds": int(len(meta_scores)),
                        "stacking_logloss": meta_mean,
                        "base_logloss": base_mean,
                    })
                    if meta_mean < base_mean - 0.005:
                        all_blocks = []
                        all_y = []
                        for key in ordered_splits:
                            stored = validation_predictions.get(key, {})
                            cut, val = key
                            y_part = np.asarray(y[cut:cut+val], dtype=int)
                            if any(name not in stored for name in top_names_sorted):
                                raise RuntimeError(f"missing OOF member for final stacking fit: {key}")
                            all_blocks.append(
                                self._stack_features(
                                    {name: np.asarray(stored[name], dtype=float) for name in top_names_sorted},
                                    top_names_sorted,
                                )
                            )
                            all_y.append(y_part)
                        stack_X = np.vstack(all_blocks)
                        stack_y = np.concatenate(all_y)
                        if np.unique(stack_y).size < k:
                            raise RuntimeError("final stacker OOF training does not contain every target class")
                        final_meta = LogisticRegression(
                            C=0.3,
                            max_iter=1000,
                            solver="lbfgs",
                            random_state=RANDOM_STATE,
                        )
                        final_meta.fit(stack_X, stack_y)
                        raw_all = final_meta.predict_proba(stack_X)
                        stack_all = np.zeros((len(stack_y), k), dtype=float)
                        for j, cls in enumerate(final_meta.classes_):
                            if int(cls) < k:
                                stack_all[:, int(cls)] = raw_all[:, j]
                        stack_all = np.apply_along_axis(clip_prob, 1, stack_all)
                        temperature = self._temperature_from_probs(stack_all, stack_y)
                        self._stacking_model = final_meta
                        self._stacking_model_names = top_names_sorted
                        self.audit.append({
                            "type": "stacking_selection",
                            "selected": True,
                            "stacking_logloss": meta_mean,
                            "base_logloss": base_mean,
                            "temperature": float(temperature),
                            "models": list(top_names_sorted),
                            "rows": int(len(stack_y)),
                        })
            except Exception as exc:
                self.audit.append({
                    "type": "stacking_error",
                    "error": f"{type(exc).__name__}: {exc}",
                })

        self._last_temperature = temperature
        # Final-fit members are already trained on the complete current
        # chronological prefix. Calibration reuses stored OOS predictions, so
        # a second full-data refit is unnecessary and would waste the budget.
        return fitted,{name:float(loss) for loss,name in scored},top[0][0]

    @staticmethod
    def _blend_probability_members(
        predictions: Dict[str, np.ndarray],
        weights: Dict[str, float],
        mode: str = "linear",
    ) -> np.ndarray:
        """Blend positive class-probability matrices in linear or log-opinion space.

        ``log_pool`` is a geometric opinion pool. It uses only the supplied
        OOS/model probabilities and is therefore safe to select on chronological
        validation evidence without introducing new data leakage.
        """
        if not predictions:
            raise ValueError("probability blend requires at least one model")
        mode = str(mode)
        if mode not in {"linear", "log_pool"}:
            raise ValueError(f"unsupported probability blend mode: {mode}")
        shapes = {np.asarray(p, dtype=float).shape for p in predictions.values()}
        if len(shapes) != 1:
            raise ValueError("probability blend inputs have inconsistent shapes")
        shape = next(iter(shapes))
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 1:
            raise ValueError("probability blend inputs must be non-empty 2D matrices")
        n, k = shape
        positive_weights = {name: max(0.0, float(weights.get(name, 0.0))) for name in predictions}
        total_weight = float(sum(positive_weights.values()))
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError("probability blend weights must have a positive finite sum")
        positive_weights = {name: w / total_weight for name, w in positive_weights.items()}
        normalized = {}
        for name, pred in predictions.items():
            p = np.asarray(pred, dtype=float)
            if not np.isfinite(p).all() or (p < 0.0).any():
                raise ValueError(f"invalid probability matrix for {name}")
            p = np.maximum(p, 1e-12)
            row_sum = p.sum(axis=1, keepdims=True)
            if np.any(~np.isfinite(row_sum)) or np.any(row_sum <= 0.0):
                raise ValueError(f"invalid probability row sum for {name}")
            normalized[name] = p / row_sum
        if mode == "linear":
            out = np.zeros((n, k), dtype=float)
            for name, p in normalized.items():
                out += float(positive_weights[name]) * p
        else:
            log_out = np.zeros((n, k), dtype=float)
            for name, p in normalized.items():
                w = float(positive_weights[name])
                if w > 0.0:
                    log_out += w * np.log(p)
            shift = np.max(log_out, axis=1, keepdims=True)
            out = np.exp(log_out - shift)
        out_sum = out.sum(axis=1, keepdims=True)
        if np.any(~np.isfinite(out_sum)) or np.any(out_sum <= 0.0):
            raise ValueError("probability blend produced invalid normalization")
        return out / out_sum

    def ensemble_proba(self, fitted, X: pd.DataFrame, league: str) -> np.ndarray:
        k=3 if league=="NPB" else 2
        p=np.zeros((len(X),k))
        labels=self._regime_router.labels(X) if self._regime_router is not None else np.array(["global"]*len(X))
        for label in np.unique(labels):
            idx=np.flatnonzero(labels==label)
            sub=X.iloc[idx]
            weights=self._regime_weights.get(str(label))
            members={}
            member_weights={}
            for model,global_w,name in fitted:
                w=float(weights.get(name,global_w)) if weights else float(global_w)
                if w <= 0.0:
                    continue
                raw=self.align_proba(model.predict_proba(sub),model.classes_,league)
                if self._calibration_mode == "individual":
                    t=float(self._model_temperatures.get(name,1.0))
                    if abs(t-1.0)>1e-9:
                        raw=TemperatureCalibration(t).transform(raw)
                members[name]=raw
                member_weights[name]=w
            if not members:
                raise RuntimeError(f"ensemble produced no positive-weight models for regime {label}")
            if (
                self._stacking_model is not None
                and self._calibration_mode == "ensemble"
                and all(name in members for name in self._stacking_model_names)
            ):
                stack_members = {
                    name: members[name]
                    for name in self._stacking_model_names
                }
                raw_meta = self._stacking_model.predict_proba(
                    self._stack_features(stack_members, self._stacking_model_names)
                )
                stack_q = np.zeros((len(sub), k), dtype=float)
                for j, cls in enumerate(self._stacking_model.classes_):
                    if int(cls) < k:
                        stack_q[:, int(cls)] = raw_meta[:, j]
                p[idx] = np.apply_along_axis(clip_prob, 1, stack_q)
            else:
                blend_mode = "linear" if self._calibration_mode == "individual" else self._ensemble_blend_mode
                p[idx]=self._blend_probability_members(members, member_weights, mode=blend_mode)

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

    def fit_score_ensemble(self, X: pd.DataFrame, y_home: np.ndarray, y_away: np.ndarray, league: str):
        if len(X) < max(80, MIN_TRAIN // 2):
            return None
        fast = os.getenv("BASEBALL_FAST_OOS", "0") == "1"
        splits = self._validation_splits(len(X))
        if fast and splits:
            splits = splits[-1:]
        score_fast_validation = os.getenv("BASEBALL_SCORE_FAST_VALIDATION", "0") == "1"
        if score_fast_validation and splits:
            splits = splits[-1:]
        score_tree_estimators = self._env_int(
            "BASEBALL_SCORE_TREE_ESTIMATORS", 90 if fast else 180, minimum=1
        )
        score_hist_iter = self._env_int(
            "BASEBALL_SCORE_HIST_MAX_ITER", 120 if fast else 180, minimum=1
        )
        specs = [
            ("Poisson", lambda: PoissonRegressor(alpha=0.15, max_iter=1000)),
            # Tweedie adds a flexible mean-variance relationship for run counts.
            # It is a challenger only; validation and the locked holdout decide
            # whether it contributes to the deployed score ensemble.
            ("Tweedie", lambda: TweedieRegressor(
                power=1.5, alpha=0.08, link="log", max_iter=1000
            )),
            ("HistPoisson", lambda: HistGradientBoostingRegressor(loss="poisson", max_iter=score_hist_iter, learning_rate=0.035, max_leaf_nodes=15, l2_regularization=1.5, random_state=42)),
            ("RFReg", lambda: RandomForestRegressor(n_estimators=score_tree_estimators, min_samples_leaf=5, max_features=0.75, random_state=42, n_jobs=self.inner_jobs)),
            ("ExtraTreesReg", lambda: ExtraTreesRegressor(n_estimators=score_tree_estimators, min_samples_leaf=4, max_features=0.8, random_state=42, n_jobs=self.inner_jobs)),
        ]
        scored=[]
        residuals_by_model={}
        for name, factory in specs:
            self._check_time_budget(f"fit_score_ensemble:{league}:{name}:start")
            losses=[]
            for tr, va in splits:
                self._check_time_budget(f"fit_score_ensemble:{league}:{name}:split")
                self._check_time_budget(
                    f"fit_score_ensemble:{league}:{name}:validation_loop"
                )
                try:
                    mh=factory(); ma=factory()
                    self._fit_model(mh, X.iloc[tr], y_home[tr], self._sample_weights(tr), league); self._fit_model(ma, X.iloc[tr], y_away[tr], self._sample_weights(tr), league)
                    ph=np.clip(mh.predict(X.iloc[va]), 0.05, 15)
                    pa=np.clip(ma.predict(X.iloc[va]), 0.05, 15)
                    nll_h=np.mean(ph - y_home[va]*np.log(ph) + np.array([math.lgamma(v+1) for v in y_home[va]]))
                    nll_a=np.mean(pa - y_away[va]*np.log(pa) + np.array([math.lgamma(v+1) for v in y_away[va]]))
                    losses.append(float((nll_h+nll_a)/2))
                    residuals_by_model.setdefault(name,[[],[]])[0].extend((y_home[va]-ph).tolist())
                    residuals_by_model.setdefault(name,[[],[]])[1].extend((y_away[va]-pa).tolist())
                except Exception as exc:
                    self.audit.append({
                        "type": "score_model_error",
                        "model": name,
                        "stage": "score_validation",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    continue
            if losses: scored.append((float(np.mean(losses)), name, factory))
        if not scored: return None
        scored.sort(key=lambda z:z[0])
        top=scored[:3]
        top_names={name for _,name,_ in top}
        # top contains (loss, model_name, factory); unpack all three fields.
        # Keep the factory available for regime-specific validation.
        global_losses={name:float(loss) for loss,name,_factory in top}
        fitted=[]
        weights=[]
        for loss,name,factory in top:
            self._check_time_budget(f"fit_score_ensemble:{league}:{name}:final_fit")
            mh=factory(); ma=factory()
            self._fit_model(mh, X, y_home, self._sample_weights(len(X)), league); self._fit_model(ma, X, y_away, self._sample_weights(len(X)), league)
            w=1.0/max(loss,1e-6)
            fitted.append((name,mh,ma)); weights.append(w)
        weights=np.asarray(weights,float); weights/=weights.sum()

        # Validation regime boundaries are fitted independently from each
        # chronological training prefix. The final router for deployment is
        # fitted on the complete training prefix only after validation.
        validation_regime_labels = self._chronological_regime_labels(X, splits)
        regime_losses={}
        regime_counts={}
        for tr, va in splits:
            self._check_time_budget(f"fit_score_ensemble:{league}:regime_split")
            labels=validation_regime_labels[(tr, va)]
            for regime in np.unique(labels):
                idx=np.flatnonzero(labels==regime)
                if len(idx) == 0:
                    continue
                regime_counts[str(regime)]=regime_counts.get(str(regime),0)+len(idx)
            for loss,name,factory in top:
                try:
                    mh=factory(); ma=factory()
                    self._fit_model(mh,X.iloc[:tr],y_home[:tr],self._sample_weights(tr),league)
                    self._fit_model(ma,X.iloc[:tr],y_away[:tr],self._sample_weights(tr),league)
                    ph=np.clip(mh.predict(X.iloc[tr:tr+va]),0.05,15)
                    pa=np.clip(ma.predict(X.iloc[tr:tr+va]),0.05,15)
                    nll=(ph-y_home[tr:tr+va]*np.log(ph)+np.array([math.lgamma(v+1) for v in y_home[tr:tr+va]]))
                    nll+=(pa-y_away[tr:tr+va]*np.log(pa)+np.array([math.lgamma(v+1) for v in y_away[tr:tr+va]]))
                    for regime in np.unique(labels):
                        idx=np.flatnonzero(labels==regime)
                        if len(idx):
                            regime_losses.setdefault(str(regime),{}).setdefault(name,[]).extend((nll[idx]/2).tolist())
                except Exception as exc:
                    self.audit.append({
                        "type": "score_model_error",
                        "model": name,
                        "stage": "score_regime_validation",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    continue
        best_score_model=top[0][1]
        residual_h,residual_a=residuals_by_model.get(best_score_model,([],[]))
        shared_lambda=estimate_shared_lambda(residual_h,residual_a)
        filtered={r:{n:float(np.mean(v)) for n,v in by.items() if n in top_names and v} for r,by in regime_losses.items()}
        final_router=RegimeRouter().fit(X)
        regime_weights=final_router.weights(global_losses,filtered,regime_counts) if filtered else {}
        return {"models":fitted,"weights":weights,"scores":global_losses,
                "regime_router":final_router,"regime_weights":regime_weights,"shared_lambda":shared_lambda}

    def predict_scores(self, fitted, xrow: pd.DataFrame, league: str) -> Tuple[float,float,float]:
        if fitted is None:
            base = 2.35 if league=="NPB" else 4.55
            return base, base, 0.0
        router=fitted.get("regime_router")
        labels=router.labels(xrow) if router is not None else np.array(["global"])
        label=str(labels[0]) if len(labels) else "global"
        regime_weights=fitted.get("regime_weights",{}).get(label)
        lh=la=0.0
        for i,(name,mh,ma) in enumerate(fitted["models"]):
            w=float(regime_weights.get(name,fitted["weights"][i])) if regime_weights else float(fitted["weights"][i])
            lh += w*float(np.clip(mh.predict(xrow)[0],0.05,15.0))
            la += w*float(np.clip(ma.predict(xrow)[0],0.05,15.0))
        return lh,la,float(np.clip(fitted.get("shared_lambda",0.0),0.0,min(lh,la)*0.75 if min(lh,la)>0 else 0.0))

    def run_walkforward(self, games: pd.DataFrame, league: str) -> pd.DataFrame:
        games = games.copy()
        games = games[games["league"] == league].sort_values(["datetime", "game_id"]).reset_index(drop=True)
        if len(games) <= MIN_TRAIN + 1:
            self.audit.append({
                "type": "insufficient_oos_games",
                "league": league,
                "games": int(len(games)),
                "minimum_required": int(MIN_TRAIN + 2),
            })
            raise RuntimeError(
                f"{league} OOS input is too small: {len(games)} <= {MIN_TRAIN + 1}"
            )
        if league == "NPB":
            starter_rate = float(
                ((games["home_starter"].fillna("").astype(str).str.len() > 0) &
                 (games["away_starter"].fillna("").astype(str).str.len() > 0)).mean()
            )
            self.audit.append({"type": "npb_starter_coverage", "games": int(len(games)), "both_starter_rate": starter_rate})
            print(f"[NPB AUDIT] both-starter coverage={starter_rate:.1%}")
            if PIT_SAFE_STARTER_DATA and starter_rate < 0.70:
                raise RuntimeError(f"NPB starter coverage too low: {starter_rate:.1%}; refusing to run a misleading PIT-safe backtest.")

        if not PIT_SAFE_STARTER_DATA:
            games = games.copy()
            games["home_starter"] = ""
            games["away_starter"] = ""
            games["confirmed_starters"] = False
            games["starter_evidence_status"] = "not_pit_safe"
        X, y, meta = self.build_features(games)

        # A game_id-only resume key is insufficient: a source correction can
        # change the derived feature state for every later game. Store a stable
        # fingerprint of the full ordered input row and invalidate the entire
        # checkpoint when any completed source row no longer matches.
        input_fingerprints: Dict[str, str] = {}
        for _, row in meta.iterrows():
            payload = {
                str(k): row[k]
                for k in meta.columns
            }
            encoded = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")
            ).encode("utf-8")
            input_fingerprints[str(row["game_id"])] = hashlib.sha256(encoded).hexdigest()

        ck = self.checkpoint_dir / f"{league.lower()}_walkforward.csv"
        existing = pd.DataFrame()
        version_file = ck.with_suffix(".version")
        expected_checkpoint_version = (
            f"{self.checkpoint_version}:{self.checkpoint_code_fingerprint}"
        )
        checkpoint_valid = (
            ck.exists()
            and version_file.exists()
            and version_file.read_text(encoding="utf-8").strip() == expected_checkpoint_version
        )
        if checkpoint_valid:
            try:
                existing = pd.read_csv(ck)
            except Exception:
                existing = pd.DataFrame()
                checkpoint_valid = False
        elif ck.exists():
            print(f"[{league}] ignoring stale checkpoint (version mismatch)")
            existing = pd.DataFrame()

        if checkpoint_valid and not existing.empty:
            if "input_fingerprint" not in existing.columns:
                print(f"[{league}] ignoring stale checkpoint (input fingerprints missing)")
                existing = pd.DataFrame()
            else:
                existing_ids = existing["game_id"].astype(str)
                fingerprint_ok = True
                for game_id, fingerprint in existing[["game_id", "input_fingerprint"]].astype(str).itertuples(index=False):
                    if input_fingerprints.get(game_id) != fingerprint:
                        fingerprint_ok = False
                        break
                if not fingerprint_ok:
                    print(f"[{league}] ignoring stale checkpoint (input fingerprint mismatch)")
                    existing = pd.DataFrame()

        completed_ids = set(existing.get("game_id", pd.Series(dtype=str)).astype(str)) if not existing.empty else set()
        all_rows = existing.to_dict("records") if not existing.empty else []
        start = max(MIN_TRAIN, int(len(X) * 0.25))
        next_heartbeat = time.monotonic()
        total_blocks = int(math.ceil(max(0, len(X) - start) / max(RETRAIN_EVERY, 1)))
        block_number = 0
        for bstart in range(start, len(X), RETRAIN_EVERY):
            block_number += 1
            bend = min(len(X), bstart + RETRAIN_EVERY)
            block_ids = set(meta.iloc[bstart:bend]["game_id"].astype(str))
            now_mono = time.monotonic()
            if now_mono >= next_heartbeat:
                elapsed = max(0.0, time.time() - self.started_at)
                remaining = max(0.0, self.time_budget_sec - elapsed)
                print(
                    f"[{league} HEARTBEAT] block={block_number}/{total_blocks} "
                    f"range={bstart}:{bend} completed={len(completed_ids)}/{max(1, len(X)-start)} "
                    f"elapsed={elapsed:.0f}s budget_remaining={remaining:.0f}s",
                    flush=True,
                )
                next_heartbeat = now_mono + self.heartbeat_sec
            if block_ids and block_ids.issubset(completed_ids):
                print(f"[{league}] resume skip block {bstart}:{bend} ({len(block_ids)} games already checkpointed)")
                continue
            if time.time() - self.started_at >= self.time_budget_sec:
                completed_oos = int(len(completed_ids))
                expected_oos = int(max(0, len(X) - start))
                self.audit.append({"type":"time_budget","league":league,"bstart":int(bstart),"budget_sec":self.time_budget_sec})
                self.audit.append({
                    "type": "walkforward_incomplete",
                    "league": league,
                    "expected_games": expected_oos,
                    "completed_games": completed_oos,
                    "missing_games": max(0, expected_oos - completed_oos),
                    "reason": "time_budget_before_block",
                })
                print(f"[{league}] time budget reached; stopping walk-forward with resumable checkpoint")
                raise TimeoutError(
                    f"walk-forward incomplete: {league} time budget reached before block {bstart}:{bend}"
                )
            try:
                print(f"[{league} HEARTBEAT] fitting block={block_number}/{total_blocks} train={bstart} eval={bend-bstart}", flush=True)
                fast_oos = os.getenv("BASEBALL_FAST_OOS", "0") == "1"
                fitted, val_scores, best_name = self.fit_ensemble(
                    X.iloc[:bstart], y[:bstart], league, fast_oos=fast_oos
                )
                if not fitted: raise RuntimeError("ensemble fitting failed")
                name = "Ensemble(" + "+".join(x[2] for x in fitted) + ")"
                p = self.ensemble_proba(fitted, X.iloc[bstart:bend], league)
                score_fit = self.fit_score_ensemble(X.iloc[:bstart], games.iloc[:bstart]["home_score"].astype(float).values, games.iloc[:bstart]["away_score"].astype(float).values, league)
            except TimeoutError:
                raise
            except Exception as e:
                print(f"[{league}] block {bstart}: model failure {e}")
                completed_oos = int(len(completed_ids))
                expected_oos = int(max(0, len(X) - start))
                self.audit.append({
                    "type": "walkforward_block_failure",
                    "league": league,
                    "bstart": int(bstart),
                    "bend": int(bend),
                    "error": f"{type(e).__name__}: {e}",
                })
                self.audit.append({
                    "type": "walkforward_incomplete",
                    "league": league,
                    "expected_games": expected_oos,
                    "completed_games": completed_oos,
                    "missing_games": max(0, expected_oos - completed_oos),
                    "reason": "model_block_failure",
                })
                raise RuntimeError(
                    f"walk-forward incomplete: {league} block {bstart}:{bend} failed"
                ) from e
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
                fx = X.iloc[idx]
                lam_h, lam_a, shared = self.predict_scores(score_fit, X.iloc[[idx]], league)
                if league == "NPB":
                    split = float(np.clip(prob[0] - prob[2], -0.35, 0.35))
                else:
                    split = float(np.clip(prob[0] - 0.5, -0.35, 0.35))
                lam_h *= (1.0 + 0.08 * split)
                lam_a *= (1.0 - 0.08 * split)
                scores = score_candidates(lam_h, lam_a, shared, 4)
                while len(scores) < 4:
                    scores.append(("その他", 0.0))
                low, high = low_high_probs(lam_h, lam_a, shared)
                block_rows.append({
                    "league": league, "game_id": r["game_id"], "datetime": r["datetime"],
                    "input_fingerprint": input_fingerprints[str(r["game_id"])],
                    "home": r["home"], "away": r["away"], "home_starter": r.get("home_starter", ""), "away_starter": r.get("away_starter", ""),
                    "pred_home": float(prob[0]), "pred_draw": float(prob[1]) if league == "NPB" else np.nan,
                    "pred_away": float(prob[2]) if league == "NPB" else float(prob[1]),
                    "prediction": pred, "actual": actual, "correct": int(pred == actual),
                    "logloss": ll, "brier": br, "model": name,
                    "validation_logloss": json.dumps(val_scores, ensure_ascii=False),
                    "lambda_home": lam_h, "lambda_away": lam_a, "shared_lambda": shared,
                    "score1": scores[0][0], "score1_prob": scores[0][1], "score2": scores[1][0], "score2_prob": scores[1][1],
                    "score3": scores[2][0], "score3_prob": scores[2][1], "score4": scores[3][0], "score4_prob": scores[3][1],
                    "low": low, "high": high,
                    "actual_home_score": float(r["home_score"]), "actual_away_score": float(r["away_score"]),
                })
            if block_rows:
                all_rows.extend(block_rows)
                completed_ids.update(str(x["game_id"]) for x in block_rows)
                try:
                    checkpoint_text = (
                        pd.DataFrame(all_rows)
                        .drop_duplicates(["game_id","model"], keep="last")
                        .to_csv(index=False)
                    )
                    atomic_write_text(ck, checkpoint_text)
                    atomic_write_text(version_file, expected_checkpoint_version + "\n")
                    print(f"[{league}] checkpoint saved: {len(completed_ids)} games", flush=True)
                except Exception as e:
                    self.audit.append({"type":"checkpoint_write_error","league":league,"error":str(e)})
                    raise RuntimeError(
                        f"{league} checkpoint persistence failed; refusing to report completed OOS: {e}"
                    ) from e
        expected_ids = set(meta.iloc[start:]["game_id"].astype(str))
        completed_current_ids = completed_ids & expected_ids
        missing_ids = expected_ids - completed_current_ids
        if missing_ids:
            # A partial walk-forward is not valid OOS evidence. This catches
            # both block-level model failures and time-budget exhaustion.
            # Checkpoints remain durable so the next bounded retry can resume.
            self.audit.append({
                "type": "walkforward_incomplete",
                "league": league,
                "expected_games": int(len(expected_ids)),
                "completed_games": int(len(completed_current_ids)),
                "missing_games": int(len(missing_ids)),
            })
            raise RuntimeError(
                f"{league} walk-forward incomplete: "
                f"completed={len(completed_current_ids)}/{len(expected_ids)}; "
                "refusing partial OOS evidence."
            )
        return pd.DataFrame(all_rows)

    def evaluate(self, df: pd.DataFrame, league: str) -> Dict[str, Any]:
        if df.empty: return {}
        out = {
            "League": league, "Predictions": len(df), "Accuracy": float(df.correct.mean()),
            "LogLoss": float(df.logloss.mean()), "Brier": float(df.brier.mean()),
            "MeanAbsoluteScoreError": float((abs(df.actual_home_score-df.lambda_home)+abs(df.actual_away_score-df.lambda_away)).mean()/2),
            "HighActualRate": float(((df.actual_home_score + df.actual_away_score) >= 7).mean()),
            "LowHighAccuracy": float((((df.high >= 0.5).astype(int)) == (((df.actual_home_score + df.actual_away_score) >= 7).astype(int))).mean()),
            "Top4ScoreHitRate": float(df.apply(lambda r: ((r.actual_home_score < 7 and r.actual_away_score < 7) and (f"{int(r.actual_home_score)}-{int(r.actual_away_score)}" in {str(r.score1), str(r.score2), str(r.score3), str(r.score4)})), axis=1).mean()),
        }
        if league == "MLB":
            try: out["AUC"] = float(roc_auc_score(df.actual, df.pred_home))
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
        if league == "MLB":
            tmp = df.copy(); tmp["bin"] = pd.cut(tmp.pred_home, np.linspace(0,1,11), include_lowest=True)
            cal = tmp.groupby("bin", observed=False).agg(n=("actual","size"), predicted=("pred_home","mean"), actual=("actual","mean")).reset_index()
            cal.to_csv(RESULTS / "mlb_calibration.csv", index=False)

    def current_mlb_schedule(self, date: str) -> pd.DataFrame:
        data = self._get_json(f"{MLB_API}/schedule", params={"sportId":1, "date":date, "hydrate":"probablePitcher"})
        rows=[]
        for d in data.get("dates", []):
            for g in d.get("games", []):
                t=g.get("teams",{}); h=t.get("home",{}); a=t.get("away",{})
                hp=(h.get("probablePitcher") or {}).get("fullName",""); ap=(a.get("probablePitcher") or {}).get("fullName","")
                confirmed=False
                rows.append({"game_id":g.get("gamePk"),"datetime":g.get("gameDate"),"home":h.get("team",{}).get("name",""),"away":a.get("team",{}).get("name",""),"home_starter":hp,"away_starter":ap,"confirmed_starters":False,
                         "starter_evidence_status":"official_probable_only" if (hp and ap) else "missing",
                         "starter_source":"MLB Stats API schedule"})
        return pd.DataFrame(rows)

    def build_future_mlb_predictions(self, schedule: pd.DataFrame) -> pd.DataFrame:
        if schedule.empty: return schedule
        out=[]
        for _,r in schedule.iterrows():
            if not bool(r.get("confirmed_starters")) or str(r.get("starter_evidence_status","")) != "official_announced":
                out.append({**r.to_dict(), "status":"保留", "reason":"公式発表済み先発のPIT証拠が揃っていない"})
            else:
                out.append({**r.to_dict(), "status":"予測対象"})
        return pd.DataFrame(out)

    def run(self, npb: bool = True, mlb: bool = True, mlb_start: int = 2020, mlb_end: int = 2026):
        RESULTS.mkdir(exist_ok=True)
        print("="*72); print("BASEBALL BACKTEST SYSTEM / NPB + MLB"); print("="*72)
        if time.time() - self.started_at >= self.time_budget_sec:
            print("[HARD STOP] computation budget reached before processing")
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
            print("[HARD STOP] computation budget reached; skipping remaining leagues")
        elif mlb:
            try:
                mlb_games = self.load_mlb(mlb_start, mlb_end)
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