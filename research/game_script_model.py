"""PIT-aware NPB Game Script research challenger.

This module predicts a baseball game as a distribution over state transitions:
plate appearance -> base/out/score state -> next plate appearance -> terminal
game state. It is research-only and intentionally does not alter production.

The runner is chronological: a target game is predicted before its PBP rows are
allowed to update the transition learner. Same-start-time games are frozen as a
group. Historical source availability timestamps are not present in the public
PBP release, so this module reports availability evidence as UNKNOWN and is not
eligible for production promotion by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

try:
    from research.competition_taxonomy import classify_npb
except Exception:  # pragma: no cover - compatibility fallback
    classify_npb = None

SCHEMA_VERSION = "game-script-research-v1"
RANDOM_SEED = 42817
DEFAULT_MAX_SIMS = 2500
DEFAULT_RECENT_GAMES = 900
DEFAULT_MIN_TRAIN_GAMES = 150
DEFAULT_CHECKPOINT_EVERY = 50
DEFAULT_TIME_BUDGET_SEC = 2400
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "game_script"
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"
RUNS_DIR = RESULTS_DIR / "runs"

HALVES = ("T", "B")
OUTCOME_CLASSES = (0, 1, 2)  # HOME / DRAW / AWAY


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_seed(value: str, base: int = RANDOM_SEED) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") ^ int(base)


def _canonical_team(value: Any) -> str:
    s = "" if value is None else str(value).strip()
    aliases = {
        "巨人": "読売ジャイアンツ",
        "読売": "読売ジャイアンツ",
        "阪神": "阪神タイガース",
        "中日": "中日ドラゴンズ",
        "広島": "広島東洋カープ",
        "ヤクルト": "東京ヤクルトスワローズ",
        "横浜": "横浜DeNAベイスターズ",
        "DeNA": "横浜DeNAベイスターズ",
        "ＤｅＮＡ": "横浜DeNAベイスターズ",
        "ソフトバンク": "福岡ソフトバンクホークス",
        "西武": "埼玉西武ライオンズ",
        "日本ハム": "北海道日本ハムファイターズ",
        "日ハム": "北海道日本ハムファイターズ",
        "ロッテ": "千葉ロッテマリーンズ",
        "楽天": "東北楽天ゴールデンイーグルス",
        "オリックス": "オリックス・バファローズ",
    }
    return aliases.get(s, s)


def _first_existing(frame: pd.DataFrame, names: Iterable[str], default: Any = np.nan) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def _base_bits(row: Mapping[str, Any]) -> int:
    bits = 0
    for bit, names in ((1, ("on_1b", "on_1b_name", "on1b")), (2, ("on_2b", "on_2b_name", "on2b")), (4, ("on_3b", "on_3b_name", "on3b"))):
        value = ""
        for name in names:
            if name in row:
                value = row[name]
                break
        if value is None:
            continue
        text = str(value).strip().lower()
        if text not in {"", "nan", "none", "nat", "0"}:
            bits |= bit
    return bits


def _half(value: Any) -> str:
    s = "" if value is None else str(value).strip().upper()
    if s in {"T", "TOP", "表", "AWAY", "先攻"}:
        return "T"
    if s in {"B", "BOT", "BOTTOM", "裏", "HOME", "後攻"}:
        return "B"
    return ""


def _inning(value: Any) -> int:
    try:
        n = int(float(value))
        return max(1, n)
    except Exception:
        return 1


def _outs(value: Any) -> int:
    try:
        n = int(float(value))
        return min(2, max(0, n))
    except Exception:
        return 0


def _score_bucket(score_diff: float) -> int:
    return int(np.clip(int(round(score_diff)), -5, 5))


def _inning_bucket(inning: int) -> int:
    return min(max(int(inning), 1), 10)


def _score_from_row(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    h = next((row[c] for c in ("home_total_runs", "H_R", "home_score") if c in row), np.nan)
    a = next((row[c] for c in ("away_total_runs", "V_R", "away_score") if c in row), np.nan)
    try:
        hf, af = float(h), float(a)
    except Exception:
        return None, None
    if not math.isfinite(hf) or not math.isfinite(af):
        return None, None
    return hf, af


def _game_time(frame: pd.DataFrame) -> pd.Timestamp:
    values = _first_existing(frame, ("game_date", "GameDate", "datetime", "date"))
    return pd.to_datetime(values, errors="coerce", utc=True).dropna().iloc[0]


def _row_order(frame: pd.DataFrame) -> pd.Series:
    values = _first_existing(frame, ("PlayInfo_SeqNo", "play_id", "ID", "row_order", "page"), 0)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(np.arange(len(frame), dtype=float))
    return pd.Series(np.arange(len(frame), dtype=float), index=frame.index)


def discover_pbp_files(data_dir: str | Path, years: set[int] | None = None) -> list[Path]:
    root = Path(data_dir)
    aggregate = root / "npb_multi_source_games_all.csv"
    season_files = sorted((root / "npb_games").glob("*_multi_source_pbp.csv")) if (root / "npb_games").exists() else []
    if aggregate.exists():
        files = [aggregate]
    elif season_files:
        files = season_files
    else:
        files = sorted(root.glob("*_multi_source_pbp.csv")) + sorted(root.glob("*_pbp.csv"))
        files = [f for f in dict.fromkeys(files) if f.name != "2026_multi_source_pbp.csv"]
    if years is None:
        return files
    selected = []
    for path in files:
        stem_year = path.name[:4]
        try:
            if int(stem_year) in years:
                selected.append(path)
        except ValueError:
            selected.append(path)
    return selected


def load_raw_pbp(data_dir: str | Path, years: set[int] | None = None) -> pd.DataFrame:
    files = discover_pbp_files(data_dir, years=years)
    if not files:
        raise FileNotFoundError("No NPB PBP source files found.")
    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        frame.columns = [str(c).strip() for c in frame.columns]
        if "game_id" not in frame.columns and "GameID" not in frame.columns:
            continue
        frames.append(frame)
    if not frames:
        raise RuntimeError("NPB PBP files were present but contained no game_id/GameID field.")
    raw = pd.concat(frames, ignore_index=True, sort=False)
    gid = _first_existing(raw, ("game_id", "GameID")).astype(str).str.strip()
    raw["__game_id"] = gid
    raw["__game_time"] = pd.to_datetime(
        _first_existing(raw, ("game_date", "GameDate", "datetime", "date")),
        errors="coerce",
        utc=True,
    )
    raw["__order"] = _row_order(raw)
    return raw.dropna(subset=["__game_id", "__game_time"]).copy()


def _phase_known(game_type: Any) -> bool:
    text = "" if game_type is None else str(game_type).strip()
    lowered = text.lower()
    excluded = ("オープン戦", "オールスター", "ファーム", "二軍", "教育", "open", "spring", "allstar", "farm")
    if any(x in text for x in excluded) or any(x in lowered for x in ("open", "spring", "allstar", "farm")):
        return False
    if classify_npb is not None:
        try:
            label = classify_npb(text)
            status = str(getattr(label, "status", "")).upper()
            if status and status not in {"KNOWN", "OK", "VALID"}:
                return False
            game_class = str(getattr(label, "game_class", "")).lower()
            if game_class and any(x in game_class for x in ("exhibition", "farm", "allstar")):
                return False
        except Exception:
            pass
    return ("公式戦" in text) or ("交流戦" in text) or text == ""


def build_game_index(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gid, group in raw.groupby("__game_id", sort=False):
        g = group.sort_values("__order")
        first = g.iloc[0]
        home = _canonical_team(first.get("home_team_name", first.get("H_NameS", "")))
        away = _canonical_team(first.get("away_team_name", first.get("V_NameS", "")))
        game_type = str(first.get("game_type_name", first.get("GameKindName", "")) or "")
        if not home or not away or home == away or not _phase_known(game_type):
            continue
        rows.append({
            "game_id": str(gid),
            "game_time": g["__game_time"].iloc[0],
            "home": home,
            "away": away,
            "game_type": game_type,
        })
    if not rows:
        raise RuntimeError("No valid NPB official-season games were found in the PBP input.")
    return pd.DataFrame(rows).sort_values(["game_time", "game_id"]).drop_duplicates("game_id").reset_index(drop=True)


def _terminal_row_score(game: pd.DataFrame) -> tuple[float, float]:
    ordered = game.sort_values("__order")
    for _, row in ordered.iloc[::-1].iterrows():
        h, a = _score_from_row(row)
        if h is not None and a is not None:
            return h, a
    raise ValueError("final score is unavailable for game")


def _pa_starts(game: pd.DataFrame) -> pd.DataFrame:
    g = game.sort_values("__order").copy()
    page_col = next((c for c in ("page",) if c in g.columns), None)
    if page_col is not None:
        page = g[page_col].astype(str)
    else:
        inning_col = _first_existing(g, ("inning", "Inning"), 1).astype(str)
        half_col = _first_existing(g, ("TB", "half", "Half"), "").astype(str)
        ab_col = _first_existing(g, ("inning_ab_num", "TextInfo_Bat_No"), "").astype(str)
        page = inning_col + "|" + half_col + "|" + ab_col
    g["__pa"] = page
    starts = g.drop_duplicates("__pa", keep="first").copy()
    return starts.sort_values("__order").reset_index(drop=True)


def _state_from_row(row: Mapping[str, Any]) -> tuple[int, str, int, int, int]:
    inning = _inning(row.get("inning", row.get("Inning", 1)))
    half = _half(row.get("TB", row.get("half", row.get("Half", ""))))
    if half == "":
        half = "T"
    outs = _outs(row.get("outs_when_up", row.get("outs", 0)))
    base = _base_bits(row)
    home, away = _score_from_row(row)
    diff = (home - away) if home is not None and away is not None else 0.0
    return _inning_bucket(inning), half, outs, base, _score_bucket(diff)


def _next_score_run(current: Mapping[str, Any], nxt: Mapping[str, Any], offense_half: str) -> int:
    ch, ca = _score_from_row(current)
    nh, na = _score_from_row(nxt)
    if None in (ch, ca, nh, na):
        return 0
    if offense_half == "B":
        return int(max(0.0, round(nh - ch)))
    return int(max(0.0, round(na - ca)))


@dataclass
class TransitionStore:
    exact: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    coarse: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    league_exact: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    league_coarse: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    global_counter: Counter[str] = field(default_factory=Counter)
    transitions: int = 0

    @staticmethod
    def _state_key(team: str, state: tuple[int, str, int, int, int]) -> str:
        inn, half, outs, base, score = state
        return f"{team}|{inn}|{half}|{outs}|{base}|{score}"

    @staticmethod
    def _coarse_key(state: tuple[int, str, int, int, int]) -> str:
        inn, half, outs, base, _score = state
        return f"{inn}|{half}|{outs}|{base}"

    @staticmethod
    def _encode(next_inning: int, next_half: str, next_outs: int, next_base: int, runs: int, terminal: bool) -> str:
        return f"{int(next_inning)}|{next_half}|{int(next_outs)}|{int(next_base)}|{int(np.clip(runs,0,8))}|{1 if terminal else 0}"

    @staticmethod
    def _decode(value: str) -> tuple[int, str, int, int, int, bool]:
        inn, half, outs, base, runs, terminal = value.split("|")
        return int(inn), half, int(outs), int(base), int(runs), bool(int(terminal))

    def add(self, team: str, state: tuple[int, str, int, int, int], next_state: tuple[int, str, int, int, int], runs: int, terminal: bool) -> None:
        exact_key = self._state_key(team, state)
        coarse_key = self._coarse_key(state)
        encoded = self._encode(next_state[0], next_state[1], next_state[2], next_state[3], runs, terminal)
        self.exact[exact_key][encoded] += 1
        self.coarse[f"{team}|{coarse_key}"][encoded] += 1
        self.league_exact["|".join(map(str, state))][encoded] += 1
        self.league_coarse[coarse_key][encoded] += 1
        self.global_counter[encoded] += 1
        self.transitions += 1

    def _weighted_distribution(self, counters: list[tuple[Counter[str], float]]) -> tuple[list[str], np.ndarray, int]:
        merged: Counter[str] = Counter()
        total_weight = 0.0
        observed = 0
        for counter, weight in counters:
            if not counter:
                continue
            total = sum(counter.values())
            if total <= 0:
                continue
            observed = max(observed, int(total))
            for key, count in counter.items():
                merged[key] += float(weight) * float(count) / float(total)
            total_weight += float(weight)
        if total_weight <= 0 or not merged:
            return [], np.empty(0), 0
        keys = list(merged)
        probs = np.asarray([merged[k] for k in keys], dtype=float)
        probs /= probs.sum()
        return keys, probs, observed

    def distribution(self, team: str, state: tuple[int, str, int, int, int]) -> tuple[list[str], np.ndarray, int]:
        exact_key = self._state_key(team, state)
        coarse_key = self._coarse_key(state)
        keys, probs, observed = self._weighted_distribution([
            (self.exact.get(exact_key, Counter()), 0.52),
            (self.coarse.get(f"{team}|{coarse_key}", Counter()), 0.24),
            (self.league_exact.get("|".join(map(str, state)), Counter()), 0.14),
            (self.league_coarse.get(coarse_key, Counter()), 0.07),
            (self.global_counter, 0.03),
        ])
        return keys, probs, observed

    def update_game(self, game: pd.DataFrame, home: str, away: str) -> int:
        starts = _pa_starts(game)
        if len(starts) < 1:
            return 0
        added = 0
        for idx in range(len(starts)):
            current = starts.iloc[idx]
            state = _state_from_row(current)
            offense = away if state[1] == "T" else home
            if idx + 1 < len(starts):
                nxt = starts.iloc[idx + 1]
                nstate = _state_from_row(nxt)
                runs = _next_score_run(current, nxt, state[1])
                self.add(offense, state, nstate, runs, terminal=False)
                added += 1
            else:
                # Terminal transition is learned from the last PA of every game;
                # the terminal label itself is not consulted until after the game
                # has been predicted.
                h, a = _terminal_row_score(game)
                diff = h - a
                nstate = (10, "E", 0, 0, _score_bucket(diff))
                runs = _next_score_run(current, {"home_score": h, "away_score": a}, state[1])
                self.add(offense, state, nstate, runs, terminal=True)
                added += 1
        return added


@dataclass
class TeamRunStats:
    scored: float = 0.0
    allowed: float = 0.0
    games: int = 0

    def update(self, scored: float, allowed: float) -> None:
        self.scored += float(scored)
        self.allowed += float(allowed)
        self.games += 1

    def means(self) -> tuple[float, float]:
        if self.games <= 0:
            return 0.0, 0.0
        return self.scored / self.games, self.allowed / self.games


@dataclass
class BaselineState:
    teams: dict[str, TeamRunStats] = field(default_factory=lambda: defaultdict(TeamRunStats))
    league_scored: float = 0.0
    league_games: int = 0

    def update(self, home: str, away: str, home_runs: float, away_runs: float) -> None:
        self.teams[home].update(home_runs, away_runs)
        self.teams[away].update(away_runs, home_runs)
        self.league_scored += home_runs + away_runs
        self.league_games += 1

    def lambdas(self, home: str, away: str) -> tuple[float, float]:
        if self.league_games <= 0:
            return 2.35, 2.35
        league_mean = max(0.2, self.league_scored / (2.0 * self.league_games))
        hs, ha = self.teams[home].means()
        aw, aa = self.teams[away].means()
        if self.teams[home].games < 8:
            hs = ha = league_mean
        if self.teams[away].games < 8:
            aw = aa = league_mean
        home_lam = max(0.05, 0.60 * hs + 0.40 * aa)
        away_lam = max(0.05, 0.60 * aw + 0.40 * ha)
        shrink = min(1.0, self.league_games / 120.0)
        home_lam = shrink * home_lam + (1 - shrink) * league_mean
        away_lam = shrink * away_lam + (1 - shrink) * league_mean
        return min(home_lam, 12.0), min(away_lam, 12.0)


def _poisson_pmf(k: int, lam: float) -> float:
    lam = max(float(lam), 1e-6)
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def _poisson_probs(lam_h: float, lam_a: float, max_runs: int = 13) -> np.ndarray:
    ph = np.asarray([_poisson_pmf(k, lam_h) for k in range(max_runs + 1)])
    pa = np.asarray([_poisson_pmf(k, lam_a) for k in range(max_runs + 1)])
    m = np.outer(ph, pa)
    return m / m.sum()


def _summary_from_score_matrix(matrix: np.ndarray) -> dict[str, Any]:
    probs = np.asarray(matrix, dtype=float)
    home_p = float(np.tril(probs, -1).sum())  # rows=home, cols=away
    draw_p = float(np.trace(probs))
    away_p = float(np.triu(probs, 1).sum())
    total_mass = float(probs.sum())
    if total_mass <= 0 or not np.isfinite(total_mass):
        raise ValueError("invalid score distribution")
    probs = probs / total_mass
    flat = []
    for h in range(probs.shape[0]):
        for a in range(probs.shape[1]):
            flat.append((float(probs[h, a]), h, a))
    flat.sort(reverse=True)
    top4 = [
        {"score": f"{h}-{a}", "probability": round(float(p), 8)}
        for p, h, a in flat[:4]
    ]
    total_idx = []
    for h in range(probs.shape[0]):
        for a in range(probs.shape[1]):
            total_idx.append((h + a, h, a, float(probs[h, a])))
    low = sum(p for total, _h, _a, p in total_idx if total <= 6)
    high = sum(p for total, _h, _a, p in total_idx if total >= 7)
    return {
        "home_probability": home_p,
        "draw_probability": draw_p,
        "away_probability": away_p,
        "low_probability": float(low),
        "high_probability": float(high),
        "expected_home_runs": float(sum(h * probs[h, a] for h in range(probs.shape[0]) for a in range(probs.shape[1]))),
        "expected_away_runs": float(sum(a * probs[h, a] for h in range(probs.shape[0]) for a in range(probs.shape[1]))),
        "top4": top4,
    }


def simulate_game(
    store: TransitionStore,
    home: str,
    away: str,
    *,
    n_sims: int,
    seed: int,
    collect_script: bool = False,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    outcome_counts = np.zeros(3, dtype=int)
    score_counts: Counter[tuple[int, int]] = Counter()
    inning_sums: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    snapshots: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    fallback = 0
    total_steps = 0
    for _ in range(int(n_sims)):
        state = (1, "T", 0, 0, 0)
        home_score = 0
        away_score = 0
        current_half = "T"
        inning = 1
        inning_half_runs = {1: [0.0, 0.0]}
        ended = False
        for step in range(150):
            if state[1] == "E":
                ended = True
                break
            current_half = state[1]
            offense = away if current_half == "T" else home
            keys, probs, observed = store.distribution(offense, state)
            if not keys:
                fallback += 1
                # Safe degradation: force a terminal, low-information branch.
                state = (10, "E", 0, 0, 0)
                ended = True
                break
            if observed < 5:
                fallback += 1
            choice = str(rng.choice(keys, p=probs))
            next_inning, next_half, next_outs, next_base, runs, terminal = store._decode(choice)
            runs = min(max(int(runs), 0), 8)
            if current_half == "T":
                away_score += runs
                inning_half_runs.setdefault(inning, [0.0, 0.0])[1] += runs
            else:
                home_score += runs
                inning_half_runs.setdefault(inning, [0.0, 0.0])[0] += runs
            total_steps += 1
            state = (next_inning, next_half, next_outs, next_base, _score_bucket(home_score - away_score))
            if terminal or next_half == "E":
                ended = True
                break
            if next_inning != inning or next_half != current_half:
                inning = max(1, next_inning)
        if not ended:
            # Explicit compute safety cap. This is not treated as a normal terminal
            # path in the evidence report; it increments truncation/fallback.
            fallback += 1
        if home_score > away_score:
            outcome_counts[0] += 1
        elif home_score == away_score:
            outcome_counts[1] += 1
        else:
            outcome_counts[2] += 1
        score_counts[(home_score, away_score)] += 1
        for inn, (hr, ar) in inning_half_runs.items():
            inning_sums[inn][0] += hr
            inning_sums[inn][1] += ar
            if inn in (3, 5, 7, 9):
                snapshots[inn][0] += float(home_score > away_score)  # coarse lead-state proxy
                snapshots[inn][1] += float(home_score == away_score)
                snapshots[inn][2] += float(home_score < away_score)
    p = outcome_counts / max(1, int(n_sims))
    matrix = np.zeros((14, 14), dtype=float)
    for (h, a), count in score_counts.items():
        matrix[min(13, h), min(13, a)] += float(count)
    matrix /= max(1, int(n_sims))
    summary = _summary_from_score_matrix(matrix)
    summary.update({
        "simulation_count": int(n_sims),
        "fallback_transition_rate": float(fallback / max(1, total_steps + fallback)),
        "mean_transition_steps": float(total_steps / max(1, int(n_sims))),
        "outcome_entropy": float(-np.sum(np.clip(p, 1e-12, 1.0) * np.log(np.clip(p, 1e-12, 1.0)))),
        "outcome_probability": p.tolist(),
    })
    if collect_script:
        summary["inning_expected_runs"] = {
            str(inn): {
                "home": float(v[0] / max(1, int(n_sims))),
                "away": float(v[1] / max(1, int(n_sims))),
            }
            for inn, v in sorted(inning_sums.items())
            if inn <= 12
        }
    return summary


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    if len(y) == 0:
        return float("nan")
    confidence = np.max(p, axis=1)
    predicted = np.argmax(p, axis=1)
    correct = (predicted == y).astype(float)
    out = 0.0
    n = len(y)
    for lo in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        hi = lo + 1.0 / bins
        mask = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        if not mask.any():
            continue
        out += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return out


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y.astype(int)] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1))) if len(y) else float("nan")


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    rows = np.arange(len(y))
    chosen = np.clip(p[rows, y.astype(int)], 1e-12, 1.0)
    return float(-np.mean(np.log(chosen))) if len(y) else float("nan")


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    if len(y) == 0:
        return {"rows": 0}
    pred = np.argmax(p, axis=1)
    return {
        "rows": int(len(y)),
        "accuracy": float(np.mean(pred == y)),
        "logloss": _logloss(y, p),
        "brier": _brier(y, p),
        "ece": _ece(y, p),
    }


def _score_metrics(actual_h: np.ndarray, actual_a: np.ndarray, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if len(summaries) == 0:
        return {"rows": 0}
    expected_h = np.asarray([float(s["expected_home_runs"]) for s in summaries])
    expected_a = np.asarray([float(s["expected_away_runs"]) for s in summaries])
    mae = 0.5 * (np.abs(expected_h - actual_h) + np.abs(expected_a - actual_a))
    top4_hits = []
    low_high_correct = []
    low_high_logloss = []
    for h, a, s in zip(actual_h, actual_a, summaries):
        exact = f"{int(h)}-{int(a)}"
        labels = {item["score"] for item in s.get("top4", [])}
        top4_hits.append(exact in labels and int(h) + int(a) < 7)
        actual_high = int(h) + int(a) >= 7
        hp = float(np.clip(s["high_probability"], 1e-9, 1 - 1e-9))
        low_high_correct.append((hp >= 0.5) == bool(actual_high))
        low_high_logloss.append(-(math.log(hp) if actual_high else math.log(1 - hp)))
    return {
        "rows": int(len(summaries)),
        "score_mae": float(np.mean(mae)),
        "top4_exact_score_hit_rate": float(np.mean(top4_hits)),
        "low_high_accuracy": float(np.mean(low_high_correct)),
        "low_high_logloss": float(np.mean(low_high_logloss)),
    }


def _hash_inputs(files: list[Path], *, config: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for path in sorted(files):
        stat = path.stat()
        h.update(str(path).encode())
        h.update(str(stat.st_size).encode())
        # Runner mtimes are not stable across executions; never use them in the
        # experiment fingerprint. Content samples + size provide a deterministic
        # cache identity without hashing every byte of multi-GB inputs.
        # Full file content hashing is expensive for very large PBP releases.
        # Hash the first/last 1 MiB plus metadata; the exact run fingerprint also
        # records the file list and size. This is a cache identity, not a data
        # integrity substitute.
        with path.open("rb") as fh:
            first = fh.read(1024 * 1024)
            if stat.st_size > 1024 * 1024:
                fh.seek(max(0, stat.st_size - 1024 * 1024))
            last = fh.read(1024 * 1024)
        h.update(first)
        h.update(last)
    h.update(json.dumps(dict(config), sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_checkpoint(fingerprint: str) -> dict[str, Any] | None:
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        payload = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("fingerprint") != fingerprint:
        return None
    if payload.get("schema_version") != SCHEMA_VERSION:
        return None
    return payload


def _actual_game(game: pd.DataFrame) -> tuple[float, float, int]:
    h, a = _terminal_row_score(game)
    target = 0 if h > a else 1 if h == a else 2
    return float(h), float(a), target


def run_game_script_cycle(
    *,
    data_dir: str | Path = "data",
    output_dir: str | Path | None = None,
    mode: str = "recent",
    max_sims: int = DEFAULT_MAX_SIMS,
    recent_games: int = DEFAULT_RECENT_GAMES,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    time_budget_sec: int = DEFAULT_TIME_BUDGET_SEC,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    started = time.monotonic()
    output_root = Path(output_dir) if output_dir else RESULTS_DIR
    output_root.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_pbp(data_dir)
    # Group indices once. Re-scanning the entire raw frame for every target game
    # turns a linear chronological replay into an accidental O(G*N) operation.
    group_indices = raw.groupby("__game_id", sort=False).indices
    index = build_game_index(raw)
    if mode == "recent":
        eval_count = max(1, int(recent_games))
        selected_ids = set(index.tail(eval_count)["game_id"].astype(str))
        index = index.loc[index["game_id"].astype(str).isin(selected_ids)].reset_index(drop=True)
    elif mode != "deep":
        raise ValueError("mode must be recent or deep")

    file_paths = discover_pbp_files(data_dir)
    fingerprint = _hash_inputs(
        file_paths,
        config={
            "schema_version": SCHEMA_VERSION,
            "mode": mode,
            "recent_games": recent_games,
            "min_train_games": min_train_games,
            "max_sims": max_sims,
            "seed": seed,
        },
    )
    checkpoint = _load_checkpoint(fingerprint)
    start_idx = 0
    if checkpoint is not None:
        start_idx = int(checkpoint.get("next_index", 0))
    if start_idx >= len(index):
        start_idx = 0
        checkpoint = None

    store = TransitionStore()
    baseline = BaselineState()
    predictions: list[dict[str, Any]] = list(checkpoint.get("predictions", [])) if checkpoint else []
    actual_home: list[float] = [float(p["actual"]["home_score"]) for p in predictions if "actual" in p]
    actual_away: list[float] = [float(p["actual"]["away_score"]) for p in predictions if "actual" in p]
    actual_target: list[int] = [int(p["actual"]["outcome_class"]) for p in predictions if "actual" in p]

    def game_frame(game_id: str) -> pd.DataFrame:
        positions = group_indices.get(str(game_id))
        if positions is None:
            return pd.DataFrame()
        return raw.iloc[positions].sort_values("__order").copy()

    # Rebuild state up to the checkpoint boundary without evaluating those games.
    for idx in range(start_idx):
        row = index.iloc[idx]
        g = game_frame(str(row["game_id"]))
        if len(g) == 0:
            continue
        if idx >= min_train_games:
            # Prior evaluated games are also legitimate training games after they
            # finish; no prediction from these rows is reused here.
            pass
        store.update_game(g, str(row["home"]), str(row["away"]))
        try:
            h, a = _terminal_row_score(g)
            baseline.update(str(row["home"]), str(row["away"]), h, a)
        except Exception:
            continue

    # For fresh runs, enforce a warm-up before producing candidate evidence.
    if start_idx < min_train_games:
        for idx in range(start_idx, min_train_games):
            row = index.iloc[idx]
            g = game_frame(str(row["game_id"]))
            store.update_game(g, str(row["home"]), str(row["away"]))
            try:
                h, a = _terminal_row_score(g)
                baseline.update(str(row["home"]), str(row["away"]), h, a)
            except Exception:
                pass
        start_idx = min_train_games

    # Same-start-time games are predicted from an identical state snapshot.
    while start_idx < len(index):
        if time.monotonic() - started >= max(60, int(time_budget_sec)):
            _atomic_json(CHECKPOINT_PATH, {
                "schema_version": SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "next_index": start_idx,
                "predictions": predictions,
                "status": "CHECKPOINTED_TIME_BUDGET",
                "updated_at_utc": _utc_now(),
            })
            raise TimeoutError(f"game-script cycle checkpointed at index={start_idx}")

        current_time = index.iloc[start_idx]["game_time"]
        same_time_indices: list[int] = []
        j = start_idx
        while j < len(index) and index.iloc[j]["game_time"] == current_time:
            same_time_indices.append(j)
            j += 1

        pending: list[tuple[int, pd.DataFrame, dict[str, Any]]] = []
        for idx in same_time_indices:
            row = index.iloc[idx]
            g = game_frame(str(row["game_id"]))
            home = str(row["home"])
            away = str(row["away"])
            baseline_lh, baseline_la = baseline.lambdas(home, away)
            base_summary = _summary_from_score_matrix(_poisson_probs(baseline_lh, baseline_la))
            summary = simulate_game(
                store,
                home,
                away,
                n_sims=max(250, int(max_sims)),
                seed=_stable_seed(str(row["game_id"]), seed),
                collect_script=(idx == same_time_indices[0]),
            )
            pending.append((idx, g, {
                "game_id": str(row["game_id"]),
                "game_time": pd.Timestamp(row["game_time"]).isoformat(),
                "home": home,
                "away": away,
                "candidate": summary,
                "research_baseline": base_summary,
                "pit": {
                    "temporal_state_separation": "PASS",
                    "target_game_rows_used_before_prediction": False,
                    "source_available_at_evidence": "UNKNOWN",
                    "overall_pit_status": "UNVERIFIABLE",
                },
            }))

        # Outcomes are read only after every game at this timestamp has received a
        # prediction, preserving same-time PIT isolation.
        for idx, g, payload in pending:
            h, a, target = _actual_game(g)
            payload["actual"] = {"home_score": h, "away_score": a, "outcome_class": int(target)}
            predictions.append(payload)
            actual_home.append(h)
            actual_away.append(a)
            actual_target.append(target)

        # Only after target-time predictions/labels are finalized do games enter
        # the learner. Updating the baseline after the entire timestamp group
        # prevents same-start-time cross-game leakage.
        for idx, g, payload in pending:
            row = index.iloc[idx]
            store.update_game(g, str(row["home"]), str(row["away"]))
            h, a = float(payload["actual"]["home_score"]), float(payload["actual"]["away_score"])
            baseline.update(str(row["home"]), str(row["away"]), h, a)

        start_idx = j
        if checkpoint_every > 0 and start_idx % checkpoint_every < len(same_time_indices):
            _atomic_json(CHECKPOINT_PATH, {
                "schema_version": SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "next_index": start_idx,
                "predictions": predictions,
                "status": "RUNNING_CHECKPOINT",
                "updated_at_utc": _utc_now(),
            })

    cand_p = np.asarray([p["candidate"]["outcome_probability"] for p in predictions], dtype=float)
    base_p = np.asarray([
        [
            p["research_baseline"]["home_probability"],
            p["research_baseline"]["draw_probability"],
            p["research_baseline"]["away_probability"],
        ]
        for p in predictions
    ], dtype=float)
    y = np.asarray(actual_target, dtype=int)
    candidate_metrics = _metrics(y, cand_p)
    baseline_metrics = _metrics(y, base_p)
    candidate_score = _score_metrics(np.asarray(actual_home), np.asarray(actual_away), [p["candidate"] for p in predictions])
    baseline_score = _score_metrics(np.asarray(actual_home), np.asarray(actual_away), [p["research_baseline"] for p in predictions])
    delta = {
        "logloss_candidate_minus_baseline": float(candidate_metrics["logloss"] - baseline_metrics["logloss"]),
        "brier_candidate_minus_baseline": float(candidate_metrics["brier"] - baseline_metrics["brier"]),
        "accuracy_candidate_minus_baseline": float(candidate_metrics["accuracy"] - baseline_metrics["accuracy"]),
        "score_mae_candidate_minus_baseline": float(candidate_score["score_mae"] - baseline_score["score_mae"]),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "EXECUTED",
        "evidence_status": "RESEARCH_ONLY_HOLD",
        "decision": "HOLD_RESEARCH_ONLY",
        "pit_status": "UNVERIFIABLE",
        "production_eligible": False,
        "mode": mode,
        "fingerprint": fingerprint,
        "seed": int(seed),
        "max_sims": int(max_sims),
        "rows": int(len(predictions)),
        "evaluation_period": {
            "start": index.iloc[min_train_games]["game_time"].isoformat() if len(index) > min_train_games else None,
            "end": index.iloc[-1]["game_time"].isoformat() if len(index) else None,
        },
        "candidate": {
            "outcome": candidate_metrics,
            "score": candidate_score,
        },
        "research_baseline": {
            "outcome": baseline_metrics,
            "score": baseline_score,
        },
        "delta_candidate_minus_baseline": delta,
        "uncertainty": {
            "mean_outcome_entropy": float(np.mean([p["candidate"]["outcome_entropy"] for p in predictions])) if predictions else float("nan"),
            "mean_fallback_transition_rate": float(np.mean([p["candidate"]["fallback_transition_rate"] for p in predictions])) if predictions else float("nan"),
        },
        "artifacts": {
            "prediction_rows": int(len(predictions)),
            "checkpoint_path": str(CHECKPOINT_PATH),
        },
        "generated_at_utc": _utc_now(),
        "runtime_seconds": float(time.monotonic() - started),
    }
    run_name = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{fingerprint[:10]}"
    run_path = RUNS_DIR / f"{run_name}.json"
    _atomic_json(run_path, {"report": report, "predictions": predictions})
    _atomic_json(output_root / "latest.json", report)
    CHECKPOINT_PATH.unlink(missing_ok=True)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--mode", choices=("recent", "deep"), default="recent")
    parser.add_argument("--max-sims", type=int, default=DEFAULT_MAX_SIMS)
    parser.add_argument("--recent-games", type=int, default=DEFAULT_RECENT_GAMES)
    parser.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--time-budget-sec", type=int, default=DEFAULT_TIME_BUDGET_SEC)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = run_game_script_cycle(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            max_sims=args.max_sims,
            recent_games=args.recent_games,
            min_train_games=args.min_train_games,
            checkpoint_every=args.checkpoint_every,
            time_budget_sec=args.time_budget_sec,
            seed=args.seed,
        )
    except TimeoutError as exc:
        print(json.dumps({
            "status": "CHECKPOINTED",
            "reason": str(exc),
            "checkpoint": str(CHECKPOINT_PATH),
        }, ensure_ascii=False))
        return 2
    print(json.dumps({
        "status": report.get("status"),
        "evidence_status": report.get("evidence_status"),
        "decision": report.get("decision"),
        "rows": report.get("rows"),
        "candidate_logloss": report.get("candidate", {}).get("outcome", {}).get("logloss"),
        "baseline_logloss": report.get("research_baseline", {}).get("outcome", {}).get("logloss"),
        "candidate_score_mae": report.get("candidate", {}).get("score", {}).get("score_mae"),
        "pit_status": report.get("pit_status"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
