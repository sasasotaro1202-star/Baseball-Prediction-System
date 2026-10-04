"""Research-only NPB state-transition + Monte Carlo game-script engine.

Historical NPB PBP is used only for research screening because the source does
not prove historical publication/availability timestamps. Results are always
PIT-UNVERIFIABLE and production-ineligible until a separate PIT evidence layer
proves target-time availability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

SCHEMA_VERSION = "game-state-transition-v2"
PIT_STATUS = "UNVERIFIABLE_HISTORICAL_PBP_AVAILABILITY"
MAX_SCORE_DIFF = 8
MAX_RUNS_PER_TRANSITION = 4


def _first(df: pd.DataFrame, names: Sequence[str], default=np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([default] * len(df), index=df.index)


def _half(x: Any) -> str:
    s = str(x).strip().upper()
    return "T" if s in {"T", "TOP", "表"} else "B" if s in {"B", "BOT", "BOTTOM", "裏"} else ""


def _base_mask(row: Mapping[str, Any]) -> int:
    mask = 0
    for bit, names in ((1, ("on_1b", "base1")), (2, ("on_2b", "base2")), (4, ("on_3b", "base3"))):
        value = next((row[n] for n in names if n in row), None)
        if value is None or pd.isna(value):
            continue
        text = str(value).strip().lower()
        if text in {"", "nan", "none", "nat", "0", "0.0", "false"}:
            continue
        try:
            if float(text) == 0.0:
                continue
        except (TypeError, ValueError):
            pass
        mask |= bit
    return mask


def _state_key(inning: int, half: str, outs: int, bases: int, diff: int) -> tuple[int, str, int, int, int]:
    return (max(1, min(12, int(inning))), str(half), max(0, min(2, int(outs))), int(bases) & 7, max(-MAX_SCORE_DIFF, min(MAX_SCORE_DIFF, int(diff))))


def canonicalize_pbp_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize public NPB PBP without filling missing state with zeros."""
    if raw.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=raw.index)
    out["game_id"] = _first(raw, ["game_id", "GameID"], "").astype("string").fillna("").str.strip()
    out["inning"] = pd.to_numeric(_first(raw, ["inning", "Inning"]), errors="coerce")
    out["half"] = _first(raw, ["TB", "half", "Half"]).map(_half)
    out["play_order"] = pd.to_numeric(_first(raw, ["PlayInfo_SeqNo", "play_id", "ID", "page"]), errors="coerce")
    out["pitch_number"] = pd.to_numeric(_first(raw, ["pitch_number", "atBatBallCount"]), errors="coerce")
    out["page"] = _first(raw, ["page", "fiveDigitSerialNumber"], "").astype("string").fillna("").str.strip()
    out["game_date"] = pd.to_datetime(_first(raw, ["game_date", "GameDate"]), errors="coerce", utc=True)
    out["home"] = _first(raw, ["home_team_name", "H_NameS"], "").astype("string").fillna("").str.strip()
    out["away"] = _first(raw, ["away_team_name", "V_NameS"], "").astype("string").fillna("").str.strip()
    out["home_score"] = pd.to_numeric(_first(raw, ["home_total_runs", "H_R"]), errors="coerce")
    out["away_score"] = pd.to_numeric(_first(raw, ["away_total_runs", "V_R"]), errors="coerce")
    out["outs"] = pd.to_numeric(_first(raw, ["outs_when_up", "out"]), errors="coerce")
    for src, dst in (("on_1b", "base1"), ("on_2b", "base2"), ("on_3b", "base3")):
        out[dst] = raw[src] if src in raw.columns else (_first(raw, [dst]))

    required = ["game_id", "inning", "half", "play_order", "game_date", "home_score", "away_score", "outs"]
    out = out[out["game_id"].ne("") & out["home"].ne("") & out["away"].ne("")]
    out = out[out["half"].isin({"T", "B"})]
    for col in required[1:]:
        out = out[out[col].notna()]
    out["inning"] = out["inning"].round().astype(int)
    out["outs"] = out["outs"].round().astype(int)
    out = out[out["inning"].between(1, 12) & out["outs"].between(0, 2)]
    out = out[(out["home_score"] >= 0) & (out["away_score"] >= 0)]
    out["_half_order"] = out["half"].map({"T": 0, "B": 1}).astype(int)
    out = out.sort_values(["game_id", "inning", "_half_order", "play_order", "pitch_number"], kind="mergesort")
    out["_identity"] = np.where(
        out["page"].str.strip().ne(""), out["game_id"] + "|" + out["page"],
        out["game_id"] + "|" + out["inning"].astype(str) + "|" + out["half"] + "|" + out["play_order"].astype(str),
    )
    out = out.drop_duplicates("_identity", keep="last").drop(columns=["_identity", "_half_order"])
    return out.reset_index(drop=True)


def _transition(a: Mapping[str, Any], b: Mapping[str, Any]) -> tuple[str, int, int, int, str] | None:
    ai, bi = int(a["inning"]), int(b["inning"])
    ah, bh = str(a["half"]), str(b["half"])
    valid = (ai == bi and ah == bh) or (ai == bi and ah == "T" and bh == "B") or (ah == "B" and bh == "T" and bi == ai + 1)
    if not valid:
        return None
    ao, bo = int(a["outs"]), int(b["outs"])
    if ai == bi and ah == bh and bo < ao:
        return None
    dh = float(b["home_score"]) - float(a["home_score"])
    da = float(b["away_score"]) - float(a["away_score"])
    if dh < 0 or da < 0 or dh > MAX_RUNS_PER_TRANSITION or da > MAX_RUNS_PER_TRANSITION or (dh > 0 and da > 0):
        return None
    runs = int(round(dh + da))
    scorer = "H" if dh > 0 else "A" if da > 0 else "N"
    return bh, max(0, min(2, bo)), _base_mask(b), runs, scorer


def _kernel_fingerprint(exact: Mapping[tuple, Counter]) -> str:
    data = [[str(k), sorted((str(o), int(n)) for o, n in v.items())] for k, v in sorted(exact.items(), key=lambda x: str(x[0]))]
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


class TransitionKernel:
    def __init__(self, exact: Mapping[tuple, Counter], by_state: Mapping[tuple, Counter], by_half: Mapping[str, Counter], transitions: int):
        self.exact = {k: Counter(v) for k, v in exact.items()}
        self.by_state = {k: Counter(v) for k, v in by_state.items()}
        self.by_half = {k: Counter(v) for k, v in by_half.items()}
        self.transitions = int(transitions)
        self.fingerprint = _kernel_fingerprint(self.exact)

    def _lookup(self, key: tuple[int, str, int, int, int]) -> Counter | None:
        inning, half, outs, bases, _ = key
        for candidate in (self.exact.get(key), self.by_state.get((inning, half, outs, bases)), self.by_state.get((12, half, outs, bases)), self.by_half.get(half)):
            if candidate:
                return candidate
        return None

    def sample(self, key: tuple[int, str, int, int, int], rng: np.random.Generator, scoring_factors: Mapping[str, float]) -> tuple[str, int, int, int, str] | None:
        counts = self._lookup(key)
        if not counts:
            return None
        weights = []
        for outcome, count in counts.items():
            scorer = outcome[-1]
            runs = int(outcome[-2])
            factor = float(np.clip(scoring_factors.get(scorer, 1.0), 0.55, 1.45))
            weight = float(count) * (factor ** runs if scorer in {"H", "A"} and runs else 1.0)
            weights.append(weight)
        p = np.asarray(weights, dtype=float)
        if not np.isfinite(p).all() or p.sum() <= 0:
            return None
        p /= p.sum()
        return list(counts.keys())[int(rng.choice(len(counts), p=p))]


def fit_transition_kernel(pbp: pd.DataFrame, *, min_transitions: int = 100) -> TransitionKernel:
    frame = canonicalize_pbp_frame(pbp)
    exact: dict[tuple, Counter] = defaultdict(Counter)
    by_state: dict[tuple, Counter] = defaultdict(Counter)
    by_half: dict[str, Counter] = defaultdict(Counter)
    n = 0
    for _, game in frame.groupby("game_id", sort=False):
        rows = game.to_dict("records")
        for a, b in zip(rows, rows[1:]):
            t = _transition(a, b)
            if t is None:
                continue
            nh, no, nb, runs, scorer = t
            key = _state_key(int(a["inning"]), str(a["half"]), int(a["outs"]), _base_mask(a), int(round(float(a["home_score"]) - float(a["away_score"]))))
            outcome = (nh, no, nb, runs, scorer)
            exact[key][outcome] += 1
            by_state[(key[0], key[1], key[2], key[3])][outcome] += 1
            by_half[key[1]][outcome] += 1
            n += 1
    if n < int(min_transitions):
        raise ValueError(f"insufficient valid transitions: {n} < {min_transitions}")
    return TransitionKernel(exact, by_state, by_half, n)


@dataclass
class RollingFactors:
    base_run: float
    shrink_games: float = 20.0

    def __post_init__(self) -> None:
        self.gf: Counter[str] = Counter()
        self.ga: Counter[str] = Counter()
        self.ng: Counter[str] = Counter()
        self.total_runs = 0.0
        self.total_games = 0

    def before(self, home: str, away: str) -> tuple[float, float, float, float]:
        league = max(0.1, self.total_runs / (2 * self.total_games)) if self.total_games else max(0.1, self.base_run)
        def one(team: str) -> tuple[float, float]:
            n = float(self.ng[team])
            off = (self.gf[team] + self.shrink_games * league) / (n + self.shrink_games)
            deff = (self.ga[team] + self.shrink_games * league) / (n + self.shrink_games)
            return float(np.clip(off / league, 0.55, 1.55)), float(np.clip(deff / league, 0.55, 1.55))
        ho, hd = one(home); ao, ad = one(away)
        return ho, hd, ao, ad

    def update(self, home: str, away: str, hs: float, aw: float) -> None:
        hs, aw = float(hs), float(aw)
        self.gf[home] += hs; self.ga[home] += aw; self.ng[home] += 1
        self.gf[away] += aw; self.ga[away] += hs; self.ng[away] += 1
        self.total_runs += hs + aw; self.total_games += 1


def poisson_matrix(home_lambda: float, away_lambda: float, max_runs: int = 14) -> np.ndarray:
    grid = np.arange(max_runs + 1)
    def pmf(lam: float) -> np.ndarray:
        x = np.exp(-lam) * np.array([lam ** int(k) / math.factorial(int(k)) for k in grid], dtype=float)
        return x / x.sum()
    return np.outer(pmf(max(0.1, home_lambda)), pmf(max(0.1, away_lambda)))


def _summary(matrix: np.ndarray, hs: int, aw: int) -> dict[str, float]:
    ph = float(np.tril(matrix, -1).sum())
    pa = float(np.triu(matrix, 1).sum())
    pd = float(np.trace(matrix))
    flat = matrix.ravel()
    top4 = np.argsort(-flat, kind="mergesort")[:4]
    actual = "H" if hs > aw else "A" if aw > hs else "D"
    p_actual = {"H": ph, "A": pa, "D": pd}[actual]
    pred = "H" if ph >= max(pa, pd) else "A" if pa >= pd else "D"
    top = int(top4[0]); phs, pas = divmod(top, matrix.shape[1])
    return {
        "accuracy": float(pred == actual),
        "logloss": float(-math.log(max(1e-12, p_actual))),
        "brier": float((ph - float(actual == "H")) ** 2 + (pd - float(actual == "D")) ** 2 + (pa - float(actual == "A")) ** 2),
        "score_mae": float((abs(phs - hs) + abs(pas - aw)) / 2),
        "top1_exact": float(phs == hs and pas == aw),
        "top4_exact": float(any(divmod(int(i), matrix.shape[1]) == (hs, aw) for i in top4)),
        "confidence": float(max(ph, pd, pa)),
        "correct": float(pred == actual),
    }


def _aggregate(rows: pd.DataFrame) -> dict[str, float | int]:
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (rows["confidence"] >= lo) & ((rows["confidence"] < hi) if hi < 1 else (rows["confidence"] <= hi))
        if m.any():
            ece += float(m.mean()) * abs(float(rows.loc[m, "correct"].mean()) - float(rows.loc[m, "confidence"].mean()))
    return {
        "rows": int(len(rows)), "accuracy": float(rows.accuracy.mean()), "logloss": float(rows.logloss.mean()), "brier": float(rows.brier.mean()),
        "ece": float(ece), "score_mae": float(rows.score_mae.mean()), "top1_exact": float(rows.top1_exact.mean()), "top4_exact": float(rows.top4_exact.mean()),
    }


def simulate_game(kernel: TransitionKernel, *, base_run: float, home_factor: float, away_factor: float, simulations: int = 500, seed: int = 42, max_innings: int = 12) -> dict[str, Any]:
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    rng = np.random.default_rng(int(seed))
    scores: list[tuple[int, int]] = []
    extras = lead_change_games = comebacks = 0
    first_lead: Counter[int] = Counter()
    sf = {"H": float(np.clip(home_factor ** 0.55, 0.55, 1.45)), "A": float(np.clip(away_factor ** 0.55, 0.55, 1.45))}
    for _ in range(int(simulations)):
        inning, half, outs, bases, hs, aw = 1, "T", 0, 0, 0, 0
        ever_home_trailing = ever_away_trailing = False
        saw_change = False; first_change: int | None = None; aborted = False
        while inning <= max_innings:
            if hs < aw: ever_home_trailing = True
            if aw < hs: ever_away_trailing = True
            if half == "B" and inning >= 9 and hs > aw: break
            t = kernel.sample(_state_key(inning, half, outs, bases, hs - aw), rng, scoring_factors=sf)
            if t is None: aborted = True; break
            nh, no, nb, runs, scorer = t
            prev = hs - aw
            if scorer == "H": hs += runs
            elif scorer == "A": aw += runs
            new = hs - aw
            if prev * new < 0:
                saw_change = True
                first_change = first_change if first_change is not None else inning
            outs, bases = no, nb
            inning, half = (inning + 1 if half == "B" and nh == "T" else inning), nh
        if aborted: continue
        scores.append((hs, aw))
        extras += int(inning > 9)
        lead_change_games += int(saw_change)
        if saw_change and first_change is not None: first_lead[first_change] += 1
        comebacks += int((hs > aw and ever_home_trailing) or (aw > hs and ever_away_trailing))
    valid = len(scores)
    if valid < max(1, math.ceil(simulations * 0.995)):
        raise RuntimeError(f"simulation coverage below fail-closed threshold: {valid}/{simulations}")
    matrix = np.zeros((15, 15), dtype=float)
    for hs, aw in scores: matrix[min(14, hs), min(14, aw)] += 1
    matrix /= matrix.sum()
    flat = matrix.ravel(); top4 = np.argsort(-flat, kind="mergesort")[:4]
    home_win, away_win, draw = float(np.tril(matrix, -1).sum()), float(np.triu(matrix, 1).sum()), float(np.trace(matrix))
    low = float(sum(matrix[i, j] for i in range(15) for j in range(15) if i + j <= 6))
    return {
        "schema_version": SCHEMA_VERSION, "simulations": int(simulations), "valid_simulations": int(valid), "seed": int(seed),
        "kernel_fingerprint": kernel.fingerprint, "kernel_transitions": kernel.transitions,
        "home_factor": float(home_factor), "away_factor": float(away_factor), "base_run": float(base_run),
        "score_distribution": matrix.tolist(), "probabilities": {"home_win": home_win, "draw": draw, "away_win": away_win, "low_le_6": low, "high_ge_7": 1 - low},
        "top4_exact_score": [{"score": f"{int(i // 15)}-{int(i % 15)}", "probability": float(flat[i])} for i in top4],
        "extra_inning_probability": float(extras / valid), "lead_change_probability": float(lead_change_games / valid),
        "comeback_probability": float(comebacks / valid), "first_lead_change_inning_distribution": {str(k): float(v / valid) for k, v in sorted(first_lead.items())},
    }


def evaluate_from_files(
    paths: Sequence[str | Path], *, development_end: str = "2024-12-31", validation_start: str = "2025-01-01", validation_end: str = "2025-12-31", holdout_start: str = "2026-01-01", max_validation_games: int = 120, max_holdout_games: int = 120, simulations: int = 500, seed: int = 42,
) -> dict[str, Any]:
    files = [Path(p) for p in paths]
    exact: dict[tuple, Counter] = defaultdict(Counter); by_state: dict[tuple, Counter] = defaultdict(Counter); by_half: dict[str, Counter] = defaultdict(Counter)
    transitions = 0; games: list[dict[str, Any]] = []; dev_end = pd.Timestamp(development_end, tz="UTC")
    for path in files:
        raw = pd.read_csv(path, low_memory=False); frame = canonicalize_pbp_frame(raw); del raw
        if frame.empty: continue
        for gid, g in frame.groupby("game_id", sort=False):
            g = g.copy(); g["_half_order"] = g.half.map({"T": 0, "B": 1})
            last = g.sort_values(["inning", "_half_order", "play_order", "pitch_number"], kind="mergesort").iloc[-1]
            games.append({"game_id": str(gid), "game_date": last.game_date, "home": str(last.home), "away": str(last.away), "home_score": float(last.home_score), "away_score": float(last.away_score), "source_file": path.name})
        dev = frame[frame.game_date <= dev_end]
        for _, g in dev.groupby("game_id", sort=False):
            rows = g.to_dict("records")
            for a, b in zip(rows, rows[1:]):
                t = _transition(a, b)
                if t is None: continue
                nh, no, nb, runs, scorer = t
                key = _state_key(int(a["inning"]), str(a["half"]), int(a["outs"]), _base_mask(a), int(round(float(a["home_score"]) - float(a["away_score"]))))
                outcome = (nh, no, nb, runs, scorer); exact[key][outcome] += 1; by_state[(key[0], key[1], key[2], key[3])][outcome] += 1; by_half[key[1]][outcome] += 1; transitions += 1
    if transitions < 100: raise ValueError(f"insufficient valid transitions: {transitions} < 100")
    kernel = TransitionKernel(exact, by_state, by_half, transitions)
    games_df = pd.DataFrame(games).drop_duplicates("game_id").sort_values(["game_date", "game_id"]).reset_index(drop=True)
    if games_df.empty: raise ValueError("no games available")
    vs, ve, hs = pd.Timestamp(validation_start, tz="UTC"), pd.Timestamp(validation_end, tz="UTC"), pd.Timestamp(holdout_start, tz="UTC")
    val = games_df[(games_df.game_date >= vs) & (games_df.game_date <= ve)].head(max_validation_games)
    hold = games_df[games_df.game_date >= hs].head(max_holdout_games)
    if val.empty or hold.empty: raise ValueError("validation or holdout game set is empty")
    dev_games = games_df[games_df.game_date <= dev_end].sort_values(["game_date", "game_id"])
    base_run = float(dev_games[["home_score", "away_score"]].stack().mean()); roll = RollingFactors(base_run=base_run)
    for r in dev_games.itertuples(index=False): roll.update(r.home, r.away, r.home_score, r.away_score)
    selected = pd.concat([val, hold], ignore_index=True).sort_values(["game_date", "game_id"])
    cand_rows: list[dict[str, Any]] = []; base_rows: list[dict[str, Any]] = []
    for r in selected.itertuples(index=False):
        ho, hd, ao, ad = roll.before(r.home, r.away)
        hf = float(np.clip(math.sqrt(ho * ad), 0.60, 1.55)); af = float(np.clip(math.sqrt(ao * hd), 0.60, 1.55))
        gid_seed = int.from_bytes(hashlib.sha256(str(r.game_id).encode()).digest()[:4], "big")
        sim = simulate_game(kernel, base_run=roll.base_run if roll.base_run > 0 else 4.0, home_factor=hf, away_factor=af, simulations=simulations, seed=seed ^ gid_seed)
        actual_h, actual_a = int(r.home_score), int(r.away_score)
        phase = "validation" if r.game_date <= ve else "holdout"
        cand_rows.append({"game_id": r.game_id, "phase": phase, **_summary(np.asarray(sim["score_distribution"], dtype=float), actual_h, actual_a)})
        base_rows.append({"game_id": r.game_id, "phase": phase, **_summary(poisson_matrix(base_run * hf, base_run * af), actual_h, actual_a)})
        roll.update(r.home, r.away, r.home_score, r.away_score)
    cand_df, base_df = pd.DataFrame(cand_rows), pd.DataFrame(base_rows)
    aggregate = {"candidate": {}, "poisson_baseline": {}}
    for phase in ("validation", "holdout"):
        aggregate["candidate"][phase] = _aggregate(cand_df[cand_df.phase == phase])
        aggregate["poisson_baseline"][phase] = _aggregate(base_df[base_df.phase == phase])
    delta = {}
    for phase in ("validation", "holdout"):
        b, c = aggregate["poisson_baseline"][phase], aggregate["candidate"][phase]
        delta[phase] = {"logloss_relative_improvement": float((b["logloss"] - c["logloss"]) / max(1e-12, b["logloss"])), "accuracy_delta": float(c["accuracy"] - b["accuracy"]), "brier_delta": float(c["brier"] - b["brier"]), "ece_delta": float(c["ece"] - b["ece"])}
    return {
        "schema_version": SCHEMA_VERSION, "status": "RESEARCH_SCREENING_ONLY", "pit_status": PIT_STATUS, "production_eligible": False, "decision": "HOLD_RESEARCH_ONLY",
        "reason": "Historical PBP publication/availability timestamps are not proven; this result cannot enter production OOS or promotion evidence.",
        "development_end": development_end, "validation_start": validation_start, "validation_end": validation_end, "holdout_start": holdout_start,
        "input_files": [str(p) for p in files], "input_file_count": len(files), "kernel": {"transitions": kernel.transitions, "fingerprint": kernel.fingerprint},
        "aggregate": aggregate, "delta_vs_poisson": delta, "evaluation_rows": {"candidate": cand_rows, "baseline": base_rows},
        "reproducibility": {"seed": int(seed), "simulations": int(simulations), "chronological_order": "game_date_then_game_id", "holdout_tuning": "FORBIDDEN", "kernel_training_boundary": development_end},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(); p.add_argument("--data-glob", default="data/*_pbp.csv"); p.add_argument("--development-end", default="2024-12-31"); p.add_argument("--validation-start", default="2025-01-01"); p.add_argument("--validation-end", default="2025-12-31"); p.add_argument("--holdout-start", default="2026-01-01"); p.add_argument("--max-validation-games", type=int, default=120); p.add_argument("--max-holdout-games", type=int, default=120); p.add_argument("--simulations", type=int, default=500); p.add_argument("--seed", type=int, default=42); p.add_argument("--output", default="results/game_state_screening.json")
    a = p.parse_args(argv); files = sorted(Path().glob(a.data_glob));
    if not files: raise SystemExit(f"no PBP files matched: {a.data_glob}")
    result = evaluate_from_files(files, development_end=a.development_end, validation_start=a.validation_start, validation_end=a.validation_end, holdout_start=a.holdout_start, max_validation_games=max(1,a.max_validation_games), max_holdout_games=max(1,a.max_holdout_games), simulations=max(1,a.simulations), seed=a.seed)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "
", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "pit_status", "production_eligible", "decision", "aggregate", "delta_vs_poisson")}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
