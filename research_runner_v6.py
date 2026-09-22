#!/usr/bin/env python3
"""Production research entry point with robust NPB target repair and parallel leagues.

The modeling code remains the repository's existing BaseballBacktest. This wrapper
hardens data acquisition/target integrity, canonicalizes walk-forward outputs, and
removes a major source of CI waste: repeated NPB 404 retries and per-file
official-result reparsing.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_MODULES = (
    "baseball_backtest",
    "data.npb_pbp_adapter",
    "research.validation_pipeline",
    "research.adoption_gate",
    "research.backtest_output_contract",
    "prediction.runner",
    "prediction.prediction_log",
    "evaluation.calibration",
)


def preflight() -> None:
    for name in REQUIRED_MODULES:
        importlib.import_module(name)


def _repair_npb_targets(pbp: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    required = {"game_id", "home_score", "away_score"}
    if not required.issubset(pbp.columns):
        raise RuntimeError("NPB PBP adapter did not expose explicit score columns")
    out = games.copy()
    out["game_id"] = out["game_id"].astype(str)
    out["home_score"] = pd.to_numeric(out["home_score"], errors="coerce")
    out["away_score"] = pd.to_numeric(out["away_score"], errors="coerce")
    score = (
        pbp.assign(game_id=pbp["game_id"].astype(str))
        .assign(
            home_score=pd.to_numeric(pbp["home_score"], errors="coerce"),
            away_score=pd.to_numeric(pbp["away_score"], errors="coerce"),
        )
        .groupby("game_id", as_index=False)
        .agg(home_score=("home_score", "max"), away_score=("away_score", "max"))
    )
    out = out.merge(score, on="game_id", how="left", suffixes=("", "_pbp"), validate="one_to_one")
    out["home_score"] = out["home_score"].fillna(out["home_score_pbp"])
    out["away_score"] = out["away_score"].fillna(out["away_score_pbp"])
    out = out.drop(columns=["home_score_pbp", "away_score_pbp"])
    if out[["home_score", "away_score"]].isna().any().any():
        missing = int(out[["home_score", "away_score"]].isna().any(axis=1).sum())
        raise RuntimeError(f"NPB target repair left missing final scores: {missing} games")
    if (out[["home_score", "away_score"]] < 0).any().any():
        raise RuntimeError("NPB target repair produced negative scores")
    return out


def _validate_targets(frame: pd.DataFrame, league: str) -> dict:
    for col in ("home_score", "away_score"):
        if col not in frame:
            raise RuntimeError(f"{league} missing target column: {col}")
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any():
            raise RuntimeError(f"{league} has non-numeric target values in {col}")
        if (values < 0).any():
            raise RuntimeError(f"{league} has negative target values in {col}")
    if "game_id" in frame and frame["game_id"].duplicated().any():
        raise RuntimeError(f"{league} has duplicate game_id rows: {int(frame['game_id'].duplicated().sum())}")
    hs = pd.to_numeric(frame["home_score"], errors="coerce")
    aw = pd.to_numeric(frame["away_score"], errors="coerce")
    total = hs + aw
    unique_total_runs = int(total.nunique(dropna=True))
    zero_zero_rate = float(((hs == 0) & (aw == 0)).mean())
    mean_total_runs = float(total.mean())
    if unique_total_runs < 5 or mean_total_runs <= 0.25:
        raise RuntimeError(
            f"{league} target integrity failure: score distribution is degenerate "
            f"(unique_total_runs={unique_total_runs}, mean_total_runs={mean_total_runs:.4f})"
        )
    if zero_zero_rate > 0.10:
        raise RuntimeError(f"{league} target integrity failure: zero-zero rate is {zero_zero_rate:.2%}")
    if league == "NPB":
        actual = np.where(hs > aw, 0, np.where(hs == aw, 1, 2))
    else:
        actual = (hs > aw).astype(int).to_numpy()
    counts = pd.Series(actual).value_counts()
    if len(counts) < 2:
        raise RuntimeError(f"{league} target integrity failure: fewer than two result classes were observed")
    return {
        "rows": int(len(frame)),
        "classes": {str(int(k)): int(v) for k, v in counts.sort_index().items()},
        "unique_total_runs": unique_total_runs,
        "mean_total_runs": mean_total_runs,
        "zero_zero_rate": zero_zero_rate,
        "home_score_mean": float(hs.mean()),
        "away_score_mean": float(aw.mean()),
    }


def _filter_confirmed_starters(games: pd.DataFrame, league: str) -> tuple[pd.DataFrame, dict]:
    """Keep only games whose both starting pitchers are known before prediction."""
    if league not in {"NPB", "MLB"}:
        return games, {"before": int(len(games)), "after": int(len(games)), "excluded": 0}
    required = {"home_starter", "away_starter"}
    if not required.issubset(games.columns):
        raise RuntimeError(f"{league} starter availability columns are missing: {sorted(required - set(games.columns))}")
    home = games["home_starter"].fillna("").astype(str).str.strip()
    away = games["away_starter"].fillna("").astype(str).str.strip()
    mask = home.ne("") & away.ne("") & home.str.lower().ne("nan") & away.str.lower().ne("nan")
    before = int(len(games))
    filtered = games.loc[mask].copy().reset_index(drop=True)
    excluded = before - int(len(filtered))
    audit = {"before": before, "after": int(len(filtered)), "excluded": excluded, "coverage": float(len(filtered) / max(before, 1))}
    if len(filtered) <= 0:
        raise RuntimeError(f"{league} has no games with both announced starters")
    return filtered, audit


def _canonicalize_walkforward(bt, league: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Persist and reload the canonical score/Low-High/Top-Draw contract.

    This is intentionally downstream of model inference. It repairs legacy
    reporting fields without changing win/draw/away probabilities.
    """
    if frame.empty:
        return frame
    path = bt.checkpoint_dir / f"{league.lower()}_walkforward.csv"
    if not path.exists():
        raise RuntimeError(f"{league} walk-forward checkpoint missing after successful run: {path}")
    from research.backtest_output_contract import normalize_walkforward
    normalize_walkforward(path, npb=(league == "NPB"))
    canonical = pd.read_csv(path)
    required = {"score1", "score2", "score3", "score4", "score1_prob", "score2_prob", "score3_prob", "score4_prob", "low", "high"}
    if league == "NPB":
        required |= {"top_draw_selection", "top_draw_probability", "top_draw_status", "date_jst"}
    missing = required - set(canonical.columns)
    if missing:
        raise RuntimeError(f"{league} canonical output missing columns: {sorted(missing)}")
    for _, row in canonical.iterrows():
        scores = [str(row[f"score{i}"]) for i in range(1, 5)]
        if len(set(scores)) != 4 or any(s == "その他" for s in scores):
            raise RuntimeError(f"{league} canonical output contains invalid exact-score candidates")
        probs = [float(row[f"score{i}_prob"]) for i in range(1, 5)]
        if probs != sorted(probs, reverse=True):
            raise RuntimeError(f"{league} canonical score probabilities are not descending")
        low, high = float(row["low"]), float(row["high"])
        if not np.isfinite(low + high) or abs(low + high - 1.0) > 1e-8:
            raise RuntimeError(f"{league} canonical Low/High probabilities do not sum to 1")
    if league == "NPB":
        counts = canonical.groupby("date_jst")["top_draw_selection"].sum()
        if not counts.empty and not bool((counts == 1).all()):
            raise RuntimeError("NPB canonical Top Draw must select exactly one game per JST date")
    return canonical


def _is_deterministic_research_error(exc: BaseException) -> bool:
    """Return True when retrying cannot add evidence in the current run.

    Chronological OOS incompleteness, target/PIT contract failures, and output
    contract violations are deterministic for a fixed commit/data snapshot.
    Immediate retries only waste Actions time; durable checkpoints are handed
    to the next scheduled/supervisor cycle instead.
    """
    message = str(exc)
    markers = (
        "walk-forward incomplete",
        "target integrity failure",
        "starter availability columns are missing",
        "starter coverage too low",
        "has no games with both announced starters",
        "produced zero predictions",
        "canonical output",
        "canonical score probabilities",
        "NPB canonical Top Draw",
        "NPB OOS frame has insufficient realized score targets",
    )
    return any(marker in message for marker in markers)


def run_one(league: str, data_dir: Path, *, mlb_start: int, mlb_end: int, retries: int = 2) -> dict:
    started = time.time()
    from baseball_backtest import BaseballBacktest
    last_error = None
    for attempt in range(retries + 1):
        bt = BaseballBacktest(data_dir)
        result = {"league": league, "status": "FAILED", "predictions": 0, "error": None, "attempts": attempt + 1}
        try:
            if league == "NPB":
                from research.npb_official_results import load_repaired_pbp
                pbp = load_repaired_pbp(data_dir)
                games = bt.aggregate_npb_games(pbp)
                if games.empty:
                    raise RuntimeError("NPB adapter produced zero games")
                games = _repair_npb_targets(pbp, games)
                starter_before = int(len(games))
                games, starter_audit = _filter_confirmed_starters(games, league)
                result["starter_filter"] = starter_audit
                target_quality = _validate_targets(games, "NPB")
                frame = bt.run_walkforward(games, "NPB")
                if len(games) >= 200 and len(frame) < max(100, int(len(games) * 0.15)):
                    raise RuntimeError(f"NPB OOS coverage unexpectedly low after starter filter: games={len(games)}, predictions={len(frame)}, raw_games={starter_before}")
            else:
                games = bt.load_mlb(mlb_start, mlb_end)
                if games.empty:
                    raise RuntimeError("MLB loader produced zero games")
                # Historical MLB starter identities are excluded unless explicit
                # announcement-time PIT evidence exists. Research OOS must still
                # be able to run on the full chronological game set; production
                # prediction/adoption separately fails closed on missing starter PIT.
                pit_safe = (
                    "starter_evidence_status" in games.columns
                    and bool((games["starter_evidence_status"] == "pit_safe").all())
                    and bool(games["confirmed_starters"].all())
                )
                if pit_safe:
                    games, starter_audit = _filter_confirmed_starters(games, league)
                else:
                    games = games.copy()
                    starter_audit = {
                        "before": int(len(games)),
                        "after": int(len(games)),
                        "excluded": 0,
                        "coverage": 0.0,
                        "status": "research_without_starter_pit",
                    }
                    for c in ("home_starter", "away_starter"):
                        if c in games.columns:
                            games[c] = ""
                    if "confirmed_starters" in games.columns:
                        games["confirmed_starters"] = False
                result["starter_filter"] = starter_audit
                target_quality = _validate_targets(games, "MLB")
                frame = bt.run_walkforward(games, "MLB")
            frame = _canonicalize_walkforward(bt, league, frame)
            result["games"] = int(len(games))
            result["target_quality"] = target_quality
            result["starter_coverage"] = float(games["confirmed_starters"].mean()) if "confirmed_starters" in games else float(result["starter_filter"]["coverage"])
            result["predictions"] = int(len(frame))
            if result["predictions"] <= 0:
                raise RuntimeError(f"{league} produced zero predictions")
            if "actual_home_score" in frame and "actual_away_score" in frame:
                hs = pd.to_numeric(frame["actual_home_score"], errors="coerce")
                aw = pd.to_numeric(frame["actual_away_score"], errors="coerce")
                minimum = max(100, int(len(frame) * 0.95))
                if hs.notna().sum() < minimum or aw.notna().sum() < minimum:
                    raise RuntimeError(f"{league} OOS frame has insufficient realized score targets")
            result["status"] = "SUCCESS"
            result["runtime_seconds"] = round(time.time() - started, 2)
            result["audit_rows"] = int(len(bt.audit))
            return result
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            result["error"] = last_error
            result["runtime_seconds"] = round(time.time() - started, 2)
            result["audit_rows"] = int(len(bt.audit))
            if _is_deterministic_research_error(exc):
                result["retryable"] = False
                return result
            if attempt < retries:
                result["retryable"] = True
                time.sleep(2 * (attempt + 1))
            else:
                result["retryable"] = True
                return result
    raise RuntimeError(last_error or "research failed")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--league", choices=["NPB", "MLB", "BOTH"], default="BOTH")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--mlb-start", type=int, default=2020)
    p.add_argument("--mlb-end", type=int, default=2026)
    p.add_argument("--manifest", default="results/research_execution_manifest.json")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    if args.retries < 0 or args.retries > 5:
        raise SystemExit("--retries must be between 0 and 5")
    preflight()
    if args.preflight_only:
        print(json.dumps({"status": "PREFLIGHT_OK", "modules": REQUIRED_MODULES}, ensure_ascii=False))
        return 0
    leagues = ["NPB", "MLB"] if args.league == "BOTH" else [args.league]
    with ThreadPoolExecutor(max_workers=len(leagues)) as ex:
        futures = [ex.submit(run_one, league, Path(args.data_dir), mlb_start=args.mlb_start, mlb_end=args.mlb_end, retries=args.retries) for league in leagues]
        results = [f.result() for f in futures]
    manifest = {
        "version": 8,
        "requested_leagues": leagues,
        "results": results,
        "overall_status": "SUCCESS" if all(r["status"] == "SUCCESS" for r in results) else "PARTIAL_OR_FAILED",
    }
    path = Path(args.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["overall_status"] == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
