#!/usr/bin/env python3
"""Reliable research entry point for the Baseball Prediction System.

This wrapper deliberately keeps the existing BaseballBacktest engine as the
source of truth while selecting a validated data adapter before execution.
It produces a machine-readable execution manifest so a workflow cannot claim
success when a league actually failed or produced no predictions.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from baseball_backtest import BaseballBacktest


def run_one(league: str, data_dir: Path, *, mlb_start: int, mlb_end: int) -> dict:
    bt = BaseballBacktest(data_dir)
    started = time.time()
    result = {"league": league, "status": "FAILED", "predictions": 0, "error": None}
    try:
        if league == "NPB":
            from data.npb_pbp_adapter import load_public_pbp

            pbp = load_public_pbp(data_dir)
            games = bt.aggregate_npb_games(pbp)
            frame = bt.run_walkforward(games, "NPB")
            result["games"] = int(len(games))
            result["starter_coverage"] = float(games["confirmed_starters"].mean()) if len(games) else 0.0
            result["predictions"] = int(len(frame))
        else:
            games = bt.load_mlb(mlb_start, mlb_end)
            if not games.empty and "confirmed_starters" in games:
                result["starter_coverage"] = float(games["confirmed_starters"].mean())
            frame = bt.run_walkforward(games, "MLB")
            result["games"] = int(len(games))
            result["predictions"] = int(len(frame))
        result["status"] = "SUCCESS" if result["predictions"] > 0 else "NO_PREDICTIONS"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["runtime_seconds"] = round(time.time() - started, 2)
        result["audit_rows"] = int(len(bt.audit))
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--league", choices=["NPB", "MLB", "BOTH"], default="BOTH")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--mlb-start", type=int, default=2020)
    p.add_argument("--mlb-end", type=int, default=2026)
    p.add_argument("--manifest", default="results/research_execution_manifest.json")
    args = p.parse_args()

    leagues = ["NPB", "MLB"] if args.league == "BOTH" else [args.league]
    results = [run_one(x, Path(args.data_dir), mlb_start=args.mlb_start, mlb_end=args.mlb_end) for x in leagues]
    manifest = {
        "version": 1,
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
