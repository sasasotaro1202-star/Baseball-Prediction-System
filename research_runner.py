#!/usr/bin/env python3
"""Reliable research entry point for the Baseball Prediction System.

The existing BaseballBacktest engine remains the source of truth. This runner
adds deterministic orchestration, bounded retries, and an explicit manifest.
It never converts a failed/empty research result into SUCCESS.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from baseball_backtest import BaseballBacktest


def run_one(league: str, data_dir: Path, *, mlb_start: int, mlb_end: int, retries: int = 2) -> dict:
    started = time.time()
    last_error = None
    for attempt in range(retries + 1):
        bt = BaseballBacktest(data_dir)
        result = {
            "league": league,
            "status": "FAILED",
            "predictions": 0,
            "error": None,
            "attempts": attempt + 1,
        }
        try:
            if league == "NPB":
                from data.npb_pbp_adapter import load_public_pbp
                pbp = load_public_pbp(data_dir)
                games = bt.aggregate_npb_games(pbp)
                frame = bt.run_walkforward(games, "NPB")
                result["games"] = int(len(games))
                result["starter_coverage"] = (
                    float(games["confirmed_starters"].mean())
                    if len(games) and "confirmed_starters" in games else 0.0
                )
                result["predictions"] = int(len(frame))
            else:
                games = bt.load_mlb(mlb_start, mlb_end)
                if not games.empty and "confirmed_starters" in games:
                    result["starter_coverage"] = float(games["confirmed_starters"].mean())
                frame = bt.run_walkforward(games, "MLB")
                result["games"] = int(len(games))
                result["predictions"] = int(len(frame))
            result["status"] = "SUCCESS" if result["predictions"] > 0 else "NO_PREDICTIONS"
            result["runtime_seconds"] = round(time.time() - started, 2)
            result["audit_rows"] = int(len(bt.audit))
            return result
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            result["error"] = last_error
            result["runtime_seconds"] = round(time.time() - started, 2)
            result["audit_rows"] = int(len(bt.audit))
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
            else:
                return result
    raise RuntimeError(last_error or "research failed without an error")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--league", choices=["NPB", "MLB", "BOTH"], default="BOTH")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--mlb-start", type=int, default=2020)
    p.add_argument("--mlb-end", type=int, default=2026)
    p.add_argument("--manifest", default="results/research_execution_manifest.json")
    p.add_argument("--retries", type=int, default=2)
    args = p.parse_args()

    if args.retries < 0 or args.retries > 5:
        raise SystemExit("--retries must be between 0 and 5")

    leagues = ["NPB", "MLB"] if args.league == "BOTH" else [args.league]
    results = [
        run_one(x, Path(args.data_dir), mlb_start=args.mlb_start,
                mlb_end=args.mlb_end, retries=args.retries)
        for x in leagues
    ]
    manifest = {
        "version": 2,
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
