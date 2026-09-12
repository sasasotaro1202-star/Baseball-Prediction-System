#!/usr/bin/env python3
"""Data collection pipeline for MLB/NPB (self-contained, no core.config dependency)"""
import json
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"


def _load_season_range():
    """config.json があれば読む。無ければ既定値 2024-2026。"""
    cfg_path = ROOT / "config.json"
    start, end = 2024, 2026
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            mlb_cfg = cfg.get("mlb", {})
            start = int(mlb_cfg.get("start_season", start))
            end = int(mlb_cfg.get("end_season", end))
        except Exception as e:
            print(f"config.json parse failed, using defaults: {e}")
    return start, end


def collect_mlb():
    """Collect MLB game data from StatsAPI. Never raises - always returns exit code 0
    so a single blocked/failed season does not kill the whole pipeline."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    start, end = _load_season_range()
    total_games = 0

    for season in range(start, end + 1):
        url = (
            "https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&startDate={season}-01-01&endDate={season}-12-31&gameType=R"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
        except Exception as e:
            print(f"Season {season}: fetch failed ({e}), skipping")
            continue

        rows = []
        for d in data.get("dates", []):
            for g in d.get("games", []):
                home = g.get("teams", {}).get("home", {})
                away = g.get("teams", {}).get("away", {})
                hs = home.get("score")
                aws = away.get("score")
                status = g.get("status", {}).get("detailedState")
                if status == "Final" and hs is not None and aws is not None:
                    rows.append({
                        "date": d.get("date"),
                        "home_team": home.get("team", {}).get("name"),
                        "away_team": away.get("team", {}).get("name"),
                        "home_score": hs,
                        "away_score": aws,
                        "league": "MLB",
                    })

        if not rows:
            print(f"Season {season}: 0 completed games found")
            continue

        df = pd.DataFrame(rows)
        out_path = RAW_DIR / f"mlb_games_{season}.csv"
        df.to_csv(out_path, index=False)
        total_games += len(df)
        print(f"Season {season}: {len(df)} games -> {out_path}")

    print(f"Total MLB games collected: {total_games}")
    return total_games


if __name__ == "__main__":
    collect_mlb()
