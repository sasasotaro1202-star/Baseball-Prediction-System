"""NPB (Nippon Professional Baseball) historical data acquisition.

Source: armstjc/Nippon-Baseball-Data-Repository (MIT licensed, public GitHub repo)
https://github.com/armstjc/Nippon-Baseball-Data-Repository

Coverage: 2018-2025 NPB schedules (8 seasons).

Verified column mapping (confirmed against real 2024 data on 2026-09-09):
  gamedate               -> date (first 10 chars: 'YYYY-MM-DD')
  hometeamnameen (or hometeamshortname if blank) -> home_team
  awayteamnameen (or awayteamshortname if blank) -> away_team
  homescore              -> home_score
  awayscore              -> away_score

NOTE (2026-09-09): source CSV headers may use mixed/PascalCase (e.g. 'GameDate').
All column names are lowercased immediately after fetch so matching is
case-insensitive regardless of how the upstream repo formats headers.

Rows with gamestate != 2 (not "finished") or missing scores are dropped,
since those represent postponed/incomplete games and would corrupt the
Elo/form calculations if treated as 0-0 results.
"""
import argparse
import io
import os
import sys

import pandas as pd
import requests

REPO_RAW_BASE = "https://raw.githubusercontent.com/armstjc/Nippon-Baseball-Data-Repository/main/schedules"
SEASONS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]


def fetch_season(year: int, timeout: int = 20):
    url = f"{REPO_RAW_BASE}/{year}_npb_schedule.csv"
    resp = requests.get(url, timeout=timeout)
    if resp.status_code != 200:
        print(f"[warn] {year}: HTTP {resp.status_code}")
        return None
    df = pd.read_csv(io.StringIO(resp.content.decode("utf-8", errors="ignore")))
    return df


def normalize(df: pd.DataFrame, year: int):
    df = df.copy()
    original_columns = df.columns.tolist()
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = ["gamedate", "homescore", "awayscore"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[warn] {year}: expected columns missing {missing}. "
              f"Actual columns (original case): {original_columns}. Saving RAW instead.")
        return df, False

    out = pd.DataFrame()
    out["date"] = df["gamedate"].astype(str).str[:10]
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if "hometeamnameen" in df.columns and "hometeamshortname" in df.columns:
        out["home_team"] = df["hometeamnameen"].fillna(df["hometeamshortname"])
    elif "hometeamnameen" in df.columns:
        out["home_team"] = df["hometeamnameen"]
    else:
        out["home_team"] = df.get("hometeamshortname")

    if "awayteamnameen" in df.columns and "awayteamshortname" in df.columns:
        out["away_team"] = df["awayteamnameen"].fillna(df["awayteamshortname"])
    elif "awayteamnameen" in df.columns:
        out["away_team"] = df["awayteamnameen"]
    else:
        out["away_team"] = df.get("awayteamshortname")

    out["home_score"] = pd.to_numeric(df["homescore"], errors="coerce")
    out["away_score"] = pd.to_numeric(df["awayscore"], errors="coerce")
    out["league"] = "NPB"

    if "gamestate" in df.columns:
        out = out[df["gamestate"] == 2]

    before = len(out)
    out = out.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    dropped = before - len(out)
    if dropped:
        print(f"[{year}] dropped {dropped} incomplete/postponed rows")

    return out, True


def acquire(out_dir: str = "data/npb"):
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for year in SEASONS:
        df = fetch_season(year)
        if df is None:
            report[year] = "fetch_failed"
            continue
        normalized, ok = normalize(df, year)
        suffix = "" if ok else "_RAW_UNNORMALIZED"
        out_path = os.path.join(out_dir, f"npb_games_{year}{suffix}.csv")
        normalized.to_csv(out_path, index=False)
        report[year] = "ok" if ok else "saved_raw_needs_manual_mapping"
        print(f"[{year}] status={report[year]} rows={len(normalized)}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/npb")
    args = parser.parse_args()
    report = acquire(args.out_dir)
    import json
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/npb_acquisition_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
