"""Free/research-only MLB Statcast acquisition and feature staging.

Baseball Savant exposes Statcast Search as CSV. This module intentionally keeps
the source research-only until a historical publication/availability boundary is
proven for PIT. It can still build lagged player/team summaries from the
previously completed games, but the resulting rows carry an explicit
UNVERIFIED PIT status and are never silently promoted into production features.
"""
from __future__ import annotations

from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

ROOT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
TIMEOUT = 60
MAX_DAYS_PER_REQUEST = 7

RAW_COLUMNS = (
    "game_date", "game_pk", "home_team", "away_team", "pitcher", "batter",
    "pitch_type", "release_speed", "release_spin", "pfx_x", "pfx_z",
    "plate_x", "plate_z", "launch_speed", "launch_angle", "hit_distance",
    "description", "events", "bb_type", "launch_speed_angle",
    "home_score", "away_score", "post_home_score", "post_away_score",
    "at_bat_number", "pitch_number",
)


def _url(start: date, end: date) -> str:
    params = {
        "all": "true",
        "hfGT": "R|",
        "hfSea": "",
        "player_type": "pitcher",
        "game_date_gt": start.isoformat(),
        "game_date_lt": end.isoformat(),
        "group_by": "name",
        "sort_col": "pitches",
        "sort_order": "desc",
        "min_pitches": "0",
        "min_results": "0",
        "type": "details",
    }
    return ROOT_URL + "?" + urlencode(params, safe="|")


def fetch_statcast(
    start_date: str | date,
    end_date: str | date,
    *,
    session: requests.Session | None = None,
    retries: int = 3,
) -> pd.DataFrame:
    """Fetch Statcast pitch-level CSV in bounded date chunks."""
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if end < start:
        raise ValueError("end_date must be >= start_date")

    sess = session or requests.Session()
    chunks: list[pd.DataFrame] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=MAX_DAYS_PER_REQUEST - 1), end)
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                resp = sess.get(_url(cursor, chunk_end), timeout=TIMEOUT)
                resp.raise_for_status()
                text = resp.text
                if not text.strip():
                    raise RuntimeError(f"empty Statcast CSV for {cursor}..{chunk_end}")
                frame = pd.read_csv(StringIO(text))
                if "game_pk" not in frame.columns:
                    raise RuntimeError("Statcast response missing game_pk")
                chunks.append(frame)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    continue
        if last_error is not None:
            raise RuntimeError(f"Statcast request failed for {cursor}..{chunk_end}: {last_error}")
        cursor = chunk_end + timedelta(days=1)

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).drop_duplicates(
        subset=[c for c in ("game_pk", "at_bat_number", "pitch_number") if c in chunks[0].columns]
    ).reset_index(drop=True)


def build_game_level_features(pitches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pitch-level Statcast into team/game metrics."""
    if pitches.empty:
        return pd.DataFrame()

    df = pitches.copy()
    for col in (
        "game_pk", "release_speed", "release_spin", "pfx_x", "pfx_z",
        "launch_speed", "launch_angle", "hit_distance",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    rows: list[dict] = []
    for game_pk, g in df.groupby("game_pk", sort=True):
        home = str(g["home_team"].dropna().iloc[0]) if "home_team" in g.columns and g["home_team"].notna().any() else ""
        away = str(g["away_team"].dropna().iloc[0]) if "away_team" in g.columns and g["away_team"].notna().any() else ""
        game_date = pd.to_datetime(g["game_date"], errors="coerce", utc=True).min() if "game_date" in g else pd.NaT
        base = {"game_pk": str(game_pk), "game_date": game_date, "home_team": home, "away_team": away}
        for side, team_col in (("home", home), ("away", away)):
            side_mask = (g["home_team"].eq(team_col) if side == "home" else g["away_team"].eq(team_col))
            s = g.loc[side_mask]
            for col, key in (
                ("release_speed", "pitch_velocity_mean"),
                ("release_spin", "spin_rate_mean"),
                ("pfx_x", "pitch_horizontal_break_mean"),
                ("pfx_z", "pitch_vertical_break_mean"),
            ):
                vals = pd.to_numeric(s.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
                base[f"{side}_{key}"] = float(vals.mean()) if len(vals) else float("nan")
                if len(vals) > 1:
                    base[f"{side}_{key}_std"] = float(vals.std(ddof=1))
                else:
                    base[f"{side}_{key}_std"] = float("nan")

            pitch_types = s.get("pitch_type", pd.Series(dtype="object")).dropna().astype(str)
            if len(pitch_types):
                counts = pitch_types.value_counts(normalize=True)
                base[f"{side}_pitch_type_count"] = float(len(counts))
                probs = counts.to_numpy(dtype=float)
                base[f"{side}_pitch_mix_entropy"] = float(-(probs * __import__("numpy").log(probs)).sum())
                base[f"{side}_pitch_top_type_share"] = float(probs.max())
            else:
                base[f"{side}_pitch_type_count"] = float("nan")
                base[f"{side}_pitch_mix_entropy"] = float("nan")
                base[f"{side}_pitch_top_type_share"] = float("nan")

            ev = pd.to_numeric(s.get("launch_speed", pd.Series(dtype=float)), errors="coerce").dropna()
            la = pd.to_numeric(s.get("launch_angle", pd.Series(dtype=float)), errors="coerce").dropna()
            dist = pd.to_numeric(s.get("hit_distance", pd.Series(dtype=float)), errors="coerce").dropna()
            base[f"{side}_exit_velocity_mean"] = float(ev.mean()) if len(ev) else float("nan")
            base[f"{side}_launch_angle_mean"] = float(la.mean()) if len(la) else float("nan")
            base[f"{side}_hit_distance_mean"] = float(dist.mean()) if len(dist) else float("nan")
            base[f"{side}_hard_hit_rate"] = float((ev >= 95).mean()) if len(ev) else float("nan")
            btype = s.get("launch_speed_angle", pd.Series(dtype=float))
            btype = pd.to_numeric(btype, errors="coerce").dropna()
            base[f"{side}_barrel_rate"] = float((btype == 6).mean()) if len(btype) else float("nan")
            base[f"{side}_pitch_count"] = float(len(s)) if len(s) else float("nan")
            base[f"{side}_launch_speed_n"] = float(len(ev))
            base[f"{side}_launch_angle_n"] = float(len(la))
        rows.append(base)
    return pd.DataFrame(rows).sort_values(["game_date", "game_pk"]).reset_index(drop=True)


def build_lagged_team_features(game_features: pd.DataFrame) -> pd.DataFrame:
    """Shift same-team game metrics so each row reflects only prior games."""
    if game_features.empty:
        return pd.DataFrame()
    g = game_features.copy().sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    metric_cols = [
        c for c in g.columns
        if c.startswith(("home_", "away_"))
        and c not in {"home_team", "away_team"}
        and pd.api.types.is_numeric_dtype(g[c])
    ]
    rows: list[dict] = []
    history: dict[str, list[dict]] = {}
    for _, row in g.iterrows():
        out = {
            "game_pk": str(row["game_pk"]),
            "game_date": row["game_date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "tracking_pit_status": "UNVERIFIED",
            "historical_backtest_eligible": False,
        }
        for side in ("home", "away"):
            team = str(row[f"{side}_team"])
            hist = history.get(team, [])
            for c in metric_cols:
                if not c.startswith(f"{side}_"):
                    continue
                key = c[len(side) + 1:]
                vals = [h[key] for h in hist[-5:] if key in h and pd.notna(h[key])]
                out[f"{side}_lag_{key}"] = float(sum(vals) / len(vals)) if vals else float("nan")
        rows.append(out)
        for side in ("home", "away"):
            team = str(row[f"{side}_team"])
            rec = {
                c[len(side) + 1:]: float(row[c])
                for c in metric_cols if c.startswith(f"{side}_") and pd.notna(row[c])
            }
            history.setdefault(team, []).append(rec)
    return pd.DataFrame(rows)


def stage_research_artifact(
    pitches: pd.DataFrame,
    *,
    output_dir: str | Path,
    source_id: str = "mlb_statcast_baseballsavant",
) -> dict:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    games = build_game_level_features(pitches)
    lagged = build_lagged_team_features(games)
    raw_path = outdir / "statcast_raw.csv"
    games_path = outdir / "statcast_game_features.csv"
    lagged_path = outdir / "statcast_lagged_team_features.csv"
    if not pitches.empty:
        pitches.to_csv(raw_path, index=False)
    games.to_csv(games_path, index=False)
    lagged.to_csv(lagged_path, index=False)
    manifest = {
        "source_id": source_id,
        "source_url": ROOT_URL,
        "rows_pitch": int(len(pitches)),
        "rows_games": int(len(games)),
        "rows_lagged": int(len(lagged)),
        "tracking_pit_status": "UNVERIFIED",
        "historical_backtest_eligible": False,
        "production_eligible": False,
        "reason": "Statcast event timestamps are available, but historical publication/availability timestamps are not proven.",
    }
    pd.DataFrame([manifest]).to_json(outdir / "manifest.json", orient="records", force_ascii=False, indent=2)
    return manifest


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--output-dir", default="data/mlb_statcast")
    args = parser.parse_args()
    days = max(1, min(int(args.days), 14))
    end = date.today()
    start = end - timedelta(days=days - 1)
    pitches = fetch_statcast(start, end)
    manifest = stage_research_artifact(pitches, output_dir=args.output_dir)
    print(pd.Series(manifest).to_json(force_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
