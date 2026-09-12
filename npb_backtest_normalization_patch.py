#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inject the missing NPB PBP normalizer into the production backtest.

The loader already calls _normalize_npb_pbp(), but the base class did not
contain that method. This patch is idempotent and fails closed on unexpected
source structure instead of guessing at edits.
"""
from pathlib import Path

P = Path("baseball_backtest.py")
MARKER = "# NPB_PBP_NORMALIZATION_HARDENING_V1"
s = P.read_text(encoding="utf-8")
if MARKER in s and "def _normalize_npb_pbp" in s:
    print("[NPB NORMALIZATION PATCH] V1 already applied")
    raise SystemExit(0)

anchor = "    def aggregate_npb_games(self, pbp: pd.DataFrame) -> pd.DataFrame:\n"
if anchor not in s:
    raise RuntimeError("aggregate_npb_games anchor missing")

method = r'''    def _normalize_npb_pbp(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalize heterogeneous NPB PBP exports before aggregation.

        Naive NPB timestamps are interpreted as Asia/Tokyo first, then converted
        to UTC for stable chronological ordering. Target-game results are not
        used here; this is schema/time normalization only.
        """
        d = raw.copy()
        d.columns = [str(c).strip() for c in d.columns]

        def pick(*names):
            lower = {str(c).strip().lower(): c for c in d.columns}
            for name in names:
                if name.lower() in lower:
                    return lower[name.lower()]
            return None

        gid = pick("game_id", "gameid", "gamepk", "id")
        if gid is None:
            raise ValueError("NPB PBP has no game_id column")
        d["game_id"] = d[gid].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        d = d[d["game_id"].ne("") & d["game_id"].ne("nan")].copy()

        dt_col = pick("datetime", "date", "game_datetime", "game_date", "start_time", "starttime")
        if dt_col is None:
            raise ValueError("NPB PBP has no date/datetime column")
        raw_dt = d[dt_col]
        dt = pd.to_datetime(raw_dt, errors="coerce")
        # Explicit JST handling for historical NPB exports that omit timezone.
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize("Asia/Tokyo", ambiguous="NaT", nonexistent="NaT")
        d["date"] = dt.dt.tz_convert("UTC")

        home_col = pick("home", "home_team", "hometeam", "homeTeam", "team_home")
        away_col = pick("away", "away_team", "awayteam", "awayTeam", "team_away")
        if home_col is None or away_col is None:
            raise ValueError("NPB PBP missing home/away team columns")
        d["home"] = d[home_col].map(lambda x: norm_team(x, "NPB"))
        d["away"] = d[away_col].map(lambda x: norm_team(x, "NPB"))

        def numeric_from(*names):
            col = pick(*names)
            return pd.to_numeric(d[col], errors="coerce") if col else pd.Series(np.nan, index=d.index)

        d["home_score"] = numeric_from("home_score", "homescore", "home_runs", "home_score_total")
        d["away_score"] = numeric_from("away_score", "awayscore", "away_runs", "away_score_total")
        d["row_order"] = numeric_from("row_order", "event_order", "seq", "sequence", "play_index")
        if d["row_order"].isna().all():
            d["row_order"] = d.groupby("game_id", sort=False).cumcount()
        else:
            d["row_order"] = d["row_order"].fillna(d.groupby("game_id", sort=False).cumcount())

        game_type_col = pick("game_type", "gametype", "game_type_name", "type")
        d["game_type"] = d[game_type_col].fillna("").astype(str) if game_type_col else ""

        for side in ("home", "away"):
            col = pick(f"{side}_pitcher", f"{side}_starter", f"{side}_starter_name", f"{side}pitcher")
            d[f"{side}_pitcher"] = d[col].fillna("").astype(str).str.strip() if col else ""

        # Preserve the original columns while guaranteeing the canonical fields
        # consumed by aggregate_npb_games().
        return d.sort_values(["date", "game_id", "row_order"], kind="mergesort").reset_index(drop=True)

'''
s = s.replace(anchor, method + anchor, 1)
s = MARKER + "\n" + s
P.write_text(s, encoding="utf-8")
print("[NPB NORMALIZATION PATCH] V1 applied")
