"""Adapters for public NPB PBP releases used by the Baseball research runner.

The domain engine remains unchanged. This module converts the public
Nippon-Baseball-Data-Repository PBP schema into the small, explicit contract
needed by ``BaseballBacktest.aggregate_npb_games``. It never invents starter
identities: when a reliable pitcher identity cannot be inferred from the PBP
ordering, the starter field remains blank and production eligibility can reject
that game.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _first_existing(df: pd.DataFrame, names: Iterable[str], default=None):
    for name in names:
        if name in df.columns:
            return df[name]
    if default is None:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.Series([default] * len(df), index=df.index)


def normalize_pbp_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize the public NPB PBP schema without using post-game targets as features."""
    if raw.empty:
        return pd.DataFrame(columns=[
            "game_id", "row_order", "date", "home", "away", "home_score",
            "away_score", "game_type", "home_pitcher", "away_pitcher",
        ])

    out = pd.DataFrame(index=raw.index)
    out["game_id"] = _first_existing(raw, ["game_id", "GameID"]).astype(str)
    out["row_order"] = pd.to_numeric(
        _first_existing(raw, ["PlayInfo_SeqNo", "play_id", "ID", "page"], 0),
        errors="coerce",
    ).fillna(0)
    out["date"] = pd.to_datetime(
        _first_existing(raw, ["game_date", "GameDate"]), errors="coerce", utc=True
    )
    out["home"] = _first_existing(raw, ["home_team_name", "H_NameS"]).astype(str)
    out["away"] = _first_existing(raw, ["away_team_name", "V_NameS"]).astype(str)
    out["home_score"] = pd.to_numeric(
        _first_existing(raw, ["home_total_runs", "H_R"]), errors="coerce"
    )
    out["away_score"] = pd.to_numeric(
        _first_existing(raw, ["away_total_runs", "V_R"]), errors="coerce"
    )
    out["game_type"] = _first_existing(raw, ["game_type_name", "GameKindName"], "").astype(str)

    # First pitcher observed in each half is a conservative starter proxy.
    # If half information is absent, leave both starters unknown rather than
    # silently assigning a potentially incorrect reliever as a starter.
    pitcher = _first_existing(raw, ["pitcher"]).astype(str)
    half = _first_existing(raw, ["TB", "top_bottom", "half"], "").astype(str).str.lower()
    out["_pitcher"] = pitcher.where(~pitcher.isin(["", "nan", "None"]), np.nan)
    out["_half"] = half

    def first_half(g: pd.DataFrame, tokens: tuple[str, ...]) -> str:
        for _, r in g.sort_values("row_order").iterrows():
            p = r.get("_pitcher")
            h = str(r.get("_half", ""))
            if not p or p != p:
                continue
            if any(t in h for t in tokens):
                return str(p)
        return ""

    starters = []
    for gid, g in out.groupby("game_id", sort=False):
        home_pitcher = first_half(g, ("top", "表", "visitor", "away", "v"))
        away_pitcher = first_half(g, ("bottom", "裏", "home", "h"))
        starters.append((gid, home_pitcher, away_pitcher))
    starter_df = pd.DataFrame(starters, columns=["game_id", "home_pitcher", "away_pitcher"])
    out = out.merge(starter_df, on="game_id", how="left")
    out = out.drop(columns=["_pitcher", "_half"])

    out = out.dropna(subset=["game_id", "date"]).reset_index(drop=True)
    return out


def load_public_pbp(data_dir: str | Path) -> pd.DataFrame:
    """Load all ``*_pbp.csv`` files staged in ``data_dir``."""
    root = Path(data_dir)
    files = sorted(root.glob("*_pbp.csv"))
    if not files:
        raise FileNotFoundError(
            "No NPB PBP files found. Stage public release assets such as "
            "2025-04_pbp.csv in the data directory first."
        )
    frames = []
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        normalized = normalize_pbp_frame(frame)
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        raise RuntimeError("NPB PBP files were found but none contained usable game rows.")
    return pd.concat(frames, ignore_index=True, sort=False)
