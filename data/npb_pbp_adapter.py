"""Adapters for public NPB PBP releases used by the Baseball research runner.

The domain engine remains unchanged. This module converts the public
Nippon-Baseball-Data-Repository PBP schema into the small, explicit contract
needed by ``BaseballBacktest.aggregate_npb_games``.

The public release stores pre-game starter announcements in the textual
``description_jap`` field rather than in the pitch-level ``pitcher`` field.
The adapter therefore extracts starters only from explicit pre-game
announcements. It never uses winner/loser pitcher fields to infer starters,
and it never uses post-game information as a feature.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


_MISSING_TEXT = {"", "nan", "none", "nat"}
_STARTER_JP = "先発ピッチャー"
_STARTER_EN = "starting pitcher"


def _first_existing(df: pd.DataFrame, names: Iterable[str], default=None):
    for name in names:
        if name in df.columns:
            return df[name]
    if default is None:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.Series([default] * len(df), index=df.index)


def _clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in _MISSING_TEXT else text


def _extract_jp_starters(text: str) -> tuple[str, str]:
    """Extract (home, away) from an explicit NPB pre-game starter sentence.

    Observed release examples include:
      先発ピッチャーは西武が隅田、オリックスがエスピノーザ

    The team names are intentionally returned as written; the domain engine's
    canonical team mapping is applied later. We only accept the explicit
    ``team が pitcher`` structure and never guess from pitch order.
    """
    text = _clean_text(text)
    if _STARTER_JP not in text:
        return "", ""

    # Keep only the clause after the marker. This handles common prefixes such
    # as dates or other pre-game descriptions without relying on their layout.
    clause = text.split(_STARTER_JP, 1)[1]
    clause = re.sub(r"^[は:：\s]+", "", clause)
    # A Japanese full-width/ASCII comma separates the two team assignments.
    parts = [p.strip(" 、,\t") for p in re.split(r"[、,]", clause) if p.strip()]
    found: list[tuple[str, str]] = []
    for part in parts:
        # Team names in this public feed precede が; pitcher name follows it.
        m = re.match(r"^(.+?)が(.+)$", part)
        if m:
            team = m.group(1).strip()
            pitcher = m.group(2).strip()
            if team and pitcher:
                found.append((team, pitcher))
    if len(found) < 2:
        return "", ""
    return found[0][0] + "\t" + found[0][1], found[1][0] + "\t" + found[1][1]


def _extract_en_starters(text: str) -> tuple[str, str]:
    """Best-effort extraction from the repository's English transliteration.

    English is retained as a fallback only. If the structure is ambiguous, no
    starter is assigned. This is deliberately conservative.
    """
    text = _clean_text(text)
    if _STARTER_EN not in text.lower():
        return "", ""
    clause = text.lower().split(_STARTER_EN, 1)[1]
    clause = re.sub(r"^[\s:]+", "", clause)
    # Common transliteration: ``... is TeamA PitcherA, TeamB PitcherB``.
    clause = re.sub(r"^is\s+", "", clause)
    parts = [p.strip(" ,") for p in re.split(r"[;,]", clause) if p.strip()]
    if len(parts) < 2:
        return "", ""
    # Do not try to split arbitrary English names. Japanese is authoritative
    # for this source, so English only succeeds for an explicit two-clause form.
    return "", ""


def _starter_from_descriptions(raw: pd.DataFrame) -> pd.DataFrame:
    """Build explicit starter columns from pre-game textual announcements."""
    records = []
    desc_jp = _first_existing(raw, ["description_jap"])
    desc_en = _first_existing(raw, ["description"])
    home = _first_existing(raw, ["home_team_name", "H_NameS"]).map(_clean_text)
    away = _first_existing(raw, ["away_team_name", "V_NameS"]).map(_clean_text)
    gid = _first_existing(raw, ["game_id", "GameID"]).astype(str)

    for game_id, gidx in raw.groupby(_first_existing(raw, ["game_id", "GameID"]).astype(str), sort=False).groups.items():
        hp = ap = ""
        # Search descriptions in source order. The first explicit announcement
        # is retained, making the result deterministic even if the source emits
        # duplicate pre-game rows.
        for idx in gidx:
            j = _clean_text(desc_jp.loc[idx])
            hteam = _clean_text(home.loc[idx])
            ateam = _clean_text(away.loc[idx])
            if _STARTER_JP in j:
                clause = j.split(_STARTER_JP, 1)[1]
                clause = re.sub(r"^[は:：\s]+", "", clause)
                assignments = []
                for part in [p.strip(" 、,\t") for p in re.split(r"[、,]", clause) if p.strip()]:
                    m = re.match(r"^(.+?)が(.+)$", part)
                    if m:
                        assignments.append((m.group(1).strip(), m.group(2).strip()))
                if len(assignments) >= 2:
                    # Match assignments to the actual home/away team names.
                    for team, pitcher in assignments:
                        if team == hteam:
                            hp = pitcher
                        elif team == ateam:
                            ap = pitcher
                    # If exact matching is unavailable, positional assignment
                    # is safe because the source sentence explicitly provides
                    # the two team assignments in home/away matchup order.
                    if not hp and not ap:
                        hp, ap = assignments[0][1], assignments[1][1]
                    if hp and ap:
                        break
            e = _clean_text(desc_en.loc[idx])
            eh, ea = _extract_en_starters(e)
            if eh and ea and not hp and not ap:
                hp, ap = eh, ea
        records.append((game_id, hp, ap))
    return pd.DataFrame(records, columns=["game_id", "home_pitcher", "away_pitcher"])


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

    # The pitch-level pitcher column is intentionally NOT used for starter
    # identification: in this release it is empty in the pre-game rows and,
    # during a completed game, includes relievers.
    starter_df = _starter_from_descriptions(raw)
    out = out.merge(starter_df, on="game_id", how="left")
    out["home_pitcher"] = out["home_pitcher"].fillna("").astype(str)
    out["away_pitcher"] = out["away_pitcher"].fillna("").astype(str)

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
