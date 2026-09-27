"""PIT-safe loader for historical total-runs market lines.

The loader is corpus-based: it never invents a line and never uses a market
observation whose available_at is after the game's prediction cutoff.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from data.market_lines import from_mapping

DEFAULT_FILENAMES = (
    "market_lines.csv",
    "market_total_runs.csv",
    "total_runs_lines.csv",
    "market_lines.jsonl",
)


def _candidate_files(data_dir: Path) -> list[Path]:
    return [data_dir / name for name in DEFAULT_FILENAMES if (data_dir / name).exists()]


def _read_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return pd.DataFrame(rows)


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    aliases = {
        "game_id": "event_id",
        "id": "event_id",
        "total_line": "line",
        "total_runs_line": "line",
        "market_total": "line",
        "observed": "observed_at",
        "available": "available_at",
    }
    for source, target in aliases.items():
        if target not in df.columns and source in df.columns:
            df[target] = df[source]
    return df


def load_market_line_corpus(data_dir: Path) -> pd.DataFrame:
    files = _candidate_files(Path(data_dir))
    cols = ["event_id", "league", "line", "source", "observed_at", "available_at", "status"]
    if not files:
        return pd.DataFrame(columns=cols)

    frames: list[pd.DataFrame] = []
    for path in files:
        raw = _normalize_keys(_read_file(path))
        if raw.empty:
            continue
        required = {"event_id", "league", "line", "observed_at", "available_at"}
        if not required.issubset(raw.columns):
            continue
        if "source" not in raw.columns:
            raw["source"] = path.name
        if "status" not in raw.columns:
            raw["status"] = "KNOWN"
        valid_rows: list[dict[str, Any]] = []
        for record in raw.to_dict("records"):
            try:
                obj = from_mapping(record)
                obj.validate()
            except (KeyError, TypeError, ValueError):
                continue
            valid_rows.append({
                "event_id": obj.event_id,
                "league": obj.league,
                "line": float(obj.line) if obj.line is not None else None,
                "source": obj.source,
                "observed_at": obj.observed_at,
                "available_at": obj.available_at,
                "status": obj.status,
            })
        if valid_rows:
            frames.append(pd.DataFrame(valid_rows))

    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    out["observed_at"] = pd.to_datetime(out["observed_at"], errors="coerce", utc=True)
    out["available_at"] = pd.to_datetime(out["available_at"], errors="coerce", utc=True)
    return out.dropna(subset=["observed_at", "available_at"]).reset_index(drop=True)


def attach_pit_safe_market_lines(games: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Attach latest PIT-safe total-runs line to each game."""
    games = games.copy()
    games["market_line_known"] = 0.0
    games["market_total_runs_line"] = pd.to_numeric(
        games.get("expected_env", pd.Series(0.0, index=games.index)),
        errors="coerce",
    ).fillna(0.0)
    games["market_line_observed_at"] = ""
    games["market_line_available_at"] = ""
    games["market_line_source"] = ""
    games["market_line_event_id"] = ""

    corpus = load_market_line_corpus(Path(data_dir))
    if corpus.empty:
        return games

    cutoff_col = "prediction_cutoff" if "prediction_cutoff" in games.columns else "datetime"
    key_cols = [c for c in ("event_id", "game_id") if c in games.columns]
    if not key_cols:
        return games

    corpus["event_id"] = corpus["event_id"].astype(str)
    for idx, row in games.iterrows():
        cutoff = pd.to_datetime(row.get(cutoff_col), errors="coerce", utc=True)
        if pd.isna(cutoff):
            continue
        league = str(row.get("league", "")).upper()
        eligible = corpus[
            (corpus["league"].astype(str).str.upper() == league)
            & (corpus["status"].astype(str).str.upper() == "KNOWN")
            & (corpus["available_at"] <= cutoff)
            & (corpus["observed_at"] <= cutoff)
        ]
        if eligible.empty:
            continue

        matched = pd.DataFrame()
        for key in key_cols:
            value = row.get(key)
            if value in (None, "") or pd.isna(value):
                continue
            candidate = eligible[eligible["event_id"] == str(value)]
            if not candidate.empty:
                matched = candidate
                break
        if matched.empty:
            continue

        picked = matched.sort_values(["available_at", "observed_at"]).iloc[-1]
        games.at[idx, "market_line_known"] = 1.0
        games.at[idx, "market_total_runs_line"] = float(picked["line"])
        games.at[idx, "market_line_observed_at"] = picked["observed_at"].isoformat()
        games.at[idx, "market_line_available_at"] = picked["available_at"].isoformat()
        games.at[idx, "market_line_source"] = str(picked["source"])
        games.at[idx, "market_line_event_id"] = str(picked["event_id"])
    return games
