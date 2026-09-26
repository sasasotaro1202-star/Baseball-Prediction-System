"""Strict PIT evidence join for chronological baseball OOS artifacts.

Only exact game/event matches from the append-only PIT ledger are joined.
No timestamps are inferred from game time, retrieval time, or post-game data.
Rows without sufficient evidence remain unresolved.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIT_DIR = ROOT / "data" / "pit"


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"PIT row is not an object: {path}")
            rows.append(obj)
    return pd.DataFrame(rows)


def attach_pit_evidence(
    oos: pd.DataFrame,
    *,
    availability_path: str | Path = PIT_DIR / "availability_observations.jsonl",
    snapshots_path: str | Path = PIT_DIR / "source_snapshots.jsonl",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "game_id" not in oos.columns:
        raise ValueError("OOS artifact requires game_id")
    if "datetime" not in oos.columns:
        raise ValueError("OOS artifact requires datetime")

    out = oos.copy()
    out["game_id"] = out["game_id"].astype(str).str.strip()
    event_dt = pd.to_datetime(out["datetime"], errors="coerce", utc=True)
    if event_dt.isna().any():
        raise ValueError("OOS artifact contains invalid datetime")

    av = _read_jsonl(Path(availability_path))
    snap = _read_jsonl(Path(snapshots_path))

    if av.empty:
        report = {
            "status": "NO_EVIDENCE",
            "oos_rows": int(len(out)),
            "matched_rows": 0,
            "coverage": 0.0,
            "unmatched_rows": int(len(out)),
            "reason": "availability_ledger_empty",
        }
        return _add_columns(out, None, None, None), report

    av["game_id"] = av.get("game_id", pd.Series(dtype=str)).astype(str).str.strip()
    av["prediction_cutoff_dt"] = pd.to_datetime(
        av.get("prediction_cutoff"), errors="coerce", utc=True
    )
    av["observed_at_dt"] = pd.to_datetime(
        av.get("observed_at"), errors="coerce", utc=True
    )

    snap_by_id = {}
    if not snap.empty and "entity_id" in snap.columns:
        for _, row in snap.iterrows():
            eid = str(row.get("entity_id", "")).strip()
            if not eid:
                continue
            status = str(row.get("status", ""))
            available = pd.to_datetime(row.get("available_at"), errors="coerce", utc=True)
            if status != "KNOWN" or pd.isna(available):
                continue
            snap_by_id[eid] = row

    evidence = []
    for idx, row in out.iterrows():
        gid = str(row["game_id"]).strip()
        cutoff_event = event_dt.iloc[idx]
        candidates = av[av["game_id"] == gid].copy()
        if candidates.empty:
            evidence.append((None, None, "UNRESOLVED", "game_id_not_in_pit_ledger"))
            continue
        safe = candidates[
            candidates["prediction_cutoff_dt"].notna()
            & (candidates["prediction_cutoff_dt"] <= cutoff_event)
        ].copy()
        if safe.empty:
            evidence.append((None, None, "UNRESOLVED", "no_pit_cutoff_at_or_before_event"))
            continue
        safe = safe.sort_values("prediction_cutoff_dt")
        chosen = safe.iloc[-1]
        cutoff = chosen["prediction_cutoff_dt"]
        sid = str(chosen.get("event_id", "")).strip()
        snap_row = snap_by_id.get(sid)
        if snap_row is None:
            # Some ledgers use the raw game_id as entity_id while event_id is
            # namespaced; try the plain id before declaring the row unresolved.
            snap_row = snap_by_id.get(gid)
        if snap_row is None:
            evidence.append((None, cutoff, "UNRESOLVED", "matching_snapshot_missing"))
            continue
        available = pd.to_datetime(snap_row.get("available_at"), errors="coerce", utc=True)
        if pd.isna(available) or available > cutoff:
            evidence.append((None, cutoff, "UNRESOLVED", "snapshot_not_available_by_prediction_cutoff"))
            continue
        # The source itself explicitly records the snapshot as KNOWN.
        evidence.append((available, cutoff, "PIT_VERIFIED", "exact_game_snapshot"))

    out["available_at"] = [x[0] for x in evidence]
    out["prediction_time"] = [x[1] for x in evidence]
    out["pit_join_status"] = [x[2] for x in evidence]
    out["pit_join_reason"] = [x[3] for x in evidence]

    matched = int((out["pit_join_status"] == "PIT_VERIFIED").sum())
    report = {
        "status": "PIT_PARTIAL" if matched < len(out) else "PIT_COMPLETE",
        "oos_rows": int(len(out)),
        "matched_rows": matched,
        "coverage": float(matched / max(1, len(out))),
        "unmatched_rows": int(len(out) - matched),
        "fully_verified": bool(matched == len(out)),
    }
    return out, report


def _add_columns(
    out: pd.DataFrame,
    available_at: Any,
    prediction_time: Any,
    status: Any,
) -> pd.DataFrame:
    out = out.copy()
    out["available_at"] = available_at
    out["prediction_time"] = prediction_time
    out["pit_join_status"] = status if status is not None else "UNRESOLVED"
    out["pit_join_reason"] = "availability_ledger_empty"
    return out
