"""Strict PIT evidence join for chronological baseball OOS artifacts.

Only exact prediction-cutoff matches from the append-only PIT ledger are joined.
No prediction timestamp is reconstructed from event time, retrieval time, or
"latest observation before the game". Rows without an explicit OOS prediction
timestamp remain unresolved.
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

    out = oos.copy().reset_index(drop=True)
    out["game_id"] = out["game_id"].astype(str).str.strip()
    event_dt = pd.to_datetime(out["datetime"], errors="coerce", utc=True).reset_index(drop=True)
    if event_dt.isna().any():
        raise ValueError("OOS artifact contains invalid datetime")

    explicit_pt = None
    if "prediction_time" in out.columns:
        explicit_pt = pd.to_datetime(out["prediction_time"], errors="coerce", utc=True).reset_index(drop=True)
    out["available_at"] = pd.NaT
    out["pit_join_status"] = "UNRESOLVED"
    out["pit_join_reason"] = "prediction_time_not_recorded_in_oos"

    av = _read_jsonl(Path(availability_path))
    snap = _read_jsonl(Path(snapshots_path))

    if explicit_pt is None:
        return out, {
            "status": "NO_EXPLICIT_PREDICTION_TIME",
            "oos_rows": int(len(out)),
            "matched_rows": 0,
            "coverage": 0.0,
            "unmatched_rows": int(len(out)),
            "fully_verified": False,
            "reason": "historical OOS artifact does not carry the original prediction_time",
        }

    invalid_pt = explicit_pt.isna()
    out.loc[invalid_pt, "pit_join_reason"] = "prediction_time_missing_or_invalid"
    if av.empty:
        out.loc[~invalid_pt, "pit_join_reason"] = "availability_ledger_empty"
        return out, {
            "status": "NO_EVIDENCE",
            "oos_rows": int(len(out)),
            "matched_rows": 0,
            "coverage": 0.0,
            "unmatched_rows": int(len(out)),
            "fully_verified": False,
            "reason": "availability_ledger_empty",
        }

    av["game_id"] = av.get("game_id", pd.Series(dtype=str)).astype(str).str.strip()
    av["prediction_cutoff_dt"] = pd.to_datetime(
        av.get("prediction_cutoff"), errors="coerce", utc=True
    )
    av["observed_at_dt"] = pd.to_datetime(
        av.get("observed_at"), errors="coerce", utc=True
    )

    snap_by_id: dict[str, list[tuple[pd.Timestamp, Any]]] = {}
    if not snap.empty and "entity_id" in snap.columns:
        for _, row in snap.iterrows():
            eid = str(row.get("entity_id", "")).strip()
            if not eid:
                continue
            status = str(row.get("status", ""))
            available = pd.to_datetime(row.get("available_at"), errors="coerce", utc=True)
            if status != "KNOWN" or pd.isna(available):
                continue
            snap_by_id.setdefault(eid, []).append((available, row))
        for eid in list(snap_by_id):
            snap_by_id[eid].sort(key=lambda x: x[0])

    evidence: list[tuple[Any, Any, str, str]] = []
    for idx, row in out.iterrows():
        gid = str(row["game_id"]).strip()
        pt = explicit_pt.iloc[idx]
        if pd.isna(pt):
            evidence.append((None, None, "UNRESOLVED", "prediction_time_missing_or_invalid"))
            continue
        if pt > event_dt.iloc[idx]:
            evidence.append((None, pt, "UNRESOLVED", "prediction_time_after_event_time"))
            continue

        candidates = av[
            (av["game_id"] == gid)
            & av["prediction_cutoff_dt"].notna()
            & (av["prediction_cutoff_dt"] == pt)
        ].copy()
        if candidates.empty:
            evidence.append((None, pt, "UNRESOLVED", "no_exact_pit_record_at_prediction_time"))
            continue

        # Exact event/cutoff identity is required. Never substitute the latest
        # earlier cutoff because doing so would reconstruct a prediction time.
        chosen = candidates.sort_values("observed_at_dt").iloc[-1]
        sid = str(chosen.get("event_id", "")).strip()
        snapshot_candidates = snap_by_id.get(sid) or snap_by_id.get(gid) or []
        safe_snapshots = [item for item in snapshot_candidates if item[0] <= pt]
        if not safe_snapshots:
            evidence.append((None, pt, "UNRESOLVED", "matching_snapshot_missing_or_after_prediction_time"))
            continue

        available, _snap_row = safe_snapshots[-1]
        evidence.append((available, pt, "PIT_VERIFIED", "exact_game_and_prediction_cutoff"))

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
