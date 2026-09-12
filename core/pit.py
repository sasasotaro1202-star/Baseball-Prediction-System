"""Point-in-time controls for the Baseball research engine.

This module is intentionally additive: existing Baseball feature builders remain
unchanged until they are explicitly wired to these helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class PITResult:
    frame: pd.DataFrame
    cutoff: pd.Timestamp
    rows_in: int
    rows_out: int


def _ts(value) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {value!r}")
    return ts


def filter_as_of(
    df: pd.DataFrame,
    cutoff,
    *,
    available_col: str = "available_ts",
    event_col: str = "event_time",
    strict_event_time: bool = False,
) -> PITResult:
    """Return rows whose information was available by *cutoff*.

    Availability is preferred over event time. Missing availability is not
    silently treated as available: when strict_event_time is true, rows without
    availability are accepted only when their event time is strictly before the
    cutoff. This prevents the generic v4.4 fallback from becoming Baseball data
    leakage.
    """
    out = df.copy()
    cutoff_ts = _ts(cutoff)
    if out.empty:
        return PITResult(out, cutoff_ts, 0, 0)

    available = (
        pd.to_datetime(out[available_col], errors="coerce", utc=True)
        if available_col in out.columns
        else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    )
    event = (
        pd.to_datetime(out[event_col], errors="coerce", utc=True)
        if event_col in out.columns
        else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    )

    known = available.notna() & (available <= cutoff_ts)
    missing_availability = available.isna()
    if strict_event_time:
        known = known | (missing_availability & event.notna() & (event < cutoff_ts))
    else:
        # Compatibility mode: only rows with explicit availability are admitted.
        # Event time is never used as a proxy for availability here.
        known = known

    result = out.loc[known].copy()
    return PITResult(result, cutoff_ts, len(out), len(result))


def require_pit_columns(df: pd.DataFrame, *, event_id: str = "event_id") -> None:
    """Validate the minimum columns required for a PIT-aware dataset."""
    missing = [c for c in (event_id, "event_time") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing PIT columns: {missing}")


def assert_no_future_rows(df: pd.DataFrame, cutoff, *, available_col="available_ts") -> None:
    """Raise if any explicit source-availability timestamp is after cutoff."""
    if available_col not in df.columns:
        return
    cutoff_ts = _ts(cutoff)
    ts = pd.to_datetime(df[available_col], errors="coerce", utc=True)
    future = ts.notna() & (ts > cutoff_ts)
    if future.any():
        raise ValueError(f"PIT violation: {int(future.sum())} rows are available after cutoff")
