"""Research-only NPB tracking-data contract.

This module does not claim direct access to NPB DMP/Hawk-Eye data. It normalizes
tracking payloads supplied by an approved collector and enforces fail-closed PIT
metadata before any feature can be used in historical OOS research.

Supported public-facing NPB+ / Hawk-Eye concepts include pitch speed, spin rate,
pitch movement, batted-ball speed/distance/angle, swing speed, sprint speed, and
throw speed. Exact source fields must be mapped explicitly; unknown fields are
ignored rather than guessed.
"""
from __future__ import annotations

from typing import Any, Mapping
import pandas as pd


FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "pitch_speed_kmh": ("pitch_speed_kmh", "pitch_velocity_kmh", "velocity_kmh", "release_speed_kmh"),
    "spin_rate_rpm": ("spin_rate_rpm", "release_spin_rate", "spin_rate"),
    "pitch_vertical_break_cm": ("pitch_vertical_break_cm", "vertical_break_cm", "induced_vertical_break"),
    "pitch_horizontal_break_cm": ("pitch_horizontal_break_cm", "horizontal_break_cm", "horizontal_break"),
    "exit_velocity_kmh": ("exit_velocity_kmh", "batted_ball_speed_kmh", "launch_speed_kmh"),
    "launch_angle_deg": ("launch_angle_deg", "batted_ball_angle_deg", "launch_angle"),
    "hit_distance_m": ("hit_distance_m", "batted_ball_distance_m", "hit_distance"),
    "swing_speed_kmh": ("swing_speed_kmh", "bat_speed_kmh", "swing_speed"),
    "sprint_speed_mps": ("sprint_speed_mps", "sprint_speed"),
    "throw_speed_kmh": ("throw_speed_kmh", "throw_velocity_kmh", "throw_speed"),
}

TIME_COLUMNS = ("event_time", "published_at", "available_at", "retrieved_at", "prediction_time")


class TrackingPITError(ValueError):
    """Raised when tracking data cannot be proven available at prediction time."""


def _first(row: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    for name in aliases:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return None


def normalize_tracking_frame(
    frame: pd.DataFrame,
    *,
    source_id: str,
    prediction_time_col: str = "prediction_time",
    require_pit: bool = True,
) -> pd.DataFrame:
    """Normalize tracking observations without inventing missing values."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("tracking frame must be a non-empty DataFrame")
    if not source_id.strip():
        raise ValueError("source_id is required")

    out = frame.copy()
    for target, aliases in FIELD_ALIASES.items():
        out[target] = [_first(row, aliases) for row in out.to_dict("records")]
        if target in out:
            out[target] = pd.to_numeric(out[target], errors="coerce")

    for col in TIME_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce", utc=True)

    # A tracking value can only enter a prediction if an explicit availability
    # boundary exists. Retrieval time is not used as a substitute historical
    # publication/availability timestamp.
    if require_pit:
        if "available_at" not in out.columns:
            raise TrackingPITError("available_at is missing; tracking data is not PIT-verifiable")
        if prediction_time_col not in out.columns:
            raise TrackingPITError("prediction_time is missing; tracking data is not PIT-verifiable")
        if out["available_at"].isna().any() or out[prediction_time_col].isna().any():
            raise TrackingPITError("tracking PIT timestamps contain missing values")
        if (out["available_at"] > out[prediction_time_col]).any():
            raise TrackingPITError("tracking data available after prediction_time")

    out["tracking_source_id"] = source_id
    out["tracking_pit_status"] = "PASS" if require_pit else "UNVERIFIED"
    return out


def lagged_player_summary(
    frame: pd.DataFrame,
    *,
    player_col: str = "player_id",
    time_col: str = "event_time",
    min_observations: int = 10,
) -> pd.DataFrame:
    """Create historical player tracking aggregates for research use only.

    The caller must supply a frame already restricted to observations available
    before the target Prediction Time. This function performs no future filtering.
    """
    if player_col not in frame.columns or time_col not in frame.columns:
        raise ValueError("player_id and event_time are required")
    if min_observations <= 0:
        raise ValueError("min_observations must be positive")

    data = frame.copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce", utc=True)
    data = data.dropna(subset=[player_col, time_col]).sort_values([player_col, time_col])
    metric_cols = [c for c in FIELD_ALIASES if c in data.columns]
    if not metric_cols:
        raise ValueError("no supported tracking metrics are present")

    rows: list[dict[str, Any]] = []
    for player, group in data.groupby(player_col, sort=False):
        usable = group.dropna(subset=metric_cols, how="all")
        if len(usable) < min_observations:
            continue
        rec: dict[str, Any] = {player_col: player, "tracking_observations": int(len(usable))}
        for col in metric_cols:
            vals = pd.to_numeric(usable[col], errors="coerce").dropna()
            if vals.empty:
                continue
            rec[f"{col}_mean"] = float(vals.mean())
            rec[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            rec[f"{col}_last"] = float(vals.iloc[-1])
        rows.append(rec)
    return pd.DataFrame(rows)
