"""Versioned feature-set selection for baseball research.

This module only selects columns. It does not fit a model or promote a feature set.
All selection decisions remain subject to PIT, chronological OOS/WFO, calibration,
ablation, robustness and frozen-holdout gates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class FeatureSetSelection:
    feature_set_id: str
    variant: str
    feature_manifest_version: str
    feature_count: int
    feature_schema_hash: str
    columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]


FEATURE_MANIFEST_VERSION = "feature-contract-v1"

_VARIANTS = (
    "BASELINE_TEAM_STATE",
    "TEAM_PLUS_STARTER",
    "TEAM_PLUS_BULLPEN",
    "TEAM_PLUS_LINEUP_PIT_SAFE",
    "TEAM_PLUS_WEATHER_PIT_SAFE",
    "FULL_VALIDATED_ENSEMBLE",
    "SCORE_MODEL_FEATURE_SET",
    "RESEARCH_STATCAST_SET",
)


def available_variants() -> tuple[str, ...]:
    return _VARIANTS


def _ordered_hash(columns: Sequence[str]) -> str:
    raw = "\n".join(str(c) for c in columns)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _base_team_state(name: str) -> bool:
    if name == "home_adv":
        return True
    if re.match(r"^[had]_(pts|gf|ga|gd|win|draw|win_shrunk|gd_shrunk|draw_shrunk)_\d+$", name):
        return True
    if re.match(r"^[ha]_(venue_n|venue_pts|venue_gf|venue_ga|elo|rest_days|matches|bp3|bp7)$", name):
        return True
    if re.match(r"^[had]_(bat_ab|bat_avg|bat_hr|bat_bb|bat_so|bat_bb_rate|bat_so_rate|bat_xbh|bat_hr_rate|bat_iso_proxy|bat_extra_base_rate)_\d+$", name):
        return True
    if re.match(r"^[had]_(gf|ga|hr|so|bb)_(sd|slope)_20$", name):
        return True
    return False


def _starter(name: str) -> bool:
    return (
        name.startswith(("hs_", "as_"))
        or any(token in name for token in (
            "starter_x_quality", "starter_recency", "starter_experience",
            "starter_kbb", "starter_hr_", "starter_recent_form", "starter_known",
            "matchup_home_bat_vs_away_fip", "matchup_away_bat_vs_home_fip",
            "elo_x_starter", "offense_power_x_starter", "starter_quality_reliability",
        ))
    )


def _bullpen(name: str) -> bool:
    return name.startswith(("h_bp", "a_bp")) or any(token in name for token in (
        "bullpen_", "bp_",
    ))


def _lineup(name: str) -> bool:
    return "lineup_" in name


def _weather(name: str) -> bool:
    return name.startswith("weather_")


def _statcast(name: str) -> bool:
    lowered = name.lower()
    return "statcast" in lowered or "_lag_pitch_" in lowered or any(
        token in lowered
        for token in ("velocity_mean", "spin_rate_mean", "horizontal_break", "vertical_break",
                      "exit_velocity", "launch_angle", "hard_hit_rate", "barrel_rate")
    )


def _family_columns(columns: Sequence[str], variant: str) -> set[str]:
    base = {c for c in columns if _base_team_state(c)}
    starter = {c for c in columns if _starter(c)}
    bullpen = {c for c in columns if _bullpen(c)}
    lineup = {c for c in columns if _lineup(c)}
    weather = {c for c in columns if _weather(c)}
    statcast = {c for c in columns if _statcast(c)}

    if variant == "BASELINE_TEAM_STATE":
        return base
    if variant == "TEAM_PLUS_STARTER":
        return base | starter
    if variant == "TEAM_PLUS_BULLPEN":
        return base | bullpen
    if variant == "TEAM_PLUS_LINEUP_PIT_SAFE":
        return base | lineup
    if variant == "TEAM_PLUS_WEATHER_PIT_SAFE":
        return base | weather
    if variant == "FULL_VALIDATED_ENSEMBLE":
        return set(columns)
    if variant == "SCORE_MODEL_FEATURE_SET":
        return set(columns)
    if variant == "RESEARCH_STATCAST_SET":
        return base | starter | bullpen | statcast
    raise ValueError(f"unknown feature-set variant: {variant}")


def select_features(
    frame: pd.DataFrame,
    variant: str,
    *,
    feature_manifest_version: str = FEATURE_MANIFEST_VERSION,
    min_features: int = 1,
) -> tuple[pd.DataFrame, FeatureSetSelection]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not frame.columns.is_unique:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"feature columns are not unique: {duplicates[:20]}")
    variant = str(variant).strip().upper()
    if variant not in _VARIANTS:
        raise ValueError(f"unknown feature-set variant: {variant}")
    if min_features < 1:
        raise ValueError("min_features must be >= 1")

    original = tuple(str(c) for c in frame.columns)
    selected_set = _family_columns(original, variant)
    selected = tuple(c for c in original if c in selected_set)
    if len(selected) < min_features:
        raise ValueError(
            f"feature-set {variant} selected only {len(selected)} columns; "
            f"minimum required is {min_features}"
        )
    excluded = tuple(c for c in original if c not in selected_set)
    digest = _ordered_hash(selected)
    selection = FeatureSetSelection(
        feature_set_id=f"{variant}:{digest[:16]}",
        variant=variant,
        feature_manifest_version=str(feature_manifest_version),
        feature_count=len(selected),
        feature_schema_hash=digest,
        columns=selected,
        excluded_columns=excluded,
    )
    return frame.loc[:, list(selected)].copy(), selection


def summarize_sets(frame: pd.DataFrame, variants: Iterable[str] | None = None) -> list[dict[str, object]]:
    chosen = tuple(variants) if variants is not None else available_variants()
    reports: list[dict[str, object]] = []
    for variant in chosen:
        selected, meta = select_features(frame, variant)
        reports.append({
            "feature_set_id": meta.feature_set_id,
            "variant": meta.variant,
            "feature_count": meta.feature_count,
            "feature_schema_hash": meta.feature_schema_hash,
            "excluded_count": len(meta.excluded_columns),
            "selected_columns": list(selected.columns),
        })
    return reports
