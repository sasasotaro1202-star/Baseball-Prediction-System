"""Compatibility API for the canonical baseball feature-set registry.

The canonical definition lives in research.feature_set_variants. This module
exists only so older callers can keep importing select_features without creating
a second, divergent feature taxonomy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from research.feature_set_variants import (
    FEATURE_SET_CONTRACT,
    SCREENING_VARIANTS,
    feature_family,
    selected_columns,
    select_feature_set,
)


FEATURE_MANIFEST_VERSION = FEATURE_SET_CONTRACT


@dataclass(frozen=True)
class FeatureSetSelection:
    feature_set_id: str
    variant: str
    feature_manifest_version: str
    feature_count: int
    feature_schema_hash: str
    columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]


def available_variants() -> tuple[str, ...]:
    return SCREENING_VARIANTS


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
    selected = selected_columns(frame.columns, variant)
    if len(selected) < min_features:
        raise ValueError(
            f"feature-set {variant} selected only {len(selected)} columns; "
            f"minimum required is {min_features}"
        )
    meta = select_feature_set(frame, "UNKNOWN", variant=variant)[1]
    selection = FeatureSetSelection(
        feature_set_id=str(meta["feature_set_id"]),
        variant=str(meta["feature_set_variant"]),
        feature_manifest_version=str(feature_manifest_version),
        feature_count=int(meta["feature_count"]),
        feature_schema_hash=str(meta["feature_schema_hash"]),
        columns=tuple(selected),
        excluded_columns=tuple(str(c) for c in frame.columns if str(c) not in set(selected)),
    )
    return frame.loc[:, selected].copy(), selection


def summarize_sets(
    frame: pd.DataFrame,
    variants: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    chosen = tuple(variants) if variants is not None else available_variants()
    return [
        {
            **select_feature_set(frame, "UNKNOWN", variant=v)[1],
            "selected_columns": list(select_feature_set(frame, "UNKNOWN", variant=v)[0].columns),
        }
        for v in chosen
    ]
