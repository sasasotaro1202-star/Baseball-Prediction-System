"""Deterministic feature-set variants for baseball research and production traceability.

Only selects already-constructed PIT-safe columns. It never creates information,
performs imputation, or changes target semantics.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

import pandas as pd

FEATURE_SET_CONTRACT = "feature-contract-v1"

_FAMILY_PATTERNS = (
    ("lineup", re.compile(r"^(?:h_|a_|d_)?lineup_")),
    ("weather", re.compile(r"^weather_")),
    ("context", re.compile(r"(?:^|_)context(?:_|$)")),
    ("starter", re.compile(r"^(?:hs_|as_|starter_)|^starter_(?:known|x_)")),
    ("bullpen", re.compile(r"^(?:h_|a_|d_)?(?:bp_|bullpen_)|^(?:bullpen_|bp_)")),
    ("offense", re.compile(r"^(?:h_|a_|d_)?(?:bat_|offense_|matchup_)")),
    ("volatility", re.compile(r"(?:sd_20|slope_20)$|^run_(?:volatility|trend)_")),
    ("interaction", re.compile(r"_x_|(?:^|_)(?:gap|diff)_(?:10|20)$")),
)

VARIANT_FAMILIES: dict[str, frozenset[str]] = {
    "BASELINE_TEAM_STATE": frozenset({"core", "volatility"}),
    "TEAM_PLUS_STARTER": frozenset({"core", "volatility", "starter"}),
    "TEAM_PLUS_BULLPEN": frozenset({"core", "volatility", "bullpen"}),
    "TEAM_PLUS_OFFENSE": frozenset({"core", "volatility", "offense"}),
    "TEAM_PLUS_LINEUP_PIT_SAFE": frozenset({"core", "volatility", "lineup"}),
    "TEAM_PLUS_WEATHER_PIT_SAFE": frozenset({"core", "volatility", "weather"}),
    "TEAM_PLUS_STARTER_BULLPEN": frozenset({"core", "volatility", "starter", "bullpen"}),
    "TEAM_PLUS_STARTER_OFFENSE": frozenset({"core", "volatility", "starter", "offense"}),
    "TEAM_PLUS_STARTER_LINEUP_PIT_SAFE": frozenset({"core", "volatility", "starter", "lineup"}),
    "TEAM_PLUS_STARTER_BULLPEN_LINEUP_PIT_SAFE": frozenset(
        {"core", "volatility", "starter", "bullpen", "lineup"}
    ),
    "TEAM_PLUS_STARTER_BULLPEN_OFFENSE": frozenset({"core", "volatility", "starter", "bullpen", "offense"}),
    "TEAM_PLUS_STARTER_BULLPEN_OFFENSE_INTERACTIONS": frozenset(
        {"core", "volatility", "starter", "bullpen", "offense", "interaction"}
    ),
    "FULL_NO_PIT_CONTEXT": frozenset(
        {"core", "volatility", "starter", "bullpen", "offense", "interaction"}
    ),
    "FULL_VALIDATED_ENSEMBLE": frozenset(
        {"core", "volatility", "starter", "bullpen", "offense", "interaction", "lineup", "weather", "context"}
    ),
}

SCREENING_VARIANTS: tuple[str, ...] = tuple(VARIANT_FAMILIES)


def feature_family(column: str) -> str:
    name = str(column)
    if name in {"home_adv", "expected_env"}:
        return "core"
    for family, pattern in _FAMILY_PATTERNS:
        if pattern.search(name):
            return family
    return "core"


def selected_columns(columns: Iterable[str], variant: str) -> list[str]:
    chosen = str(variant).strip().upper()
    if chosen not in VARIANT_FAMILIES:
        raise ValueError(
            f"unknown feature-set variant {chosen!r}; allowed={','.join(SCREENING_VARIANTS)}"
        )
    families = VARIANT_FAMILIES[chosen]
    selected = [str(c) for c in columns if feature_family(str(c)) in families]
    if not selected:
        raise ValueError(f"feature-set variant {chosen} selected zero columns")
    return selected


def select_feature_set(
    X: pd.DataFrame,
    league: str,
    *,
    variant: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    chosen = str(variant or "FULL_VALIDATED_ENSEMBLE").strip().upper()
    cols = selected_columns(X.columns, chosen)
    out = X.loc[:, cols].copy()
    digest = hashlib.sha256("\n".join(cols).encode("utf-8")).hexdigest()
    families = {feature_family(c) for c in cols}
    family_counts: dict[str, int] = {}
    for col in cols:
        fam = feature_family(col)
        family_counts[fam] = family_counts.get(fam, 0) + 1
    return out, {
        "feature_contract": FEATURE_SET_CONTRACT,
        "feature_set_variant": chosen,
        "feature_set_id": f"{FEATURE_SET_CONTRACT}:{chosen}:{digest[:16]}",
        "feature_count": int(len(cols)),
        "feature_schema_hash": digest,
        "feature_context_mode": (
            "PIT_SAFE_CONTEXT_ACTIVE" if "context" in families
            else "BASELINE_NO_PIT_SAFE_CONTEXT"
        ),
        "feature_family_counts": {k: int(v) for k, v in sorted(family_counts.items())},
        "league": str(league),
    }
