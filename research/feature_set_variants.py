"""Canonical deterministic feature-set registry.

The registry defines *research variants*. Selection only drops columns already
constructed by the PIT-safe feature builder; it never creates information,
imputes missing values, or changes targets.
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
    ("interaction", re.compile(r"^(?:matchup_)|_x_|(?:^|_)(?:gap|diff)(?:_|$)")),
    ("starter", re.compile(r"^(?:hs_|as_|starter_)|^starter_(?:known|x_)")),
    ("bullpen", re.compile(r"^(?:h_|a_|d_)?(?:bp(?:_|[0-9])|bullpen_)|^(?:bullpen_|bp_)")),
    ("offense", re.compile(r"^(?:h_|a_|d_)?(?:bat_|offense_)")),
    ("volatility", re.compile(r"(?:sd_20|slope_20)$|^run_(?:volatility|trend)_")),
)

# All variants are explicit. The exact feature schema is still derived from
# the actual frame, so changes to upstream feature construction cannot silently
# masquerade as the same schema.
VARIANT_FAMILIES: dict[str, frozenset[str]] = {
    "BASELINE_TEAM_STATE": frozenset({"core", "volatility"}),
    "TEAM_CORE_SHORT_HORIZON": frozenset({"core"}),
    "TEAM_CORE_MEDIUM_HORIZON": frozenset({"core"}),
    "TEAM_CORE_LONG_HORIZON": frozenset({"core"}),
    "TEAM_CORE_NO_SHRINK": frozenset({"core"}),
    "TEAM_CORE_SHRINK_ONLY": frozenset({"core"}),
    "TEAM_CORE_NO_VOLATILITY": frozenset({"core"}),
    "TEAM_PLUS_STARTER": frozenset({"core", "volatility", "starter"}),
    "TEAM_PLUS_STARTER_RECENT": frozenset({"core", "volatility", "starter"}),
    "TEAM_PLUS_BULLPEN": frozenset({"core", "volatility", "bullpen"}),
    "TEAM_PLUS_BULLPEN_WORKLOAD": frozenset({"core", "volatility", "bullpen"}),
    "TEAM_PLUS_BULLPEN_QUALITY": frozenset({"core", "volatility", "bullpen"}),
    "TEAM_PLUS_OFFENSE": frozenset({"core", "volatility", "offense"}),
    "TEAM_PLUS_OFFENSE_RATE": frozenset({"core", "volatility", "offense"}),
    "TEAM_PLUS_OFFENSE_POWER": frozenset({"core", "volatility", "offense"}),
    "TEAM_PLUS_STARTER_BULLPEN": frozenset({"core", "volatility", "starter", "bullpen"}),
    "TEAM_PLUS_STARTER_OFFENSE": frozenset({"core", "volatility", "starter", "offense"}),
    "TEAM_PLUS_STARTER_INTERACTIONS": frozenset({"core", "volatility", "starter", "interaction"}),
    "TEAM_PLUS_STARTER_OFFENSE_INTERACTIONS": frozenset({"core", "volatility", "starter", "offense", "interaction"}),
    "TEAM_PLUS_STARTER_BULLPEN_OFFENSE_NO_INTERACTIONS": frozenset(
        {"core", "volatility", "starter", "bullpen", "offense"}
    ),
    "TEAM_PLUS_STARTER_BULLPEN_OFFENSE_INTERACTIONS": frozenset(
        {"core", "volatility", "starter", "bullpen", "offense", "interaction"}
    ),
    "TEAM_PLUS_LINEUP_PIT_SAFE": frozenset({"core", "volatility", "lineup"}),
    "TEAM_PLUS_STARTER_LINEUP_PIT_SAFE": frozenset({"core", "volatility", "starter", "lineup"}),
    "TEAM_PLUS_STARTER_BULLPEN_LINEUP_PIT_SAFE": frozenset(
        {"core", "volatility", "starter", "bullpen", "lineup"}
    ),
    "TEAM_PLUS_WEATHER_PIT_SAFE": frozenset({"core", "volatility", "weather"}),
    "FULL_NO_STARTER": frozenset({"core", "volatility", "bullpen", "offense", "interaction"}),
    "FULL_NO_BULLPEN": frozenset({"core", "volatility", "starter", "offense", "interaction"}),
    "FULL_NO_OFFENSE": frozenset({"core", "volatility", "starter", "bullpen", "interaction"}),
    "FULL_NO_INTERACTIONS": frozenset({"core", "volatility", "starter", "bullpen", "offense"}),
    "FULL_NO_PIT_CONTEXT": frozenset(
        {"core", "volatility", "starter", "bullpen", "offense", "interaction"}
    ),
    "FULL_PIT_LINEUP_ONLY": frozenset(
        {"core", "volatility", "starter", "bullpen", "offense", "interaction", "lineup"}
    ),
    "FULL_PIT_WEATHER_ONLY": frozenset(
        {"core", "volatility", "starter", "bullpen", "offense", "interaction", "weather"}
    ),
    "FULL_VALIDATED_ENSEMBLE": frozenset(
        {"core", "volatility", "starter", "bullpen", "offense", "interaction", "lineup", "weather", "context"}
    ),
}

REGISTERED_VARIANTS: tuple[str, ...] = tuple(VARIANT_FAMILIES)
SCREENING_VARIANTS: tuple[str, ...] = tuple(VARIANT_FAMILIES)

_STABLE_CORE = re.compile(
    r"^(?:home_adv|expected_env|h_(?:venue_|elo$|rest_days$|matches$|bp3$|bp7$)|"
    r"a_(?:venue_|elo$|rest_days$|matches$|bp3$|bp7$)|"
    r"d_(?:venue_|elo$|rest_days$|matches$|bp3$|bp7$))$"
)
_ANY_WINDOW = re.compile(r"_(?:3|5|10|20|30|45|60)$")

_VARIANT_EXCLUDES: dict[str, re.Pattern[str]] = {
    "TEAM_CORE_SHORT_HORIZON": re.compile(r"_(?:20|30|45|60)$"),
    "TEAM_CORE_MEDIUM_HORIZON": re.compile(r"_(?:3|30|45|60)$"),
    "TEAM_CORE_LONG_HORIZON": re.compile(r"_(?:3|5)$"),
    "TEAM_CORE_NO_SHRINK": re.compile(r"_shrunk_"),
    "TEAM_CORE_SHRINK_ONLY": re.compile(r"^(?!.*_shrunk_).*"),
    "TEAM_CORE_NO_VOLATILITY": re.compile(r"(?:sd_20|slope_20)$"),
    "TEAM_PLUS_STARTER_RECENT": re.compile(r"^(?:hs_|as_)(?!(?:recent_|starts$))"),
    "TEAM_PLUS_BULLPEN_WORKLOAD": re.compile(r"^(?:h_|a_|d_)?(?:bp_(?:era|whip|k9|bb9|hr9|actual_coverage)|bullpen_)"),
    "TEAM_PLUS_BULLPEN_QUALITY": re.compile(r"^(?:h_|a_|d_)?(?:bp(?:3|7|_app_10|_er_10|_runs_10)$|bullpen_)"),
    "TEAM_PLUS_OFFENSE_RATE": re.compile(r"^(?:h_|a_|d_)?(?:bat_(?:ab|hr|so|bb|xbh)_|matchup_)"),
    "TEAM_PLUS_OFFENSE_POWER": re.compile(r"^(?:h_|a_|d_)?(?:bat_(?!(?:hr|hr_rate|xbh|iso_proxy|extra_base_rate)_))"),
}


def feature_family(column: str) -> str:
    name = str(column)
    if name in {"home_adv", "expected_env"}:
        return "core"
    for family, pattern in _FAMILY_PATTERNS:
        if pattern.search(name):
            return family
    return "core"


def _keep_variant_specific(name: str, variant: str) -> bool:
    chosen = str(variant).strip().upper()
    if chosen == "TEAM_CORE_SHORT_HORIZON":
        return bool(
            _STABLE_CORE.search(name)
            or (feature_family(name) == "core" and re.search(r"_(?:3|5|10)$", name))
        )
    if chosen == "TEAM_CORE_MEDIUM_HORIZON":
        return bool(
            _STABLE_CORE.search(name)
            or (feature_family(name) == "core" and re.search(r"_(?:5|10|20)$", name))
        )
    if chosen == "TEAM_CORE_LONG_HORIZON":
        return bool(
            _STABLE_CORE.search(name)
            or (feature_family(name) == "core" and re.search(r"_(?:10|20|30|45|60)$", name))
        )
    if chosen == "TEAM_CORE_NO_SHRINK":
        return "_shrunk_" not in name
    if chosen == "TEAM_CORE_SHRINK_ONLY":
        return bool(_STABLE_CORE.search(name) or "_shrunk_" in name)
    if chosen == "TEAM_CORE_NO_VOLATILITY":
        return feature_family(name) != "volatility"
    if chosen == "TEAM_PLUS_STARTER_RECENT":
        return bool(
            feature_family(name) != "starter"
            or re.search(r"^(?:hs_|as_)(?:recent_|starts$)", name)
        )
    if chosen == "TEAM_PLUS_BULLPEN_WORKLOAD":
        return bool(
            feature_family(name) != "bullpen"
            or re.search(r"^(?:h_|a_|d_)?(?:bp3|bp7|bp_app_10|bp_runs_10|bp_er_10)$", name)
        )
    if chosen == "TEAM_PLUS_BULLPEN_QUALITY":
        return bool(
            feature_family(name) != "bullpen"
            or re.search(r"^(?:h_|a_|d_)?(?:bp_ip_10|bp_era_10|bp_whip_10|bp_k9_10|bp_bb9_10|bp_hr9_10|bp_actual_coverage_10)$", name)
        )
    if chosen == "TEAM_PLUS_OFFENSE_RATE":
        return bool(
            feature_family(name) != "offense"
            or re.search(r"^(?:h_|a_|d_)?(?:bat_(?:avg|bb_rate|so_rate|hr_rate|iso_proxy|extra_base_rate)_|matchup_)", name)
        )
    if chosen == "TEAM_PLUS_OFFENSE_POWER":
        return bool(
            feature_family(name) != "offense"
            or re.search(r"^(?:h_|a_|d_)?(?:bat_(?:hr|xbh|hr_rate|iso_proxy|extra_base_rate)_|offense_power_gap_10|matchup_)", name)
        )
    return True


def selected_columns(columns: Iterable[str], variant: str) -> list[str]:
    chosen = str(variant).strip().upper()
    if chosen not in VARIANT_FAMILIES:
        raise ValueError(
            f"unknown feature-set variant {chosen!r}; allowed={','.join(SCREENING_VARIANTS)}"
        )
    families = VARIANT_FAMILIES[chosen]
    selected = [
        str(c)
        for c in columns
        if feature_family(str(c)) in families and _keep_variant_specific(str(c), chosen)
    ]
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
            "PIT_SAFE_CONTEXT_ACTIVE" if families.intersection({"context", "lineup", "weather"})
            else "BASELINE_NO_PIT_SAFE_CONTEXT"
        ),
        "feature_family_counts": {k: int(v) for k, v in sorted(family_counts.items())},
        "league": str(league),
    }
