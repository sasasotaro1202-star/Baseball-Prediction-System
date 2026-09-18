"""Curated, leakage-safe pregame interaction features.

This module is intentionally small and explicit. It combines only features that
already exist at prediction time, so it cannot introduce target leakage by itself.
The research runner can import `add_validated_interactions` after the base feature
builder and evaluate the resulting candidate under the normal chronological OOS
and holdout gates.
"""
from __future__ import annotations

import math
from typing import Dict


def _finite(value: object, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def normalized_product(a: object, b: object, a_scale: float, b_scale: float) -> float:
    x = _finite(a) / max(float(a_scale), 1e-9)
    y = _finite(b) / max(float(b_scale), 1e-9)
    return max(-5.0, min(5.0, x * y))


def add_validated_interactions(features: Dict[str, float]) -> Dict[str, float]:
    """Return a copy with a conservative whitelist of pregame interactions."""
    out = dict(features)
    out["ix_home_starter_quality_vs_away_so"] = normalized_product(features.get("starter_x_quality_proxy"), features.get("a_bat_so_rate_10"), 2.0, 0.25)
    out["ix_away_starter_quality_vs_home_so"] = normalized_product(features.get("starter_x_quality_proxy"), features.get("h_bat_so_rate_10"), 2.0, 0.25)
    out["ix_home_starter_kbb_vs_away_contact"] = normalized_product(features.get("starter_kbb_gap"), features.get("a_lineup_contact_rate"), 4.0, 0.70)
    out["ix_away_starter_kbb_vs_home_contact"] = normalized_product(features.get("starter_kbb_gap"), features.get("h_lineup_contact_rate"), 4.0, 0.70)
    out["ix_home_power_vs_away_hr9"] = normalized_product(features.get("offense_power_gap_10"), features.get("as_hr9"), 0.05, 1.0)
    out["ix_away_power_vs_home_hr9"] = normalized_product(-_finite(features.get("offense_power_gap_10")), features.get("hs_hr9"), 0.05, 1.0)
    out["ix_home_walk_vs_away_bb9"] = normalized_product(features.get("h_bat_bb_rate_10"), features.get("as_bb9"), 0.08, 3.0)
    out["ix_away_walk_vs_home_bb9"] = normalized_product(features.get("a_bat_bb_rate_10"), features.get("hs_bb9"), 0.08, 3.0)
    out["ix_home_so_offense_vs_away_k9"] = normalized_product(features.get("h_bat_so_rate_10"), features.get("as_k9"), 0.20, 7.5)
    out["ix_away_so_offense_vs_home_k9"] = normalized_product(features.get("a_bat_so_rate_10"), features.get("hs_k9"), 0.20, 7.5)
    out["ix_bullpen_fatigue_x_run_environment"] = normalized_product(features.get("bullpen_fatigue_diff"), features.get("expected_env"), 3.0, 4.0)
    out["ix_bullpen_fatigue_x_starter_gap"] = normalized_product(features.get("bullpen_fatigue_diff"), features.get("starter_x_quality_proxy"), 3.0, 2.0)
    out["ix_home_iso_x_wind"] = normalized_product(features.get("h_bat_iso_proxy_10"), features.get("weather_wind_kmh"), 0.13, 15.0)
    out["ix_away_iso_x_wind"] = normalized_product(features.get("a_bat_iso_proxy_10"), features.get("weather_wind_kmh"), 0.13, 15.0)
    out["ix_total_power_x_temperature"] = normalized_product(_finite(features.get("h_bat_iso_proxy_10")) + _finite(features.get("a_bat_iso_proxy_10")), features.get("weather_temp_c"), 0.26, 20.0)
    out["ix_home_form_x_run_volatility"] = normalized_product(features.get("h_gd_10"), features.get("run_volatility_gap_20"), 2.0, 1.5)
    out["ix_away_form_x_run_volatility"] = normalized_product(-_finite(features.get("a_gd_10")), features.get("run_volatility_gap_20"), 2.0, 1.5)
    for key, value in list(out.items()):
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            out[key] = 0.0
    return out
