import numpy as np
import pandas as pd
import pytest

from research.feature_set_variants import (
    SCREENING_VARIANTS,
    feature_family,
    select_feature_set,
)


def _sample_frame():
    columns = [
        "home_adv",
        "h_elo", "a_elo", "d_elo",
        "h_gf_10", "a_gf_10", "d_gf_10",
        "h_gf_sd_20", "a_gf_sd_20", "d_gf_sd_20",
        "hs_era", "as_era", "starter_kbb_gap",
        "h_bp_app_10", "a_bp_app_10", "bullpen_fatigue_diff",
        "h_bat_avg_10", "a_bat_avg_10", "d_bat_avg_10",
        "offense_power_gap_10",
        "form_x_rest_gap",
        "h_lineup_avg", "a_lineup_avg", "d_lineup_avg",
        "weather_temp_c", "weather_run_signal",
        "context_pit_safe",
        "expected_env",
    ]
    return pd.DataFrame(np.arange(len(columns) * 2, dtype=float).reshape(2, -1), columns=columns)


def test_feature_family_is_explicit():
    assert feature_family("h_elo") == "core"
    assert feature_family("h_gf_sd_20") == "volatility"
    assert feature_family("hs_era") == "starter"
    assert feature_family("h_bp_app_10") == "bullpen"
    assert feature_family("h_bat_avg_10") == "offense"
    assert feature_family("form_x_rest_gap") == "interaction"
    assert feature_family("h_lineup_avg") == "context"
    assert feature_family("weather_temp_c") == "context"


def test_baseline_excludes_starter_bullpen_offense_context_and_interactions():
    X, meta = select_feature_set(_sample_frame(), "NPB", variant="BASELINE_TEAM_STATE")
    assert set(["home_adv", "h_elo", "a_elo", "d_elo", "expected_env"]).issubset(X.columns)
    assert not any(str(c).startswith(("hs_", "as_", "h_bp_", "a_bp_", "h_bat_", "a_bat_", "h_lineup_", "a_lineup_", "weather_")) for c in X.columns)
    assert "form_x_rest_gap" not in X.columns
    assert meta["feature_set_variant"] == "BASELINE_TEAM_STATE"


@pytest.mark.parametrize("variant", SCREENING_VARIANTS)
def test_every_variant_is_nonempty_and_deterministic(variant):
    frame = _sample_frame()
    x1, m1 = select_feature_set(frame, "NPB", variant=variant)
    x2, m2 = select_feature_set(frame, "NPB", variant=variant)
    assert list(x1.columns) == list(x2.columns)
    assert m1["feature_schema_hash"] == m2["feature_schema_hash"]
    assert m1["feature_set_id"] == m2["feature_set_id"]
    assert len(x1.columns) > 0
    assert np.isfinite(x1.to_numpy()).all()


def test_full_validated_ensemble_preserves_all_columns():
    frame = _sample_frame()
    x, meta = select_feature_set(frame, "NPB", variant="FULL_VALIDATED_ENSEMBLE")
    assert list(x.columns) == list(frame.columns)
    assert meta["feature_count"] == len(frame.columns)


def test_unknown_variant_fails_closed():
    with pytest.raises(ValueError):
        select_feature_set(_sample_frame(), "NPB", variant="NOT_A_REAL_VARIANT")


def test_pit_context_variants_fail_closed_without_required_context():
    base = _sample_frame().drop(columns=["h_lineup_avg", "a_lineup_avg", "d_lineup_avg", "weather_temp_c", "weather_run_signal"])
    with pytest.raises(ValueError, match="requires unavailable PIT-safe feature families"):
        select_feature_set(base, "NPB", variant="TEAM_PLUS_LINEUP_PIT_SAFE")
    with pytest.raises(ValueError, match="requires unavailable PIT-safe feature families"):
        select_feature_set(base, "NPB", variant="TEAM_PLUS_WEATHER_PIT_SAFE")
