from __future__ import annotations

import pandas as pd
import pytest

from research.feature_set_router import (
    available_variants,
    select_features,
    summarize_sets,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "home_adv": [1.0, 1.0],
        "h_elo": [1500.0, 1520.0],
        "a_elo": [1490.0, 1510.0],
        "d_elo": [10.0, 10.0],
        "h_gf_10": [4.0, 3.5],
        "a_gf_10": [3.0, 4.2],
        "d_gf_10": [1.0, -0.7],
        "h_rest_days": [2.0, 4.0],
        "a_rest_days": [3.0, 3.0],
        "d_rest_days": [-1.0, 1.0],
        "h_bat_avg_10": [0.27, 0.25],
        "a_bat_avg_10": [0.24, 0.26],
        "d_bat_avg_10": [0.03, -0.01],
        "hs_era": [2.5, 3.0],
        "as_era": [3.5, 3.2],
        "starter_x_quality_proxy": [1.2, 0.2],
        "bullpen_fatigue_diff": [0.4, -0.1],
        "h_bp_era_10": [3.0, 4.0],
        "a_bp_era_10": [4.0, 3.5],
        "h_lineup_obp": [0.34, 0.33],
        "a_lineup_obp": [0.31, 0.35],
        "weather_temp_c": [21.0, 18.0],
        "weather_run_signal": [0.2, -0.1],
        "unrelated_context": [7.0, 8.0],
    })


def test_variants_registered():
    assert "BASELINE_TEAM_STATE" in available_variants()
    assert "TEAM_PLUS_STARTER" in available_variants()
    assert "TEAM_PLUS_BULLPEN" in available_variants()
    assert "TEAM_PLUS_LINEUP_PIT_SAFE" in available_variants()
    assert "TEAM_PLUS_WEATHER_PIT_SAFE" in available_variants()
    assert "FULL_VALIDATED_ENSEMBLE" in available_variants()


def test_baseline_excludes_conditional_and_unrelated():
    frame, meta = select_features(sample_frame(), "BASELINE_TEAM_STATE")
    assert "hs_era" not in frame
    assert "h_lineup_obp" not in frame
    assert "weather_temp_c" not in frame
    assert "unrelated_context" not in frame
    assert meta.feature_count == len(frame.columns)
    assert meta.feature_schema_hash


def test_starter_adds_starter_family():
    baseline, _ = select_features(sample_frame(), "BASELINE_TEAM_STATE")
    starter, _ = select_features(sample_frame(), "TEAM_PLUS_STARTER")
    assert set(baseline.columns) < set(starter.columns)
    assert "hs_era" in starter
    assert "starter_x_quality_proxy" in starter


def test_bullpen_and_context_are_separate():
    bullpen, _ = select_features(sample_frame(), "TEAM_PLUS_BULLPEN")
    lineup, _ = select_features(sample_frame(), "TEAM_PLUS_LINEUP_PIT_SAFE")
    weather, _ = select_features(sample_frame(), "TEAM_PLUS_WEATHER_PIT_SAFE")
    assert "h_bp_era_10" in bullpen
    assert "h_lineup_obp" in lineup
    assert "weather_temp_c" not in lineup
    assert "weather_temp_c" in weather
    assert "h_bp_era_10" not in weather


def test_full_variant_preserves_order():
    source = sample_frame()
    selected, meta = select_features(source, "FULL_VALIDATED_ENSEMBLE")
    assert list(selected.columns) == list(source.columns)
    assert meta.feature_count == len(source.columns)


def test_duplicate_columns_fail_closed():
    source = sample_frame()
    source["home_adv_dup"] = source["home_adv"]
    source = source.rename(columns={"home_adv_dup": "home_adv"})
    with pytest.raises(ValueError):
        select_features(source, "BASELINE_TEAM_STATE")


def test_summaries_are_reproducible():
    source = sample_frame()
    first = summarize_sets(source)
    second = summarize_sets(source)
    assert first == second
