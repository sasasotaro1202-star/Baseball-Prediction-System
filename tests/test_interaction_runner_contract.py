from research.interaction_layer import add_validated_interactions


def test_candidate_layer_adds_only_prefixed_interactions():
    base = {
        "starter_x_quality_proxy": 1.0,
        "a_bat_so_rate_10": 0.20,
        "h_bat_so_rate_10": 0.20,
        "starter_kbb_gap": 1.0,
        "a_lineup_contact_rate": 0.70,
        "h_lineup_contact_rate": 0.70,
        "offense_power_gap_10": 0.02,
        "as_hr9": 1.0,
        "hs_hr9": 1.0,
        "h_bat_bb_rate_10": 0.08,
        "a_bat_bb_rate_10": 0.08,
        "as_bb9": 3.0,
        "hs_bb9": 3.0,
        "as_k9": 7.5,
        "hs_k9": 7.5,
        "bullpen_fatigue_diff": 0.5,
        "expected_env": 4.0,
        "h_bat_iso_proxy_10": 0.13,
        "a_bat_iso_proxy_10": 0.13,
        "weather_wind_kmh": 10.0,
        "weather_temp_c": 20.0,
        "h_gd_10": 0.5,
        "a_gd_10": -0.2,
        "run_volatility_gap_20": 0.3,
    }
    out = add_validated_interactions(base)
    assert set(out) >= set(base)
    interaction_keys = [k for k in out if k.startswith("ix_")]
    assert len(interaction_keys) == 17
    assert all(-5.0 <= float(out[k]) <= 5.0 for k in interaction_keys)
