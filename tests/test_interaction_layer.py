from research.interaction_layer import add_validated_interactions, normalized_product


def test_normalized_product_is_bounded_and_finite():
    assert normalized_product(1000, 1000, 1, 1) == 5.0
    assert normalized_product(-1000, 1000, 1, 1) == -5.0
    assert normalized_product(float('nan'), 2, 1, 1) == 0.0


def test_interactions_are_deterministic_and_do_not_mutate_input():
    base = {
        "starter_x_quality_proxy": 1.5,
        "a_bat_so_rate_10": 0.22,
        "h_bat_so_rate_10": 0.18,
        "starter_kbb_gap": 2.0,
        "a_lineup_contact_rate": 0.72,
        "h_lineup_contact_rate": 0.68,
        "offense_power_gap_10": 0.03,
        "as_hr9": 1.1,
        "hs_hr9": 0.9,
        "h_bat_bb_rate_10": 0.09,
        "a_bat_bb_rate_10": 0.07,
        "as_bb9": 3.1,
        "hs_bb9": 2.7,
        "h_bat_so_rate_10": 0.19,
        "a_bat_so_rate_10": 0.22,
        "as_k9": 8.1,
        "hs_k9": 7.2,
        "bullpen_fatigue_diff": 1.2,
        "expected_env": 4.5,
        "h_bat_iso_proxy_10": 0.16,
        "a_bat_iso_proxy_10": 0.13,
        "weather_wind_kmh": 12.0,
        "weather_temp_c": 24.0,
        "h_gd_10": 0.7,
        "a_gd_10": -0.2,
        "run_volatility_gap_20": 0.4,
    }
    original = dict(base)
    out1 = add_validated_interactions(base)
    out2 = add_validated_interactions(base)
    assert base == original
    assert out1 == out2
    assert len(out1) == len(base) + 17
    assert all(-5.0 <= float(out1[k]) <= 5.0 for k in out1 if k.startswith("ix_"))
