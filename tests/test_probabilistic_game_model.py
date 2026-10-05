import numpy as np
import pytest

from research.probabilistic_game_model import (
    Scenario,
    assert_pit_ready,
    build_prediction_contract,
    model_disagreement_summary,
    mix_probability_vectors,
    normal_normal_update,
    posterior_predictive_run,
    uncertainty_decomposition,
)


def test_normal_normal_update_shrinks_small_sample_and_moves_with_more_data():
    small = normal_normal_update(
        prior_mean=0.0,
        prior_variance=1.0,
        observed_mean=2.0,
        observed_variance=1.0,
        observation_count=1,
    )
    large = normal_normal_update(
        prior_mean=0.0,
        prior_variance=1.0,
        observed_mean=2.0,
        observed_variance=1.0,
        observation_count=20,
    )
    assert 0.0 < small.mean < 2.0
    assert small.variance > large.variance
    assert abs(large.mean - 2.0) < abs(small.mean - 2.0)


def test_pit_gate_fails_closed_for_unknown_or_late_data():
    with pytest.raises(ValueError):
        assert_pit_ready(
            pit_status="UNVERIFIABLE",
            available_at="2026-10-05T17:00:00+09:00",
            prediction_cutoff="2026-10-05T18:00:00+09:00",
        )
    with pytest.raises(ValueError):
        assert_pit_ready(
            pit_status="PASS",
            available_at="2026-10-05T19:00:00+09:00",
            prediction_cutoff="2026-10-05T18:00:00+09:00",
        )


def test_probability_mix_is_coherent():
    mixed = mix_probability_vectors(
        [[0.8, 0.2], [0.2, 0.8]],
        [0.75, 0.25],
    )
    assert np.allclose(mixed, [0.65, 0.35])
    assert np.isclose(mixed.sum(), 1.0)


def test_uncertainty_decomposition_reconstructs_total_variance():
    out = uncertainty_decomposition(
        scenario_probabilities=[0.9, 0.5],
        scenario_weights=[0.7, 0.3],
    )
    assert out["aleatoric_variance"] >= 0
    assert out["epistemic_scenario_variance"] >= 0
    assert out["variance_reconstruction_error"] < 1e-12


def test_model_disagreement_is_zero_for_identical_models():
    out = model_disagreement_summary([[0.6, 0.4], [0.6, 0.4]])
    assert out["max_component_sd"] == 0.0
    assert out["max_pairwise_js"] == 0.0


def test_posterior_predictive_mixes_explicit_scenarios_and_exposes_mc():
    def simulator(scenario, rng, n):
        if scenario.scenario_id == "A":
            return np.column_stack([
                np.ones(n, dtype=int) * 5,
                np.ones(n, dtype=int) * 2,
            ])
        return np.column_stack([
            np.ones(n, dtype=int) * 2,
            np.ones(n, dtype=int) * 5,
        ])

    out = posterior_predictive_run(
        [
            Scenario("A", 0.75, {"starter": "A"}),
            Scenario("B", 0.25, {"starter": "B"}),
        ],
        simulate_fn=simulator,
        simulations=1000,
        seed=123,
        target="NPB",
        pit_status="PASS",
        available_at="2026-10-05T17:00:00+09:00",
        prediction_cutoff="2026-10-05T18:00:00+09:00",
    )
    assert out["status"] == "RESEARCH_ONLY"
    assert out["production_eligible"] is False
    assert np.isclose(out["probabilities"]["home_win"], 0.75)
    assert np.isclose(out["probabilities"]["away_win"], 0.25)
    assert np.isclose(out["probabilities"]["draw"], 0.0)
    assert out["uncertainty"]["epistemic_scenario_variance"] > 0
    assert out["mc"]["sample_count"] == 1000


def test_mlb_target_rejects_tied_terminal_score_without_explicit_extra_innings():
    def simulator(_scenario, _rng, n):
        return np.column_stack([np.zeros(n, dtype=int), np.zeros(n, dtype=int)])

    with pytest.raises(ValueError, match="extra-inning"):
        posterior_predictive_run(
            [Scenario("tied", 1.0)],
            simulate_fn=simulator,
            simulations=200,
            seed=1,
            target="MLB",
            pit_status="PASS",
            available_at="2026-10-05T17:00:00+09:00",
            prediction_cutoff="2026-10-05T18:00:00+09:00",
        )


def test_prediction_contract_requires_prediction_after_cutoff():
    with pytest.raises(ValueError):
        build_prediction_contract(
            game_id="g1",
            target="NPB",
            prediction_time="2026-10-05T17:59:59+09:00",
            prediction_cutoff="2026-10-05T18:00:00+09:00",
            available_at="2026-10-05T17:30:00+09:00",
            pit_status="PASS",
        )
