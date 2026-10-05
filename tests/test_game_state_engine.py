from __future__ import annotations

import numpy as np
import pandas as pd

from research.game_state_engine import (
    PIT_STATUS,
    TransitionKernel,
    _transition,
    canonicalize_pbp_frame,
    fit_transition_kernel,
    simulate_game,
)


def _frame(n_games: int = 6) -> pd.DataFrame:
    rows = []
    for game in range(n_games):
        home_score = away_score = 0
        for inning in range(1, 4):
            for half in ("T", "B"):
                for outs in (0, 1, 2):
                    if inning == 2 and half == "T" and outs == 2:
                        away_score += 1
                    if inning == 3 and half == "B" and outs == 1:
                        home_score += 1
                    rows.append(
                        {
                            "game_id": f"g{game}",
                            "inning": inning,
                            "half": half,
                            "play_id": len(rows) + 1,
                            "game_date": f"2024-01-{game + 1:02d}",
                            "home_team_name": "H",
                            "away_team_name": "A",
                            "home_total_runs": home_score,
                            "away_total_runs": away_score,
                            "addedRuns": int(
                                (inning == 2 and half == "T" and outs == 2)
                                or (inning == 3 and half == "B" and outs == 1)
                            ),
                            "outs_when_up": outs,
                            "on_1b": 0,
                            "on_2b": 0,
                            "on_3b": 0,
                        }
                    )
    return pd.DataFrame(rows)


def test_missing_state_is_dropped_not_imputed():
    raw = _frame(1)
    raw.loc[0, "outs_when_up"] = np.nan
    assert len(canonicalize_pbp_frame(raw)) == 17


def test_top_before_bottom():
    normalized = canonicalize_pbp_frame(_frame(1))
    inning_one = normalized[normalized.inning == 1]["half"].tolist()
    assert inning_one[0] == "T"
    assert inning_one[-1] == "B"


def test_reconstructed_state_rejects_inconsistent_final_score_label():
    raw = _frame(1)
    raw["home_total_runs"] = 9
    raw["away_total_runs"] = 8
    normalized = canonicalize_pbp_frame(raw)
    assert normalized.empty

def test_score_reversal_is_rejected():
    current = {
        "inning": 1,
        "half": "T",
        "outs": 1,
        "home_score": 1,
        "away_score": 0,
        "state_home_score": 1,
        "state_away_score": 0,
        "on_1b": 0,
        "on_2b": 0,
        "on_3b": 0,
    }
    nxt = {**current, "outs": 2, "state_home_score": 0}
    assert _transition(current, nxt) is None


def test_simulation_is_reproducible_and_normalized():
    kernel = fit_transition_kernel(_frame(), min_transitions=10)
    first = simulate_game(
        kernel,
        base_run=3.5,
        home_factor=1.05,
        away_factor=0.95,
        simulations=50,
        seed=4,
        max_innings=3,
    )
    second = simulate_game(
        kernel,
        base_run=3.5,
        home_factor=1.05,
        away_factor=0.95,
        simulations=50,
        seed=4,
        max_innings=3,
    )
    assert first == second
    probabilities = first["probabilities"]
    assert abs(
        sum(probabilities[key] for key in ("home_win", "draw", "away_win")) - 1.0
    ) < 1e-9


def test_sampling_keeps_observed_support():
    no_run = ("T", 1, 0, 0, "N")
    home_run = ("T", 1, 0, 1, "H")
    kernel = TransitionKernel(
        {(1, "T", 0, 0, 0): {no_run: 5, home_run: 5}},
        {(1, "T", 0, 0): {no_run: 5, home_run: 5}},
        {"T": {no_run: 5, home_run: 5}},
        10,
    )
    rng = np.random.default_rng(1)
    draws = {
        kernel.sample((1, "T", 0, 0, 0), rng, {"H": 1.4, "A": 0.7})
        for _ in range(100)
    }
    assert draws <= {no_run, home_run}


def test_pit_status_is_not_production_eligible():
    assert PIT_STATUS.startswith("UNVERIFIABLE")