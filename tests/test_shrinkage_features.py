from collections import deque

import numpy as np

import baseball_backtest


def _state_for(matches=10, wins=10):
    s = baseball_backtest.TeamState()
    s.total_matches = matches
    results = [0] * wins + [2] * max(0, matches - wins)
    s.results.extend(results[-60:])
    return s


def test_small_sample_win_rate_is_shrunk_toward_half():
    bt = baseball_backtest.BaseballBacktest.__new__(baseball_backtest.BaseballBacktest)
    bt.states = {("MLB", "X"): _state_for(matches=3, wins=3)}
    bt.elo_ratings = {}
    f = bt._team_features("MLB", "X", "home", np.datetime64("2026-09-22"))
    assert 0.50 < f["win_shrunk_3"] < f["win_3"]
    assert abs(f["win_shrunk_3"] - (3 + 10) / 23) < 1e-12


def test_shrinkage_never_uses_future_and_draw_feature_is_npb_only():
    bt = baseball_backtest.BaseballBacktest.__new__(baseball_backtest.BaseballBacktest)
    bt.states = {("NPB", "X"): _state_for(matches=10, wins=8)}
    bt.elo_ratings = {}
    f = bt._team_features("NPB", "X", "home", np.datetime64("2026-09-22"))
    assert "draw_shrunk_10" in f
    mlb = bt._team_features("MLB", "X", "home", np.datetime64("2026-09-22"))
    assert "draw_shrunk_10" not in mlb
