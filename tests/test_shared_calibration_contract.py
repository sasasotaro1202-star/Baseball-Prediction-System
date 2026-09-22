import numpy as np

import baseball_backtest
import research.npb_candidate_replay as npb_candidate


class _Cal:
    def __init__(self, temperature):
        self.temperature = temperature


def test_npb_candidate_delegates_to_shared_temperature_contract(monkeypatch):
    called = {}

    def fake_fit_temperature(probabilities, y_true):
        called["shape"] = (probabilities.shape, y_true.shape)
        return _Cal(4.25)

    monkeypatch.setattr(npb_candidate, "fit_temperature", fake_fit_temperature)
    p = np.array([[0.6, 0.1, 0.3], [0.2, 0.5, 0.3]], dtype=float)
    y = np.array([0, 2], dtype=int)

    assert npb_candidate._fit_temperature(y, p) == 4.25
    assert called["shape"] == ((2, 3), (2,))


def test_core_temperature_helper_delegates_to_shared_contract(monkeypatch):
    called = {}

    def fake_fit_temperature(probabilities, y_true):
        called["shape"] = (probabilities.shape, y_true.shape)
        return _Cal(5.5)

    monkeypatch.setattr(baseball_backtest, "fit_temperature", fake_fit_temperature)
    bt = baseball_backtest.BaseballBacktest.__new__(baseball_backtest.BaseballBacktest)
    p = np.tile(np.array([[0.7, 0.3], [0.4, 0.6]], dtype=float), (13, 1))[:25]
    y = np.tile(np.array([0, 1], dtype=int), 13)[:25]

    assert bt._temperature_from_probs(p, y) == 5.5
    assert called["shape"] == ((25, 2), (25,))
