import numpy as np
import pandas as pd

from baseball_backtest import BaseballBacktest


def test_fast_oos_model_profile(monkeypatch):
    monkeypatch.setenv("BASEBALL_FAST_OOS", "1")
    bt = BaseballBacktest()
    models = bt.models("MLB")

    assert models["HistGB"].max_iter == 180
    assert models["RandomForest"].n_estimators == 180
    assert models["ExtraTrees"].n_estimators == 180
    if "LightGBM" in models:
        assert models["LightGBM"].n_estimators == 180
    if "XGBoost" in models:
        assert models["XGBoost"].n_estimators == 120
    if "CatBoost" in models:
        assert models["CatBoost"].get_params()["iterations"] == 120


def test_fit_model_caps_to_recent_rows(monkeypatch):
    monkeypatch.setenv("BASEBALL_MAX_FIT_ROWS", "3")
    bt = BaseballBacktest()
    seen = {}

    class CaptureModel:
        def fit(self, X, y, sample_weight=None):
            seen["rows"] = len(X)
            seen["y"] = list(y)
            seen["weights"] = list(sample_weight) if sample_weight is not None else None
            return self

    X = pd.DataFrame({"x": np.arange(5, dtype=float)})
    y = np.arange(5)
    weights = np.arange(10, 15, dtype=float)
    bt._fit_model(CaptureModel(), X, y, weights, "MLB")

    assert seen["rows"] == 3
    assert seen["y"] == [2, 3, 4]
    assert seen["weights"] == [12.0, 13.0, 14.0]


def test_invalid_model_env_fails_closed(monkeypatch):
    monkeypatch.setenv("BASEBALL_XGB_ESTIMATORS", "not-an-int")
    bt = BaseballBacktest()
    try:
        bt.models("MLB")
    except ValueError as exc:
        assert "BASEBALL_XGB_ESTIMATORS" in str(exc)
    else:
        raise AssertionError("invalid model configuration was silently accepted")


def test_configurable_hard_cap(monkeypatch):
    monkeypatch.setenv("BASEBALL_TIME_BUDGET_SEC", "7200")
    monkeypatch.setenv("BASEBALL_HARD_CAP_SEC", "12600")
    bt = BaseballBacktest()
    assert bt.time_budget_sec == 7200.0


def test_hard_cap_limits_requested_budget(monkeypatch):
    monkeypatch.setenv("BASEBALL_TIME_BUDGET_SEC", "7200")
    monkeypatch.setenv("BASEBALL_HARD_CAP_SEC", "5400")
    bt = BaseballBacktest()
    assert bt.time_budget_sec == 5400.0
