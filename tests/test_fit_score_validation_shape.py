import numpy as np
import pandas as pd

import baseball_backtest as bb


class _ShapeCheckingRegressor:
    """Tiny deterministic regressor that rejects Series validation input."""

    def __init__(self, *args, **kwargs):
        self.fitted = False

    def fit(self, X, y, sample_weight=None):
        assert isinstance(X, pd.DataFrame), "training features must remain 2D"
        assert len(X) == len(y)
        self.fitted = True
        return self

    def predict(self, X):
        assert isinstance(X, pd.DataFrame), "validation features collapsed to Series"
        assert self.fitted
        return np.full(len(X), 1.5, dtype=float)


def test_fit_score_ensemble_keeps_validation_features_2d(monkeypatch):
    """Regression test for the historical X.iloc[va] Series-shape bug."""
    monkeypatch.setenv("BASEBALL_FAST_OOS", "1")
    monkeypatch.setenv("BASEBALL_SCORE_FAST_VALIDATION", "1")
    monkeypatch.setenv("BASEBALL_SCORE_TREE_ESTIMATORS", "1")
    monkeypatch.setenv("BASEBALL_SCORE_HIST_MAX_ITER", "1")

    # All score-model factories are replaced by the same shape-checking stub.
    monkeypatch.setattr(bb, "PoissonRegressor", _ShapeCheckingRegressor)
    monkeypatch.setattr(bb, "TweedieRegressor", _ShapeCheckingRegressor)
    monkeypatch.setattr(bb, "HistGradientBoostingRegressor", _ShapeCheckingRegressor)
    monkeypatch.setattr(bb, "RandomForestRegressor", _ShapeCheckingRegressor)
    monkeypatch.setattr(bb, "ExtraTreesRegressor", _ShapeCheckingRegressor)

    model = bb.BaseballBacktest()
    X = pd.DataFrame(
        {
            "feature_a": np.linspace(0.0, 1.0, 130),
            "feature_b": np.sin(np.linspace(0.0, 3.0, 130)),
        }
    )
    y_home = np.full(130, 2.0)
    y_away = np.full(130, 1.0)

    result = model.fit_score_ensemble(X, y_home, y_away, "MLB")

    assert result is not None
    assert len(result["models"]) == 3
