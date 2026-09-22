import numpy as np
from sklearn.linear_model import LogisticRegression

from research.feature_influence_stability import permutation_importance_by_window


def _fit(X, y, seed):
    return LogisticRegression(max_iter=300, random_state=seed).fit(X, y)


def _predict(model, X):
    return model.predict_proba(X)


def test_feature_influence_is_stable_for_persistent_signal():
    rng = np.random.default_rng(321)
    X = rng.normal(size=(360, 4))
    y = (X[:, 0] - 0.25 * X[:, 1] > 0).astype(int)
    windows = [
        ("w1", X[:180], y[:180], X[180:260], y[180:260]),
        ("w2", X[:260], y[:260], X[260:340], y[260:340]),
    ]
    report = permutation_importance_by_window(
        windows=windows,
        feature_names=("signal", "weak", "noise1", "noise2"),
        fit_model=_fit,
        predict_proba=_predict,
        repeats=4,
        seed=42,
    )
    assert report.status == "STABLE"
    assert report.top_feature_frequency["signal"] == 1.0


def test_feature_influence_rejects_duplicate_feature_names():
    rng = np.random.default_rng(654)
    X = rng.normal(size=(80, 2))
    y = (X[:, 0] > 0).astype(int)
    try:
        permutation_importance_by_window(
            windows=[("w", X[:50], y[:50], X[50:], y[50:])],
            feature_names=("a", "a"),
            fit_model=_fit,
            predict_proba=_predict,
            repeats=2,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected duplicate-name validation failure")
