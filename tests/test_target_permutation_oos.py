import numpy as np
from sklearn.linear_model import LogisticRegression

from research.target_permutation_oos import audit_target_permutation


def _fit_predict(X_train, y_train, X_eval, seed):
    model = LogisticRegression(max_iter=300, random_state=seed)
    model.fit(X_train, y_train)
    return model.predict_proba(X_eval)


def test_target_permutation_separates_honest_signal():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(320, 4))
    y = (X[:, 0] + 0.25 * X[:, 1] > 0).astype(int)
    report = audit_target_permutation(
        X_train=X[:220], y_train=y[:220],
        X_eval=X[220:], y_eval=y[220:],
        fit_predict=_fit_predict,
        seeds=(7, 19, 43, 71, 101, 137, 181, 223),
    )
    assert report.status == "SEPARATED"
    assert not report.risk_flag


def test_target_permutation_confirms_target_bearing_feature_has_real_signal():
    rng = np.random.default_rng(456)
    y = rng.integers(0, 2, size=320)
    X = rng.normal(size=(320, 3))
    X[:, 0] = y
    report = audit_target_permutation(
        X_train=X[:220], y_train=y[:220],
        X_eval=X[220:], y_eval=y[220:],
        fit_predict=_fit_predict,
        seeds=(7, 19, 43, 71, 101, 137, 181, 223),
    )
    # The feature is literally the realized target. Shuffling training labels
    # should destroy the learned relationship, so the correct diagnostic is
    # strong real-vs-null separation rather than a leak flag.
    assert report.status == "SEPARATED"
    assert not report.risk_flag
    assert report.separation["LogLoss_null_mean_minus_real"] > 0
    assert report.separation["Brier_null_mean_minus_real"] > 0


def test_target_permutation_requires_enough_seeds():
    rng = np.random.default_rng(789)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] > 0).astype(int)
    try:
        audit_target_permutation(
            X_train=X[:50], y_train=y[:50],
            X_eval=X[50:], y_eval=y[50:],
            fit_predict=_fit_predict,
            seeds=(7,),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected seed-count validation failure")
