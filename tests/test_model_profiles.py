from __future__ import annotations

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.model_profiles import MODEL_PROFILES, apply_model_profile

def test_model_profiles_are_explicit_and_deterministic():
    assert MODEL_PROFILES == (
        "BALANCED", "ROBUST", "SMOOTH", "DEEP", "LOCAL", "REGULARIZED"
    )

def test_balanced_profile_leaves_estimators_unchanged():
    model = Pipeline([("scale", StandardScaler()), ("m", LogisticRegression(C=0.5, max_iter=100))])
    apply_model_profile({"Logistic": model}, "BALANCED")
    assert model.named_steps["m"].C == 0.5

def test_regularized_profile_changes_linear_regularization():
    model = Pipeline([("scale", StandardScaler()), ("m", LogisticRegression(C=0.5, max_iter=100))])
    apply_model_profile({"Logistic": model}, "REGULARIZED")
    assert model.named_steps["m"].C == 0.05

def test_unknown_model_profile_fails_closed():
    with pytest.raises(ValueError, match="unknown model profile"):
        apply_model_profile({}, "NOT_REAL")
