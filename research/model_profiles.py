"""Deterministic research-only model hyperparameter profiles.

Profiles are activated only through BASEBALL_MODEL_PROFILE in research jobs.
The Production default remains BALANCED.
"""
from __future__ import annotations

from typing import Any, Mapping

MODEL_PROFILES = (
    "BALANCED",
    "ROBUST",
    "SMOOTH",
    "DEEP",
    "LOCAL",
    "REGULARIZED",
)

PROFILE_PARAMS: Mapping[str, Mapping[str, Mapping[str, Any]]] = {
    "BALANCED": {},
    "ROBUST": {
        "Logistic": {"m__C": 0.20},
        "HistGB": {"learning_rate": 0.025, "max_leaf_nodes": 11, "min_samples_leaf": 20, "l2_regularization": 4.0},
        "RandomForest": {"max_depth": 8, "min_samples_leaf": 10, "max_features": 0.45},
        "ExtraTrees": {"max_depth": 10, "min_samples_leaf": 8, "max_features": 0.55},
        "KNNAnalog": {"m__n_neighbors": 65, "m__p": 2},
        "LightGBM": {"learning_rate": 0.018, "num_leaves": 11, "min_child_samples": 25, "reg_lambda": 4.0},
        "XGBoost": {"max_depth": 3, "min_child_weight": 12, "learning_rate": 0.02, "reg_lambda": 4.0},
        "CatBoost": {"depth": 5, "learning_rate": 0.025, "l2_leaf_reg": 8.0},
    },
    "SMOOTH": {
        "Logistic": {"m__C": 0.10},
        "HistGB": {"learning_rate": 0.020, "max_leaf_nodes": 7, "min_samples_leaf": 28, "l2_regularization": 8.0},
        "RandomForest": {"max_depth": 6, "min_samples_leaf": 14, "max_features": 0.40},
        "ExtraTrees": {"max_depth": 8, "min_samples_leaf": 12, "max_features": 0.50},
        "KNNAnalog": {"m__n_neighbors": 90, "m__p": 2},
        "LightGBM": {"learning_rate": 0.015, "num_leaves": 7, "min_child_samples": 35, "reg_lambda": 6.0},
        "XGBoost": {"max_depth": 2, "min_child_weight": 18, "learning_rate": 0.018, "reg_lambda": 6.0},
        "CatBoost": {"depth": 4, "learning_rate": 0.02, "l2_leaf_reg": 10.0},
    },
    "DEEP": {
        "Logistic": {"m__C": 1.20},
        "HistGB": {"learning_rate": 0.050, "max_leaf_nodes": 31, "min_samples_leaf": 7, "l2_regularization": 0.5},
        "RandomForest": {"max_depth": 16, "min_samples_leaf": 3, "max_features": 0.80},
        "ExtraTrees": {"max_depth": 18, "min_samples_leaf": 2, "max_features": 0.85},
        "KNNAnalog": {"m__n_neighbors": 25, "m__p": 2},
        "LightGBM": {"learning_rate": 0.035, "num_leaves": 31, "min_child_samples": 12, "reg_lambda": 1.0},
        "XGBoost": {"max_depth": 6, "min_child_weight": 4, "learning_rate": 0.04, "reg_lambda": 1.0},
        "CatBoost": {"depth": 8, "learning_rate": 0.04, "l2_leaf_reg": 3.0},
    },
    "LOCAL": {
        "Logistic": {"m__C": 0.35},
        "HistGB": {"learning_rate": 0.030, "max_leaf_nodes": 15, "min_samples_leaf": 10, "l2_regularization": 2.0},
        "RandomForest": {"max_depth": 12, "min_samples_leaf": 5, "max_features": 0.60},
        "ExtraTrees": {"max_depth": 13, "min_samples_leaf": 4, "max_features": 0.70},
        "KNNAnalog": {"m__n_neighbors": 15, "m__weights": "distance", "m__p": 1},
        "LightGBM": {"learning_rate": 0.027, "num_leaves": 19, "min_child_samples": 16, "reg_lambda": 2.0},
        "XGBoost": {"max_depth": 4, "min_child_weight": 6, "learning_rate": 0.028, "reg_lambda": 2.0},
        "CatBoost": {"depth": 6, "learning_rate": 0.032, "l2_leaf_reg": 5.0},
    },
    "REGULARIZED": {
        "Logistic": {"m__C": 0.05},
        "HistGB": {"learning_rate": 0.023, "max_leaf_nodes": 9, "min_samples_leaf": 22, "l2_regularization": 6.0},
        "RandomForest": {"max_depth": 7, "min_samples_leaf": 12, "max_features": 0.35},
        "ExtraTrees": {"max_depth": 9, "min_samples_leaf": 10, "max_features": 0.45},
        "KNNAnalog": {"m__n_neighbors": 75, "m__p": 2},
        "LightGBM": {"learning_rate": 0.017, "num_leaves": 9, "min_child_samples": 30, "reg_alpha": 0.5, "reg_lambda": 7.0},
        "XGBoost": {"max_depth": 2, "min_child_weight": 16, "learning_rate": 0.02, "reg_alpha": 0.5, "reg_lambda": 7.0},
        "CatBoost": {"depth": 4, "learning_rate": 0.022, "l2_leaf_reg": 12.0},
    },
}


def apply_model_profile(models: dict[str, Any], profile: str) -> dict[str, Any]:
    chosen = str(profile or "BALANCED").strip().upper()
    if chosen not in MODEL_PROFILES:
        raise ValueError(
            f"unknown model profile {chosen!r}; allowed={','.join(MODEL_PROFILES)}"
        )
    for name, params in PROFILE_PARAMS[chosen].items():
        model = models.get(name)
        if model is None or not params:
            continue
        try:
            model.set_params(**dict(params))
        except Exception as exc:
            raise ValueError(
                f"cannot apply model profile {chosen} to {name}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    return models
