"""Compatibility tests for the Baseball v4.4 bridge.

The test intentionally exercises the real BaseballBacktest.fit_ensemble path.
The bridge must delegate to that same implementation without changing the
numerical/model-selection result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.engine import BaseballResearchEngine


def test_research_engine_bridge_fit_ensemble_is_compatible() -> None:
    rng = np.random.RandomState(42)
    n = 180
    X = pd.DataFrame(
        rng.normal(size=(n, 8)),
        columns=[f"feature_{i}" for i in range(8)],
    )
    # Deterministic, non-degenerate binary target for MLB model selection.
    y = (X["feature_0"] + 0.35 * X["feature_1"] - 0.15 * X["feature_2"] > 0).astype(int)

    engine = BaseballResearchEngine()
    result = engine.fit_ensemble_compatibility(X, y, "MLB")

    assert result.passed, result.details
    assert result.details["score_equal"] is True
    assert result.details["best_model_equal"] is True
    assert result.details["model_names_equal"] is True
    assert result.details["bridge_audit_tail"]
    assert result.details["bridge_audit_tail"][0]["delegated_to_existing_engine"] is True
