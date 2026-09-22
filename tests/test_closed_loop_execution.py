import numpy as np
import pandas as pd
import pytest

from research.closed_loop_execute import development_candidate_id, hilo_probs


def test_development_candidate_id_is_deterministic_and_development_only():
    development = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "datetime": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
            "pred_home": [0.6, 0.4],
            "pred_away": [0.4, 0.6],
            "actual": [0, 1],
            "model": ["m", "m"],
            "input_fingerprint": ["a", "b"],
        }
    )
    cid1 = development_candidate_id(
        development,
        league="MLB",
        temperature=1.2,
        probability_source="raw_classifier_strict",
    )
    cid2 = development_candidate_id(
        development.copy(),
        league="MLB",
        temperature=1.2,
        probability_source="raw_classifier_strict",
    )
    assert cid1 == cid2
    assert cid1.startswith("tempcal-v2-")

    # Holdout rows are intentionally excluded from the identity input.
    holdout_a = pd.concat(
        [development, pd.DataFrame({"game_id": ["h1"], "pred_home": [0.9], "pred_away": [0.1]})],
        ignore_index=True,
    )
    holdout_b = pd.concat(
        [development, pd.DataFrame({"game_id": ["h2"], "pred_home": [0.1], "pred_away": [0.9]})],
        ignore_index=True,
    )
    assert development_candidate_id(
        holdout_a.iloc[:2].copy(),
        league="MLB",
        temperature=1.2,
        probability_source="raw_classifier_strict",
    ) == development_candidate_id(
        holdout_b.iloc[:2].copy(),
        league="MLB",
        temperature=1.2,
        probability_source="raw_classifier_strict",
    )


def test_hilo_probs_fails_closed_on_missing_or_invalid_canonical_columns():
    with pytest.raises(RuntimeError, match="canonical OOS artifact"):
        hilo_probs(pd.DataFrame({"lambda_home": [2.0], "lambda_away": [2.0]}))

    invalid = pd.DataFrame({"low": [0.9], "high": [0.9]})
    with pytest.raises(RuntimeError, match="synthetic probability reconstruction"):
        hilo_probs(invalid)

    valid = pd.DataFrame({"low": [0.75, 0.25], "high": [0.25, 0.75]})
    out = hilo_probs(valid)
    assert out.shape == (2, 2)
    assert np.allclose(out.sum(axis=1), 1.0)
